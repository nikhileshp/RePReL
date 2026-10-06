# RePReL rebuild — design specification (M0)

Status: proposed, awaiting approval. Companion: `2026-10-06-m0-original-reprel-summary.md`.

## 0. Understanding of the brief

Goal: a clean, modular Python package that reproduces RePReL (Kokel et al., ICAPS 2021) with an
HTN planner on top and tabular Q-learning per operator below, driven by hand-written D-FOCI
abstractions. Milestone 1 reproduces the paper's qualitative results on Taxi; later milestones
add Office and Box World, then NDR-based structure learning of D-FOCI edges from logged
exploration transitions. The interfaces must let (a) learned abstractions and (b) multi-agent
settings drop in without touching other modules, but neither is built now.

Success criteria for M5: on Taxi, learning curves (mean ± std over ≥ 5 seeds, return and
success rate vs. environment steps) show D-FOCI RePReL > no-abstraction RePReL > flat Q-learning in
sample efficiency, and D-FOCI RePReL trained on 1–2 passengers transfers to 3–5 passengers.
Anything that contradicts the paper is reported, not tuned away.

Stated assumptions (correct me if wrong):
- Taxi has no fuel; the D-FOCI closure test uses a synthetic spec with a fuel statement to
  exercise the ancestor logic. The domain can gain fuel later as new atoms + one YAML statement.
- "Others from the original repo" for M6 = Box World. Craft World has no released code and is
  out of scope unless requested.
- Plain YAML + frozen dataclasses for config (the brief lists pyyaml and no Hydra); a Hydra
  front-end can wrap `RunConfig` later without changes elsewhere.
- Git branch `master` (your global convention), Conventional Commits, feature branch per
  milestone merged with `--no-ff`. No AI attribution anywhere in the repo or history.

## 1. Package layout

```
reprel/
  core/
    atoms.py          Obj, Atom, Literal, Substitution, unify(), is_variable()
    state.py          State (frozenset of ground atoms + typed objects): holds, objects_of_type,
                      substitute, lift, diff. Hashable, immutable.
    domain.py         Domain ABC + Transition + InstanceConfig; registry of domains
  domains/
    taxi.py           Relational multi-passenger Taxi (domain + HTN methods)
    office.py         M6
    boxworld.py       M6
  planning/
    operators.py      OperatorSpec, OperatorInstance, Goal
    planner.py        Planner ABC; PlanningFailure
    htn_gtpyhop.py    GTPyhopPlanner: builds a gtpyhop.Domain from OperatorSpecs + method callables
  abstraction/
    abstraction.py    Abstraction ABC + registry
    none.py           NoAbstraction (full ground state; the "no abstraction" baseline)
    dfoci.py          DFOCISpec (YAML I/O, validation), closure, DFOCIAbstraction
  agents/
    schedules.py      EpsilonSchedule (linear-per-episode, constant)
    q_learning.py     QLearningAgent (tabular, dict[key, ndarray])
    agent_pool.py     AgentPool: operator name -> agent; save/load
  execution/
    executor.py       RePReLExecutor: plan → run operator subtasks → replan on failure/timeout
    explore.py        ExplorationCollector: random policy inside subtasks, logs transitions only
    flat.py           FlatExecutor: single Q-learning agent on the full task
    evaluation.py     greedy evaluation episodes → EvalResult
  logging/
    transitions.py    TransitionRecord, TransitionLogger (Parquet; JSONL fallback), reader
    metrics.py        MetricsLogger (CSV), RunMetadata (config, git hash, seed, env info)
  structure/          M7: ndr_adapter.py, compare.py
  configs/            YAML experiment configs (taxi_flat.yaml, taxi_reprel_none.yaml,
                      taxi_reprel_dfoci.yaml, taxi_transfer.yaml, taxi_explore.yaml)
  dfoci/              taxi.yaml (hand), later office.yaml, boxworld.yaml
  scripts/            run.py, explore.py, plot.py, sweep.py (multi-seed launcher)
  config.py           RunConfig dataclasses + YAML loader + dotted overrides
  seeding.py          seed_everything(seed) -> numpy Generator
third_party/          gitignored clones (GLIB for M7)
tests/                one file per module
plan/                 design docs (this file)
```

Dependencies: numpy, pyyaml, pandas, pyarrow, matplotlib, gtpyhop (PyPI 2.0.2, Clear BSD).
Dev: pytest, ruff, mypy. Python ≥ 3.11 (`.python-version` = 3.12), managed by `uv`.

## 2. Core data model (`reprel/core`)

```python
@dataclass(frozen=True)
class Obj:            name: str; type: str
@dataclass(frozen=True)
class Atom:           pred: str; args: tuple[str, ...]      # args are object names or variables
@dataclass(frozen=True)
class Literal:        atom: Atom; positive: bool = True
Substitution = Mapping[str, str]                            # variable -> object name
```

- Variables are identifiers starting with an uppercase letter (`P`, `L1`), matching the paper.
  Objects are lowercase (`p1`, `taxi`, `l_3_4`).
- `State(atoms: frozenset[Atom], objects: frozenset[Obj])`:
  - `holds(atom)`, `objects_of_type(t)`, `type_of(name)`
  - `substitute(theta)` → new State with object names renamed (used for lifting)
  - `lift(binding: Mapping[role, obj])` → State with bound objects renamed to `?role`
  - `diff(other)` → (added, deleted) atom sets (used by transition logs and the NDR adapter)
  - `with_atoms(add=..., remove=...)` → new State (domains build successor states with this)
- `unify(pattern: Atom, ground: Atom, theta, types) -> Substitution | None` with type checks
  from the predicate signature. Pure functions; nothing global.
- Static atoms (walls) are ordinary atoms. They never change, so they cost nothing in Q-table
  cardinality within one layout, and they make the D-FOCI statements honest
  (`at(T,L2)` depends on `wall(L1,Dir)`).

## 3. Domain interface (`reprel/core/domain.py`)

```python
@dataclass(frozen=True)
class Transition: next_state: State; reward: float; done: bool; info: Mapping[str, Any]

class Domain(ABC):
    name: ClassVar[str]
    types: Mapping[str, str | None]            # type -> parent type (flat for now)
    predicates: Mapping[str, tuple[str, ...]]  # pred -> argument types
    actions: tuple[Action, ...]                # primitive actions; Action = str
    def reset(self, rng: np.random.Generator) -> State
    def step(self, state: State, action: Action, rng) -> Transition   # pure: no hidden state
    def goal(self, state: State) -> Goal                                  # literals the planner must achieve
    def is_success(self, state: State) -> bool
    max_steps: int                                                        # instance attribute from config
```

`step` is a pure function of `(state, action)`; episode step counting lives in the executor.
This is what makes transition logging exact, tests trivial, and a later multi-agent `step`
(joint action = tuple of actions, agents as objects of type `agent`) a new Domain subclass rather
than a change to the interface. Each domain has a frozen `*Config` dataclass (instance size,
layout, rewards, max_steps); the constructor takes `(config=None, **overrides)` so YAML
keyword config and programmatic configs both work. The `DOMAINS` registry maps names to classes.

### Taxi (M1)
- Objects: `taxi` (type `taxi`), `p1..pN` (`passenger`), `l_r_c` for every non-wall cell
  (`location`), `n s e w` (`dir`).
- Predicates: `at(thing, location)` where thing is taxi or passenger; `in(passenger, taxi)`;
  `dest(passenger, location)`; `delivered(passenger)`; `wall(location, dir)` static.
- Actions: `north south east west pickup dropoff`.
- Dynamics and rewards follow RePReL-domains: moves blocked by walls (no penalty), pickup
  succeeds iff taxi empty and some undelivered passenger is at the taxi cell (lowest index
  boards), dropoff iff carried passenger's destination is the taxi cell. Rewards step −0.1,
  pickup +10, drop +20, illegal pickup/drop −1; done when all delivered or `max_steps`
  (default 1000; per-instance override in config). All numbers in `TaxiConfig`.
- Instance generation: `num_passengers` is the size parameter; taxi at a random free
  non-depot cell (as in RePReL-domains);
  each passenger: pickup depot and distinct destination depot sampled from `{R,G,B,Y}`.
  Layouts: `eight` (paper) and `five` (classic 5x5), selectable by name.
- Goal: `{delivered(p) for all p}`.

## 4. Planning (`reprel/planning`)

```python
@dataclass(frozen=True)
class OperatorSpec:
    name: str
    params: tuple[tuple[str, str], ...]          # (role, type), e.g. (("P", "passenger"),)
    preconditions: tuple[Literal, ...]            # may use existential variables (L)
    add: tuple[Atom, ...]; delete: tuple[Atom, ...]
    termination: tuple[Literal, ...]              # β(o): conjunction that must hold
    terminal_reward: float                        # tR

@dataclass(frozen=True)
class OperatorInstance:
    spec: OperatorSpec; binding: Mapping[str, str]        # role -> object
    def is_terminated(self, state) -> bool
    def subtask_reward(self, env_reward, next_state) -> float   # env_reward (+ tR on termination)
    def key(self) -> tuple[str, tuple[str, ...]]              # ("pickup", ("p2",))

class Planner(ABC):
    def plan(self, state: State, goal: Goal) -> list[OperatorInstance]   # raises PlanningFailure
```

`GTPyhopPlanner(operators, methods, root_task)` converts OperatorSpecs into GTPyhop actions
automatically (check preconditions by unification, apply add/delete) and registers the
domain's task-method callables, which are written against our `State` (not GTPyhop's attribute
state), e.g. for Taxi:

```python
def m_transport_all(state, goal):   # first undelivered passenger
def m_transport_in_taxi(state, p):  # -> [("drop", p)]
def m_transport(state, p):          # -> [("pickup", p), ("drop", p)]
```

GTPyhop keeps a module-level `current_domain`; the wrapper sets it on every `plan()` call and
documents that planning is not thread-safe. Methods are defined once with variables, so the
planner is not tied to a fixed object count (fixes the original's per-index registration).
Planner tests check plan validity by simulating the operators' add/delete effects.

## 5. Abstraction (`reprel/abstraction`)

```python
class Abstraction(ABC):
    def abstract(self, state: State, op: OperatorInstance) -> Hashable
ABSTRACTIONS = {"none": NoAbstraction, "dfoci": DFOCIAbstraction}
```

- `NoAbstraction(include_binding=False)`: key = `state.atoms` (faithful to the original
  "trl"/no-abstraction baseline); with `include_binding=True` the operator binding is appended.
- `DFOCISpec` (YAML):

```yaml
domain: taxi
source: hand                    # hand | learned
types: {P: passenger, L: location, L1: location, L2: location, Dir: dir}
reward_parents:                 # global R parents (block lists: atoms contain commas)
  - at(taxi,L1)
  - move(Dir)
  - wall(L1,Dir)
statements:
  - influences:
      - at(taxi,L1)
      - move(Dir)
      - wall(L1,Dir)
    target: at(taxi,L2)
  - operator: pickup(P)
    if: []
    influences:
      - at(taxi,L1)
      - at(P,L)
      - in(P,taxi)
    target: in(P,taxi)
  - operator: drop(P)
    influences:
      - at(taxi,L1)
      - in(P,taxi)
      - dest(P,L)
      - delivered(P)
    target: delivered(P)
operators:
  pickup(P):
    reward_parents: [in(P,taxi)]        # single-atom flow lists are fine
    termination_parents: [in(P,taxi)]
  drop(P):
    reward_parents: [delivered(P)]
    termination_parents: [delivered(P)]
```
Unknown keys are rejected. Statement-local variables are renamed apart from operator roles
during the closure, so a statement that happens to use a role's name is still existential.

- Closure: for operator `o`, start from global reward parents ∪ `o`'s reward and termination
  parents; repeatedly add `influences` ∪ `if` of every statement (unconditional or tagged `o`)
  whose target unifies with a literal already in the set, until a fixpoint (optional
  `max_depth`; the paper used 2). Action literals (`move(Dir)`) are recorded but never projected.
- Projection + lifting: an atom survives iff it unifies with some relevant literal under the
  operator binding (argument roles fixed, other variables existential, typed); surviving atoms
  are lifted by renaming bound objects to `?role`. Key = frozenset of lifted atoms. For
  `pickup(p2)` this keeps `at(taxi,·)`, `at(?P,·)`, `in(?P,taxi)`, all `wall` atoms, and drops
  every atom about other passengers, as the brief's hand-checked example requires.
- A learned abstraction is either a YAML with `source: learned` loaded by the same class, or a
  new subclass registered in `ABSTRACTIONS`; nothing else changes.

## 6. Agents (`reprel/agents`)

```python
class Agent(ABC):
    def act(self, key, rng, epsilon) -> int
    def update(self, key, action, reward, next_key, done) -> None
    def state_dict() / load_state_dict()
```

`QLearningAgent(n_actions, alpha, gamma)`: `dict[Hashable, np.ndarray]`, zero init, random
tie-breaking, ε-greedy (correct, unlike the original). `EpsilonSchedule` is per-episode linear
(default 0.75 → 0.01 over N episodes, matching the original's intent) or constant.
`AgentPool` maps operator *name* → agent, so lifted keys share one table across instances;
`save/load` via pickle of plain dicts, enabling the transfer experiment.

## 7. Execution (`reprel/execution`)

`RePReLExecutor(domain, planner, abstraction, pool, cfg)` implements Algorithm 1 with explicit
failure handling:
1. `state = reset`; `plan = planner.plan(state, goal)`.
2. For each operator instance: if β already holds, skip. Otherwise loop: key → act → step →
   next key; reward = env reward + tR on termination; `update` the active agent only;
   log transition (full ground states + op + args) if a logger is attached.
3. Stop the episode on env `done` or `max_steps` (no stepping a finished episode).
4. If `max_operator_steps` is set and exceeded → count a subtask failure and replan from the
   current state (`replan_on_timeout: true`); default config mirrors the original (no budget).
5. Evaluation: every `eval_every` env steps run `eval_episodes` greedy episodes with
   `eval_epsilon`; record mean return, success rate, mean steps-to-success.

`ExplorationCollector` reuses the same loop with a uniform-random policy and no learning, writes
transitions per operator (input to M7). `FlatExecutor` runs one agent on `NoAbstraction`
keys with the raw environment reward and the same evaluation hook.

## 8. Logging and reproducibility (`reprel/logging`, `reprel/seeding.py`)

- Transition log schema (Parquet, one row per step): `run_id, seed, episode, t, operator, args
  (list<str>), state (list<str>), action, reward, done, next_state (list<str>), terminated`.
  Atoms serialise as `pred(a,b)` strings; the reader parses them back into `State`s
  (round-trip test).
- Metrics CSV: `condition, seed, env_steps, episodes, eval_return_mean, eval_return_std,
  success_rate, mean_steps_to_success, epsilon`.
- Every run directory `outputs/<experiment>/<condition>/seed_<k>/` contains `config.yaml`
  (resolved), `meta.json` (git hash, dirty flag, seed, timestamp, Python/numpy versions),
  `metrics.csv`, optional `transitions.parquet`, `agents.pkl`.
- `seed_everything(seed)` returns a `numpy.random.Generator`; all randomness (domain, agents,
  schedules) takes an explicit `rng`. No module-level RNG, no `random`/`np.random` globals.

## 9. Configuration and scripts

`RunConfig` (frozen dataclasses): `domain` (name + size + rewards), `planner`, `abstraction`
(name + yaml path), `agent` (alpha, gamma, epsilon schedule, tR), `executor` (budget in env steps,
eval cadence, operator step budget), `logging` (transitions on/off), `seeds` (list), `transfer`
(optional: list of successive domain sizes; agents carried over). Loaded from YAML with
`--set a.b=c` dotted overrides. `scripts/run.py` runs one config across its seeds sequentially
(or `--seed k`), `scripts/explore.py` collects exploration data, `scripts/plot.py` aggregates
`metrics.csv` files into mean ± std curves (return and success rate vs. env steps) per condition.

## 10. Experiments (M5)

| Condition | Executor | Abstraction | Reward to learner |
|---|---|---|---|
| flat | FlatExecutor | none | env reward |
| reprel_none | RePReLExecutor | none | env reward + tR per operator |
| reprel_dfoci | RePReLExecutor | dfoci (`reprel/dfoci/taxi.yaml`) | same |
| transfer | RePReLExecutor | dfoci | train on 1 then 2 passengers, then evaluate + continue on 3, 4, 5 without reset |

Defaults to confirm at M5: α=0.01, γ=0.99, ε 0.75→0.01 over 20k episodes, tR scaled to the Taxi
reward scale (proposed 10, i.e. the pickup reward; the original used 1 at reward scale 1 and
100 at scale 10–100), 5 seeds, evaluation every 10k env steps with 100 greedy episodes.

## 11. Tests (pytest)

`tests/test_state.py` (atoms, unify, substitute, lift, diff), `test_taxi.py` (every action's
semantics, rewards, termination, instance generation determinism per seed), `test_planner.py`
(plans valid for 1–5 passengers, including a passenger already in the taxi), `test_dfoci.py`
(closure on the Taxi YAML and on a synthetic fuel spec; key equality across instances differing
only in irrelevant passengers; `if:` clauses included), `test_none_abstraction.py`, `test_q_learning.py`
(converges on a 5-state chain MDP), `test_executor.py` (terminates, skips satisfied operators,
replans on timeout), `test_transitions.py` (write → read → identical States),
`test_config.py` (YAML round trip + overrides).

## 12. Deviations from the original (intentional)
1. Correct ε-greedy exploration (the original's never fires).
2. Episode ends cleanly when the step limit is hit mid-subtask.
3. Optional per-operator step budget with replanning (off by default).
4. GTPyhop instead of Pyhop; methods use variables, not per-object registration.
5. Relational Taxi/Office states as ground atoms instead of vectors/strings; abstraction keys are
   order-independent frozensets with argument objects renamed to roles.
6. Walls are explicit static atoms so the D-FOCI statements for movement are complete.

## 13. Open items that need you
- GitHub push access: on this machine the SSH key is rejected by GitHub, `gh` is not installed,
  and the GitHub connector failed to connect. Before M1 is pushed, either add
  `~/.ssh/id_ed25519.pub` to your GitHub account, or install `gh` and run `gh auth login`, and
  tell me the repository name (proposed `nikhileshp/RePReL`). I will create it empty (or push to
  one you create) with no AI attribution in commits, PRs, or files.
- Confirm the Taxi tR and budget defaults in §10, or leave them to M5 tuning-free defaults.
