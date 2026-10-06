# M1 Implementation Plan — core state model + Taxi domain + tests

**Goal:** A tested `reprel.core` (atoms, states, domain interface, seeding) and a relational
multi-passenger Taxi domain whose `step` semantics match RePReL-domains.

**Spec:** `plan/2026-10-06-reprel-design-spec.md` §1–§3, §8 (seeding), §11 (tests).

**Global constraints:** Python ≥ 3.11 (`.python-version` 3.12), `uv` project, deps numpy +
pyyaml only for M1; dev deps pytest, ruff, mypy. Type hints everywhere, frozen dataclasses,
no module-level RNG, files ≤ 400 lines, Conventional Commits, author = repo owner only.

**Review focus (inputs the spec implies but could bite):**
1. Two passengers sharing a pickup depot: pickup must board the lowest-index undelivered one
   and leave the other `at` the depot → test in `test_taxi.py`.
2. A passenger whose pickup depot equals the taxi's start cell: first `pickup` succeeds → test.
3. `dropoff` with no passenger aboard and `pickup` while carrying: reward −1, state unchanged → test.
4. Variable-vs-object detection for names like `L1` vs `l_1_2`: `is_variable` must be based on
   the first character only → test in `test_atoms.py`.
5. `State.lift` with a binding whose object does not appear anywhere: returns an equal state → test.

---

### Task 1: project scaffold
Files: `pyproject.toml`, `.python-version`, `reprel/__init__.py`, `tests/__init__.py`,
`README.md` (short), `tests/test_smoke.py`.
Steps: `uv init` equivalent by hand; `uv add numpy pyyaml`; `uv add --dev pytest ruff mypy`;
smoke test `import reprel`; `uv run pytest`; commit `chore: scaffold uv project`.

### Task 2: atoms and unification (`reprel/core/atoms.py`, `tests/test_atoms.py`)
Produces:
```python
def is_variable(name: str) -> bool                      # first char uppercase
@dataclass(frozen=True) class Obj: name; type
@dataclass(frozen=True) class Atom: pred; args: tuple[str, ...]
    @classmethod parse(cls, text: str) -> Atom          # "at(p1,l_0_0)"
    def __str__ -> "at(p1,l_0_0)"; is_ground; variables(); substitute(theta) -> Atom
@dataclass(frozen=True) class Literal: atom; positive=True; parse("not in(P,T)")
Substitution = Mapping[str, str]
def unify(pattern: Atom, ground: Atom, theta: Substitution | None = None,
          types: Mapping[str, str] | None = None, signature: Mapping[str, tuple[str, ...]] | None = None
          ) -> dict[str, str] | None
```
Tests: parse/str round trip; is_variable on `P`, `L1`, `p1`, `l_1_2`; unify success binds
variables, unify fails on pred mismatch, arity mismatch, conflicting binding, type mismatch
when signature+types given; substitute leaves unbound variables.

### Task 3: State (`reprel/core/state.py`, `tests/test_state.py`)
Produces:
```python
@dataclass(frozen=True) class State:
    atoms: frozenset[Atom]; objects: frozenset[Obj]
    holds(atom) -> bool; type_of(name) -> str; objects_of_type(t) -> tuple[str, ...] (sorted)
    atoms_with(pred) -> frozenset[Atom]
    with_atoms(add=(), remove=()) -> State
    substitute(theta) -> State            # renames object names in atoms and objects
    lift(binding: Mapping[role, obj]) -> State   # obj -> "?role"
    diff(other) -> tuple[frozenset, frozenset]    # (added, removed) from self to other
    to_strings() -> list[str] (sorted); from_strings(strs, objects)
```
Tests: immutability/hash equality; with_atoms; lift renames bound object everywhere, unbound
binding is a no-op; diff; to_strings/from_strings round trip; objects_of_type sorted.

### Task 4: Domain ABC + seeding (`reprel/core/domain.py`, `reprel/seeding.py`, `tests/test_seeding.py`)
Produces:
```python
Action = str
@dataclass(frozen=True) class Transition: next_state; reward: float; done: bool; info: Mapping
class Domain(ABC): name: ClassVar[str]; types; predicates; actions
    reset(rng) -> State; step(state, action, rng) -> Transition; goal(state) -> frozenset[Literal]
    is_success(state) -> bool; max_steps: int
DOMAINS: dict[str, type[Domain]]; register_domain(name); make_domain(name, **cfg)
def seed_everything(seed: int) -> np.random.Generator   # also seeds random + PYTHONHASHSEED
```
Tests: same seed → same Generator draws; registry round trip.

### Task 5: Taxi domain (`reprel/domains/taxi.py` + `reprel/domains/taxi_layouts.py`, `tests/test_taxi.py`)
Produces `TaxiConfig(num_passengers=1, layout="eight", max_steps=1000, step_reward=-0.1,
pickup_reward=10.0, drop_reward=20.0, illegal_reward=-1.0)`, `TaxiDomain(cfg)` registered as
`"taxi"`. Objects/predicates/actions per spec §3. Layout parser: ASCII → cells, depots, walls.
Tests: reset produces N passengers with pickup ≠ dest, taxi on free cell, wall atoms match
layout; deterministic per seed; north moves/blocked; pickup/dropoff success + illegal cases;
shared depot boards lowest index; success → done + goal satisfied; max_steps is config only
(step is pure, no counter); step never mutates input state.

### Task 6: M1 closeout
`uv run ruff check . && uv run mypy reprel && uv run pytest -q`; update README; merge feature
branch into master with `--no-ff`; report.
