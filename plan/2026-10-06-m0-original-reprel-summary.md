# M0 — Summary of the original RePReL implementation

Sources read (all public, cloned read-only for study; nothing copied):

| Source | What it contains |
|---|---|
| `harshakokel/RePReL` (ICAPS 2021 tabular code) | **Box World only.** Pyhop planner, string-based abstraction, tabular Q-learning, 5-run CSV results. ~1.8k lines. |
| `harshakokel/RePReL-domains` | Gym envs for **Taxi** (vector obs) and **Office World** (MultiDiscrete obs). No planner, no abstraction. |
| `starling-lab/DeepRePReL` | rlkit-based DQN version. Contains the **Taxi and Office Pyhop planners**, the per-operator abstraction masks, and the rollout loop. |
| ICAPS 2021 paper + StarAI 2021 abstract (arXiv:2110.08318) | Formal definitions, Algorithm 1, **Table 1 with all D-FOCI statements** for Craft, Office, Taxi, Box World. |

Not available: the tabular Taxi/Office/Craft code used for the ICAPS figures was never
released (the tabular repo only ships Box World), and the paper's appendix site
(`starling.utdallas.edu/papers/RePReL`) timed out. Craft World has no released code at all.

## 1. Domains

### Taxi (RePReL-domains `relational_taxi.py`)
- Grid from an ASCII layout (`eight_layout` 8x8 interior used in experiments; `w` = wall; `R G B Y` depots).
- Taxi starts at a random non-wall cell. Each passenger samples a pickup depot and a distinct
  destination depot from `{R,G,B,Y}`; different passengers may share depots.
- Actions: up, down, left, right, pickup, drop (6).
- `pickup`: only if taxi is empty and a not-yet-delivered passenger's pickup equals the taxi cell
  (first match by passenger index boards). `drop`: only if the carried passenger's destination
  equals the taxi cell.
- Rewards: step `-0.1`; pickup `+10`; drop `+20`; illegal pickup/drop `-1`; wall bump `0`
  (`no_move_cost=0`). Episode ends when all passengers delivered or `max_steps=1000`.
- Tasks: 1, 2, 3 passengers (`max_passenger=3` pads the vector obs so policies transfer).
- Paper text: "Task 3 is to drop all three passengers ... in that order" — the planner's
  `achieve_goal` method always picks the lowest-index undelivered passenger.

### Office World (RePReL-domains `office.py`, from Illanes et al. 2020)
- 12x9 grid with thin walls between cells (`WALLS` set of blocked cell pairs). Objects at
  fixed cells: `a b c d` (visit targets), `e` mail, `f` coffee (2 cells), `g` office,
  `n` plants (6 cells, penalised).
- State: `(x, y, 9 boolean facts)` = visited-a/b/c/d, has-mail, has-coffee, visited-office,
  delivered-mail, delivered-coffee. Facts flip on entering a cell; entering the office while
  holding mail/coffee converts `has-*` to `delivered-*`.
- Actions: up/down/left/right (4). Rewards: cost `1` per step, `+10` extra for an invalid move
  (wall/bounds) and `+10` extra for standing on a plant (all negated), terminal `+100` when the
  task's target facts hold. No step limit in the env itself.
- Tasks (paper): deliver mail; deliver coffee; deliver mail and coffee; visit A, B, C, D.
- Agent start: random non-object cell.

### Box World (RePReL tabular repo `box_world_env.py`)
- 10x10 interior (12x12 with border), Zambaldi-style keys/locks with 18 colours, gem goal.
  Obs type `relational`: list of predicate strings
  `neighbor(Dir,Obj)`, `direction(Obj,Dir)`, `color(Obj,C)`, `inside(Lock,Key)`, `own(Key)`,
  `open(Lock)`, `agent-at(Obj)`. Only the 8 neighbours plus coarse directions are observed.
- Actions: 4 moves. Rewards: `step_cost 0.1`, `no_move_cost 0.2`, `reward_key 1`,
  `reward_gem 1`, wrong lock `-1` and episode ends. `max_steps` 100/200/300 for goal lengths 1/2/3.
- Tasks: task1 (1 lock, agent owns key), task2 (collect key, open lock), task5 (= task2 without
  the initial key), task3 (two boxes).

### Craft World
Described in the paper only (Andreas et al. 2017 variant, 11 objects, tasks: get wood+iron,
make stick, make axe, mine gem). No code released. **Out of scope unless you want it rebuilt from
the paper description.**

## 2. Predicates, operators, HTN methods

| Domain | Planner predicates (planner state) | Operators (RL subtasks) | Methods |
|---|---|---|---|
| Taxi | `at(P,L)`, `in-taxi(P)`, `dest(P,L)`, `at-dest(P)`, `taxi-at(L)` | `pickup(P)` pre `∃L at(P,L) ∧ ¬in-taxi(P)`, eff `taxi-at(L) ∧ in-taxi(P)`, β `in-taxi(P)`; `drop(P)` pre `∃L in-taxi(P) ∧ dest(P,L)`, eff `taxi-at(L) ∧ at-dest(P) ∧ ¬in-taxi(P)`, β `at-dest(P)` | `achieve_goal(G)`: first undelivered P → `[transport(P), achieve_goal(G)]`; `transport(P)` → `[drop(P)]` if in taxi else `[pickup(P), drop(P)]` |
| Office | facts as objects `0..8`; `with-agent(X)`, `delivered(X)`, `office(L)` in the paper | `pickup(X)` (visit a/b/c/d, get mail/coffee, visit office), β `with-agent(X)`; `deliver(X)` (has-X at office → delivered-X), β `delivered(X)` | `achieve_goal(G)` → `solve(first unachieved)`; `solve(delivered-X)` → `[pickup(has-X), deliver(has-X, delivered-X)]`; `solve(visit-X)` → `[pickup(X)]` |
| Box World | `keys`, `locks`, `inside`, `color`, `own_key` | `pick_key(K)` β `own(K)`; `unlock(L)` β `open(L)` | `collect_gem` → chain of `collect_some_key`/`open_some_lock` by colour |

All planners are Pyhop (SHOP-style, global method/operator registries, methods registered
dynamically per object to emulate variables).

## 3. D-FOCI statements (paper Table 1) and derived relevant sets

```
# shared (no operator)
{taxi-at(L1), move(Dir)}  -+1->  taxi-at(L2)
{taxi-at(L1), move(Dir)}  --->   R
pickup(P): {taxi-at(L1), at(P,L), in-taxi(P)}            -+1-> in-taxi(P)
pickup(P): in-taxi(P)                                     ---> Ro
drop(P):   {taxi-at(L1), in-taxi(P), dest(P,L), at-dest(P)} -+1-> at-dest(P)
drop(P):   at-dest(P)                                     ---> Ro
```
Relevant sets: `pickup(P)`: `{taxi-at(L1), at(P,L), in-taxi(P), move(Dir)}`;
`drop(P)`: `{taxi-at(L1), in-taxi(P), dest(P,L), at-dest(P), move(Dir)}`.

Office: `pickup(X)`: `{agent-at(L1), at(X,L), with-agent(X), move}`; `deliver(X)`:
`{agent-at(L1), with-agent(X), office(L), delivered(X), move}`.
Box World: `pick_key(K)`: `{neighbor(Dir,C), agent-at(L1), direction(K,Dir2), own(K), move}`;
`unlock(L)`: same with `open(L)`.

How the code realises the abstraction:
- Box World (tabular): keep predicate strings whose name is in the relevant list, replace the
  bound object name with the literal `X`, join into one string → Q-table key. Other keys/locks
  are dropped by the name filter. No renaming of non-argument objects.
- Taxi (deep): grid image + the target passenger's 5-dim pickup one-hot (pickup) or
  `in-taxi` bit + 4-dim destination one-hot (drop). Other passengers zeroed.
- Office (deep): `(x, y)` + mask over facts: `pickup(X)` keeps fact X; `deliver(X)` keeps
  has-X, delivered-X and visited-office.
- Unrolling depth: paper bounds to k=2 levels and 1 time step; in practice the hand-derived sets
  in Table 1 are the full ancestor closure.

## 4. Subtask reward and termination
- Per Algorithm 1 and `RePReL_QLearning.train`: the active operator's agent receives the
  environment reward `r`, plus a fixed terminal bonus `tR` when the next state satisfies β(o).
  `tR = 1` in the tabular Box World code (reward scale 1); `task_terminal_reward = 100` in the
  deep Taxi/Office code (reward scale 10–100).
- If β(o) already holds when the operator starts, it is skipped.
- Only the active operator's policy is updated per step (contrast: Taskable RL updates all).
- No per-operator step budget and no replanning: a subtask runs until β(o) or the episode's
  `max_steps`. There is no failure handling.

## 5. Tabular Q-learning hyperparameters (`RePReL_QLearning.py`)
| Parameter | Value |
|---|---|
| learning rate α | 0.01 |
| discount γ | 0.99 |
| Q init | zeros, ties broken uniformly at random |
| ε schedule | per-episode, linear from 0.75 down to 0.01 over 20 000 episodes, then 0.01 |
| terminal bonus tR | 1 |
| budget | 1.5M env steps per task (Box World); paper: 50K (Craft), 30K (Office); Taxi not stated in text |
| evaluation | every 10 000 env steps, 100 greedy episodes with ε=0.01; log success rate and mean return |
| runs | 5 |

Note: `get_action` uses `np.random.randn(0, 1) < ep`, which evaluates to an empty array and is
always falsy, so **the ε-greedy branch never fires** in the released code. Exploration comes only
from zero-initialised Q-values with random tie-breaking. We will implement proper ε-greedy and
flag the difference.

## 6. Evaluation protocol (paper + code)
- Conditions: RePReL, RePReL+T (transfer), trl (Taskable RL "seq" variant: planner, no
  abstraction, full state), trl+T, and hrl (options per location; not in released code).
- Transfer: Q-tables are carried from task i to task i+1 without reset (Box World: task2 → task5 → task3).
- Curves: mean over 5 runs of evaluation return (and success rate) vs. environment steps.
- Expected qualitative result: RePReL > trl in sample efficiency on every task; RePReL+T reaches
  near-optimal on larger instances with little or no extra training (Taxi task 2/3, Box World task 3).

## 7. Bugs / quirks in the original to avoid reproducing
1. ε-greedy never triggers (see §5).
2. After an episode hits `max_steps` mid-subtask, the training loop continues to the next
   operator and keeps calling `env.step` on a finished episode.
3. Pyhop uses module-level global registries; Taxi methods are registered once per passenger
   index, so the planner is tied to `max_passenger`.
4. Abstraction keys are concatenated strings whose order depends on the env's emission order.
