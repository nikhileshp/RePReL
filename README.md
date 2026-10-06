# RePReL

A modular re-implementation of **RePReL** (Kokel et al., ICAPS 2021: *RePReL: Integrating
Relational Planning and Reinforcement Learning for Effective Abstraction*): a relational HTN
planner on top, per-operator reinforcement learners below, and D-FOCI statements that derive
safe, task-specific state abstractions.

Status: under construction. See `plan/` for the design specification and milestone plans.

## Layout

```
reprel/
  core/         atoms, immutable relational State, Domain interface + registry
  domains/      Taxi (multi-passenger, relational) + its HTN model; Office and Box World to follow
  planning/     OperatorSpec / OperatorInstance, Planner interface, GTPyhop HTN wrapper
  abstraction/  Abstraction interface + registry: `none` (full state) and `dfoci` (D-FOCI induced)
  dfoci/        hand-written D-FOCI statements per domain (YAML), e.g. taxi.yaml
  agents/       tabular Q-learning, random agent, epsilon schedules, per-operator AgentPool
  execution/    RePReLExecutor (plan -> subtasks -> replan), FlatExecutor baseline,
                ExplorationCollector, evaluate(), train()
  logging/      transition logs (Parquet/JSONL, full ground states) and metrics CSV + run metadata
  seeding.py    seed_everything(seed) -> numpy Generator
tests/
plan/           design spec and milestone plans
```

## Setup

```bash
uv sync --extra dev
uv run pytest
uv run ruff check . && uv run mypy reprel tests
```

## Quick look

```python
import numpy as np
from reprel.core import make_domain
import reprel.domains  # registers domains

dom = make_domain("taxi", num_passengers=2)
rng = np.random.default_rng(0)
state = dom.reset(rng)
print(state.to_strings())
t = dom.step(state, "north", rng)
print(t.reward, t.done)

from reprel.domains.taxi_planning import make_taxi_planner
plan = make_taxi_planner().plan(state, dom.goal(state))
print([str(op) for op in plan])  # ['pickup(p1)', 'drop(p1)', 'pickup(p2)', 'drop(p2)']

from reprel.abstraction import make_abstraction
abstraction = make_abstraction("dfoci", spec="taxi", signature=dom.predicates)
key = abstraction.abstract(state, plan[0])   # lifted relevant atoms for pickup(p1)
print(sorted(str(a) for a in key if a.pred != "wall"))  # ['at(?P,l_0_0)', 'at(taxi,l_6_3)']

from reprel.agents import AgentPool, LinearSchedule, QLearningAgent
from reprel.execution import RePReLExecutor, train
executor = RePReLExecutor(
    dom, make_taxi_planner(terminal_reward=10.0), abstraction,
    AgentPool(lambda: QLearningAgent(dom.n_actions, alpha=0.1, gamma=0.99)),
)
history = train(executor, rng, total_steps=100_000, eval_every=25_000, eval_episodes=20,
                eval_epsilon=0.01, schedule=LinearSchedule(0.75, 0.01, 2000))
print([(p.env_steps, p.eval.success_rate) for p in history])
```
