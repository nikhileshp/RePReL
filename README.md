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
  domains/      Taxi (multi-passenger, relational); Office and Box World to follow
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
```
