# RePReL

A modular re-implementation of **RePReL** (Kokel et al., ICAPS 2021): a relational HTN planner
on top, per-operator reinforcement learners below, and D-FOCI statements that derive safe,
task-specific state abstractions.

Status: under construction. See `plan/` for the design specification and milestone plans.

## Setup

```bash
uv sync --extra dev
uv run pytest
```
