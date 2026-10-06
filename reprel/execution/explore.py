"""Exploration-only data collection inside subtasks: random policy, no learning, full logs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from reprel.abstraction.none import NoAbstraction
from reprel.agents.agent_pool import AgentPool
from reprel.agents.q_learning import RandomAgent
from reprel.core.domain import Domain
from reprel.logging.transitions import TransitionLogger
from reprel.planning.planner import Planner

from .executor import ExecutorConfig, RePReLExecutor


class ExplorationCollector:
    """Follows the planner's operator sequence but acts uniformly at random within each subtask.

    ``max_operator_steps`` bounds how long a random policy may wander inside one subtask before
    the executor replans; it keeps episodes finite and spreads data across operators.
    """

    def __init__(
        self,
        domain: Domain,
        planner: Planner,
        max_operator_steps: int = 50,
        max_replans: int = 20,
    ) -> None:
        self.executor = RePReLExecutor(
            domain,
            planner,
            NoAbstraction(),
            AgentPool(lambda: RandomAgent(domain.n_actions)),
            ExecutorConfig(max_operator_steps=max_operator_steps, max_replans=max_replans),
        )

    def collect(
        self,
        rng: np.random.Generator,
        episodes: int,
        path: str | Path,
        run_id: str = "explore",
        seed: int = -1,
    ) -> dict[str, Any]:
        transitions, successes = 0, 0
        with TransitionLogger(path, run_id=run_id, seed=seed) as logger:
            for _ in range(episodes):
                result = self.executor.run_episode(rng, epsilon=1.0, learn=False, logger=logger)
                transitions += result.env_steps
                successes += int(result.success)
        return {"episodes": episodes, "transitions": transitions, "successes": successes}
