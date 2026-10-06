"""Tabular Q-learning over hashable abstract-state keys."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable
from typing import Any

import numpy as np


class Agent(ABC):
    """Interface for a subtask policy learner. Keys are whatever the Abstraction returns."""

    n_actions: int

    @abstractmethod
    def act(self, key: Hashable, rng: np.random.Generator, epsilon: float) -> int:
        """Choose an action index; explore uniformly with probability ``epsilon``."""

    @abstractmethod
    def update(
        self, key: Hashable, action: int, reward: float, next_key: Hashable, *, terminal: bool
    ) -> None:
        """Learn from one transition. ``terminal`` disables bootstrapping from ``next_key``."""

    @abstractmethod
    def state_dict(self) -> dict[str, Any]: ...

    @abstractmethod
    def load_state_dict(self, state: dict[str, Any]) -> None: ...

    @property
    def n_keys(self) -> int:
        return 0


class QLearningAgent(Agent):
    """Q-learning with a dict table, zero initialisation and uniform random tie-breaking."""

    def __init__(self, n_actions: int, alpha: float = 0.01, gamma: float = 0.99) -> None:
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.q: dict[Hashable, np.ndarray] = {}

    def _row(self, key: Hashable) -> np.ndarray:
        row = self.q.get(key)
        if row is None:
            row = np.zeros(self.n_actions)
            self.q[key] = row
        return row

    def act(self, key: Hashable, rng: np.random.Generator, epsilon: float) -> int:
        if epsilon > 0.0 and rng.random() < epsilon:
            return int(rng.integers(self.n_actions))
        row = self.q.get(key)
        if row is None:
            return int(rng.integers(self.n_actions))
        best = np.flatnonzero(row == row.max())
        return int(best[0]) if len(best) == 1 else int(rng.choice(best))

    def update(
        self, key: Hashable, action: int, reward: float, next_key: Hashable, *, terminal: bool
    ) -> None:
        row = self._row(key)
        bootstrap = 0.0 if terminal else float(self._row(next_key).max())
        row[action] += self.alpha * (reward + self.gamma * bootstrap - row[action])

    def value(self, key: Hashable) -> float:
        row = self.q.get(key)
        return 0.0 if row is None else float(row.max())

    @property
    def n_keys(self) -> int:
        return len(self.q)

    def state_dict(self) -> dict[str, Any]:
        return {
            "n_actions": self.n_actions,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "q": {k: v.copy() for k, v in self.q.items()},
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state["n_actions"] != self.n_actions:
            raise ValueError("action count mismatch")
        self.q = {k: np.asarray(v, dtype=float).copy() for k, v in state["q"].items()}


class RandomAgent(Agent):
    """Uniform random policy that never learns (exploration-only data collection)."""

    def __init__(self, n_actions: int) -> None:
        self.n_actions = n_actions

    def act(self, key: Hashable, rng: np.random.Generator, epsilon: float) -> int:
        return int(rng.integers(self.n_actions))

    def update(
        self, key: Hashable, action: int, reward: float, next_key: Hashable, *, terminal: bool
    ) -> None:
        return None

    def state_dict(self) -> dict[str, Any]:
        return {"n_actions": self.n_actions}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        return None
