"""One agent per operator *name*, created lazily, so lifted keys share a table across instances."""

from __future__ import annotations

import pickle
from collections.abc import Callable
from pathlib import Path

from .q_learning import Agent


class AgentPool:
    def __init__(self, factory: Callable[[], Agent]) -> None:
        self._factory = factory
        self._agents: dict[str, Agent] = {}

    def get(self, operator: str) -> Agent:
        agent = self._agents.get(operator)
        if agent is None:
            agent = self._factory()
            self._agents[operator] = agent
        return agent

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._agents)

    @property
    def total_keys(self) -> int:
        return sum(agent.n_keys for agent in self._agents.values())

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as fh:
            pickle.dump({name: agent.state_dict() for name, agent in self._agents.items()}, fh)

    def load(self, path: str | Path) -> None:
        with open(path, "rb") as fh:
            states = pickle.load(fh)  # noqa: S301 - our own files
        for name, state in states.items():
            self.get(name).load_state_dict(state)
