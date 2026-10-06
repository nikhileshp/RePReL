"""One agent per operator *name*, created lazily, so lifted keys share a table across instances."""

from __future__ import annotations

import inspect
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .q_learning import Agent


class AgentPool:
    """``factory`` builds an agent; it may take the agent's name to size special tables."""

    def __init__(self, factory: Callable[..., Agent]) -> None:
        self._factory = factory
        self._takes_name = len(inspect.signature(factory).parameters) >= 1
        self._factories: dict[str, Callable[[], Agent]] = {}
        self._agents: dict[str, Agent] = {}

    def register(self, name: str, factory: Callable[[], Agent]) -> None:
        """Use a dedicated factory for ``name`` (e.g. a meta-controller with its own action set)."""
        self._factories[name] = factory

    def get(self, operator: str) -> Agent:
        agent = self._agents.get(operator)
        if agent is None:
            if operator in self._factories:
                agent = self._factories[operator]()
            elif self._takes_name:
                agent = self._factory(operator)
            else:
                agent = self._factory()
            self._agents[operator] = agent
        return agent

    def __contains__(self, operator: str) -> bool:
        return operator in self._agents

    def state(self) -> dict[str, Any]:
        return {name: agent.state_dict() for name, agent in self._agents.items()}

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
