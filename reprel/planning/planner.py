"""Planner interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from reprel.core.state import State

from .operators import Goal, OperatorInstance


class PlanningFailure(RuntimeError):
    """Raised when no plan achieves the goal from the given state."""


class Planner(ABC):
    """Maps a (state, goal) problem to a sequence of grounded operators."""

    @abstractmethod
    def plan(self, state: State, goal: Goal) -> list[OperatorInstance]:
        """Return a plan, or raise :class:`PlanningFailure`."""
