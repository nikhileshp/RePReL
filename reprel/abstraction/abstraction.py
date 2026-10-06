"""Abstraction interface: maps (state, grounded operator) to the key an RL agent learns over."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable
from typing import Any, TypeVar

from reprel.core.state import State
from reprel.planning.operators import OperatorInstance


class Abstraction(ABC):
    """Task-specific state abstraction.

    A learned abstraction is a new subclass registered in ``ABSTRACTIONS`` (or a learned
    D-FOCI spec loaded by the existing class); nothing else in the system changes.
    """

    @abstractmethod
    def abstract(self, state: State, op: OperatorInstance) -> Hashable:
        """Return a hashable key for ``state`` while executing operator instance ``op``."""


A = TypeVar("A", bound=type[Abstraction])

ABSTRACTIONS: dict[str, type[Abstraction]] = {}


def register_abstraction(name: str) -> Callable[[A], A]:
    def decorator(cls: A) -> A:
        ABSTRACTIONS[name] = cls
        return cls

    return decorator


def make_abstraction(name: str, **config: Any) -> Abstraction:
    if name not in ABSTRACTIONS:
        raise KeyError(f"unknown abstraction {name!r}; registered: {sorted(ABSTRACTIONS)}")
    return ABSTRACTIONS[name](**config)
