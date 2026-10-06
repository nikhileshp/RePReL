"""Abstract Domain interface and registry.

A Domain is a goal-directed relational MDP. ``step`` is a pure function of ``(state, action)``:
episode bookkeeping (step counts, budgets) belongs to the executor, which keeps transition
logging exact and lets multi-agent domains subclass without changing this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, TypeVar

import numpy as np

from .atoms import Literal
from .state import State

Action = str
Goal = frozenset[Literal]


@dataclass(frozen=True)
class Transition:
    """Result of applying one primitive action."""

    next_state: State
    reward: float
    done: bool
    info: Mapping[str, Any] = field(default_factory=dict)


class Domain(ABC):
    """Interface every environment implements.

    Class attributes:
        name: Registry name.
        types: Type name -> parent type (None for roots).
        predicates: Predicate -> argument types (``"object"`` accepts any type).
        actions: Primitive actions, in a fixed order (the agent's action index).
        max_steps: Default episode length limit enforced by the executor.
    """

    name: ClassVar[str]
    types: ClassVar[Mapping[str, str | None]]
    predicates: ClassVar[Mapping[str, tuple[str, ...]]]
    actions: ClassVar[tuple[Action, ...]]
    max_steps: int

    @abstractmethod
    def reset(self, rng: np.random.Generator) -> State:
        """Sample a new problem instance and return its initial state."""

    @abstractmethod
    def step(self, state: State, action: Action, rng: np.random.Generator) -> Transition:
        """Apply ``action`` in ``state``. Must not mutate ``state``."""

    @abstractmethod
    def goal(self, state: State) -> Goal:
        """Goal literals the planner must achieve for the instance ``state`` belongs to."""

    @abstractmethod
    def is_success(self, state: State) -> bool:
        """True when the task is solved in ``state``."""

    # --- helpers shared by all domains ------------------------------------------------
    def validate_action(self, action: Action) -> None:
        if action not in self.actions:
            raise ValueError(f"unknown action {action!r}; expected one of {self.actions}")

    def action_index(self, action: Action) -> int:
        self.validate_action(action)
        return self.actions.index(action)

    @property
    def n_actions(self) -> int:
        return len(self.actions)


D = TypeVar("D", bound=type[Domain])

DOMAINS: dict[str, type[Domain]] = {}


def register_domain(name: str) -> Callable[[D], D]:
    """Class decorator adding a Domain subclass to ``DOMAINS``."""

    def decorator(cls: D) -> D:
        DOMAINS[name] = cls
        return cls

    return decorator


def make_domain(name: str, **config: Any) -> Domain:
    """Instantiate a registered domain by name, forwarding keyword config."""
    if name not in DOMAINS:
        raise KeyError(f"unknown domain {name!r}; registered: {sorted(DOMAINS)}")
    return DOMAINS[name](**config)
