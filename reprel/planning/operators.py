"""Operator specifications (planner-level model) and grounded operator instances (RL subtasks)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from reprel.core.atoms import Atom, Literal, type_compatible
from reprel.core.query import Signature, matches, satisfied
from reprel.core.state import State

Goal = frozenset[Literal]


@dataclass(frozen=True)
class OperatorSpec:
    """A planning operator ``o = <name(params), pre, eff, beta>`` (paper, Definition 2).

    Variables named in ``params`` are the operator's argument roles; any other variable in
    the preconditions is existentially quantified and bound when the operator is applied.
    """

    name: str
    params: tuple[tuple[str, str], ...]
    preconditions: tuple[Literal, ...]
    add: tuple[Atom, ...]
    delete: tuple[Atom, ...]
    termination: tuple[Literal, ...]
    terminal_reward: float = 1.0

    def __post_init__(self) -> None:
        bound = set(self.roles)
        for lit in self.preconditions:
            bound.update(lit.atom.variables())
        for atom in (*self.add, *self.delete):
            unbound = set(atom.variables()) - bound
            if unbound:
                raise ValueError(f"{self.name}: effect {atom} uses unbound variables {unbound}")

    @property
    def roles(self) -> tuple[str, ...]:
        return tuple(role for role, _ in self.params)

    def instantiate(self, args: Sequence[str]) -> OperatorInstance:
        if len(args) != len(self.params):
            raise ValueError(f"{self.name} takes {len(self.params)} args, got {args!r}")
        return OperatorInstance(self, tuple(args))

    def applicable(
        self, state: State, binding: Mapping[str, str], *, signature: Signature | None = None
    ) -> dict[str, str] | None:
        """Return the full substitution (roles + existential variables) if ``pre`` holds."""
        types = state.types()
        for role, required in self.params:
            obj = binding.get(role)
            if obj is None or obj not in types or not type_compatible(types[obj], required):
                return None
        return next(matches(self.preconditions, state, binding, signature=signature), None)

    def apply(
        self, state: State, binding: Mapping[str, str], *, signature: Signature | None = None
    ) -> State | None:
        """Deterministic planner-level successor, or None when not applicable."""
        theta = self.applicable(state, binding, signature=signature)
        if theta is None:
            return None
        delete = [a.substitute(theta) for a in self.delete]
        add = [a.substitute(theta) for a in self.add]
        return state.with_atoms(add=add, remove=delete)


@dataclass(frozen=True)
class OperatorInstance:
    """A grounded operator: the unit of work handed to one RL agent."""

    spec: OperatorSpec
    args: tuple[str, ...]

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def binding(self) -> dict[str, str]:
        return dict(zip(self.spec.roles, self.args, strict=True))

    @property
    def key(self) -> tuple[str, tuple[str, ...]]:
        return self.name, self.args

    def __str__(self) -> str:
        return f"{self.name}({','.join(self.args)})"

    def is_terminated(self, state: State, *, signature: Signature | None = None) -> bool:
        return satisfied(self.spec.termination, state, self.binding, signature=signature)

    def subtask_reward(
        self, env_reward: float, next_state: State, *, signature: Signature | None = None
    ) -> float:
        """Environment reward plus the terminal bonus when ``next_state`` satisfies beta."""
        if self.is_terminated(next_state, signature=signature):
            return env_reward + self.spec.terminal_reward
        return env_reward
