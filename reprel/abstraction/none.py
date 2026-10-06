"""No abstraction: the key is the full ground state (the paper's no-abstraction baseline)."""

from __future__ import annotations

from collections.abc import Hashable, Iterable

from reprel.core.state import State
from reprel.planning.operators import OperatorInstance

from .abstraction import Abstraction, register_abstraction


@register_abstraction("none")
class NoAbstraction(Abstraction):
    """Full grounded state as the key.

    Args:
        include_binding: If True, append the operator's grounded arguments to the key so the
            agent can tell ``pickup(p1)`` from ``pickup(p2)``. Off by default, matching the
            original implementation's "trl" baseline.
        exclude: Predicates dropped from the key. Use it to reproduce an original environment
            whose observation hid part of the state (Box World's sensors carry no positions:
            ``exclude=("at",)``).
    """

    def __init__(self, include_binding: bool = False, exclude: Iterable[str] = ()) -> None:
        self.include_binding = include_binding
        self.exclude = frozenset(exclude)

    def abstract(self, state: State, op: OperatorInstance) -> Hashable:
        atoms = state.atoms
        if self.exclude:
            atoms = frozenset(a for a in atoms if a.pred not in self.exclude)
        if self.include_binding:
            return atoms, op.args
        return atoms
