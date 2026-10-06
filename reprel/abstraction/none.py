"""No abstraction: the key is the full ground state (the paper's no-abstraction baseline)."""

from __future__ import annotations

from collections.abc import Hashable

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
    """

    def __init__(self, include_binding: bool = False) -> None:
        self.include_binding = include_binding

    def abstract(self, state: State, op: OperatorInstance) -> Hashable:
        if self.include_binding:
            return state.atoms, op.args
        return state.atoms
