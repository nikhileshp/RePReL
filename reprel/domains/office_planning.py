"""HTN model of Office World: ``pickup(X)`` (visit/collect) and ``deliver(X)`` operators."""

from __future__ import annotations

from dataclasses import replace

from reprel.core.atoms import Atom, Literal
from reprel.core.state import State
from reprel.planning.htn_gtpyhop import GTPyhopPlanner, Method, Task
from reprel.planning.operators import Goal, OperatorSpec

from .office import AGENT, OfficeDomain


def _lits(*texts: str) -> tuple[Literal, ...]:
    return tuple(Literal.parse(t) for t in texts)


def _atoms(*texts: str) -> tuple[Atom, ...]:
    return tuple(Atom.parse(t) for t in texts)


# One operator for visiting a location and for collecting an item: neither has a
# precondition beyond the item's location and both have the same effect (paper, Office World).
PICKUP = OperatorSpec(
    name="pickup",
    params=(("X", "item"),),
    preconditions=_lits(f"at({AGENT},L0)", "at(X,L)", "not with(X)"),
    add=_atoms("with(X)", f"at({AGENT},L)"),
    delete=_atoms(f"at({AGENT},L0)"),
    termination=_lits("with(X)"),
)

DELIVER = OperatorSpec(
    name="deliver",
    params=(("X", "item"),),
    preconditions=_lits(f"at({AGENT},L0)", "with(X)", "office(L)"),
    add=_atoms("delivered(X)", "with(office)", f"at({AGENT},L)"),
    delete=_atoms("with(X)", f"at({AGENT},L0)"),
    termination=_lits("delivered(X)"),
)

OFFICE_OPERATORS: tuple[OperatorSpec, ...] = (PICKUP, DELIVER)


def m_achieve(state: State, goal: Goal) -> list[Task] | None:
    """Solve the alphabetically first unsatisfied goal literal, then recurse."""
    pending = sorted(
        (lit for lit in goal if lit.positive and not state.holds(lit.atom)),
        key=lambda lit: (lit.atom.pred, lit.atom.args),
    )
    if not pending:
        return []
    return [("solve", pending[0]), ("achieve", goal)]


def m_solve_visit(state: State, lit: Literal) -> list[Task] | None:
    if lit.atom.pred == "with":
        return [("pickup", lit.atom.args[0])]
    return None


def m_solve_deliver(state: State, lit: Literal) -> list[Task] | None:
    if lit.atom.pred != "delivered":
        return None
    item = lit.atom.args[0]
    if state.holds(Atom("with", (item,))):
        return [("deliver", item)]
    return [("pickup", item), ("deliver", item)]


OFFICE_METHODS: dict[str, list[Method]] = {
    "achieve": [m_achieve],
    "solve": [m_solve_visit, m_solve_deliver],
}


def make_office_planner(terminal_reward: float = 1.0) -> GTPyhopPlanner:
    operators = tuple(replace(op, terminal_reward=terminal_reward) for op in OFFICE_OPERATORS)
    return GTPyhopPlanner(
        operators,
        OFFICE_METHODS,
        root_task=lambda goal: [("achieve", goal)],
        signature=OfficeDomain.predicates,
        name="office",
    )
