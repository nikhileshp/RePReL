"""HTN model of the Taxi domain: operator specs and task methods (paper, Figure 2)."""

from __future__ import annotations

from reprel.core.atoms import Atom, Literal
from reprel.core.state import State
from reprel.planning.htn_gtpyhop import GTPyhopPlanner, Method, Task
from reprel.planning.operators import Goal, OperatorSpec

from .taxi import TAXI, TaxiDomain, passenger_order


def _lits(*texts: str) -> tuple[Literal, ...]:
    return tuple(Literal.parse(t) for t in texts)


def _atoms(*texts: str) -> tuple[Atom, ...]:
    return tuple(Atom.parse(t) for t in texts)


PICKUP = OperatorSpec(
    name="pickup",
    params=(("P", "passenger"),),
    preconditions=_lits(f"at({TAXI},L0)", "at(P,L)", f"not in(P,{TAXI})", "not delivered(P)"),
    add=_atoms(f"in(P,{TAXI})", f"at({TAXI},L)"),
    delete=_atoms("at(P,L)", f"at({TAXI},L0)"),
    termination=_lits(f"in(P,{TAXI})"),
)

DROP = OperatorSpec(
    name="drop",
    params=(("P", "passenger"),),
    preconditions=_lits(f"at({TAXI},L0)", f"in(P,{TAXI})", "dest(P,L)"),
    add=_atoms("delivered(P)", "at(P,L)", f"at({TAXI},L)"),
    delete=_atoms(f"in(P,{TAXI})", f"at({TAXI},L0)"),
    termination=_lits("delivered(P)"),
)

TAXI_OPERATORS: tuple[OperatorSpec, ...] = (PICKUP, DROP)


# ------------------------------------------------------------------ task methods
def m_achieve(state: State, goal: Goal) -> list[Task] | None:
    """Transport the lowest-index passenger whose goal literal is not yet satisfied."""
    pending = sorted(
        (
            lit.atom.args[0]
            for lit in goal
            if lit.positive and lit.atom.pred == "delivered" and not state.holds(lit.atom)
        ),
        key=passenger_order,
    )
    if not pending:
        return []
    return [("transport", pending[0]), ("achieve", goal)]


def m_transport_in_taxi(state: State, p: str) -> list[Task] | None:
    if state.holds(Atom("in", (p, TAXI))):
        return [("drop", p)]
    return None


def m_transport_waiting(state: State, p: str) -> list[Task] | None:
    if state.holds(Atom("delivered", (p,))):
        return None
    return [("pickup", p), ("drop", p)]


TAXI_METHODS: dict[str, list[Method]] = {
    "achieve": [m_achieve],
    "transport": [m_transport_in_taxi, m_transport_waiting],
}


def make_taxi_planner(terminal_reward: float = 1.0) -> GTPyhopPlanner:
    """Build a planner for the Taxi domain; ``terminal_reward`` is tR for both operators."""
    operators = tuple(
        OperatorSpec(**{**op.__dict__, "terminal_reward": terminal_reward}) for op in TAXI_OPERATORS
    )
    return GTPyhopPlanner(
        operators,
        TAXI_METHODS,
        root_task=lambda goal: [("achieve", goal)],
        signature=TaxiDomain.predicates,
        name="taxi",
    )
