"""HTN model of Box World: ``pick_key(K)`` and ``unlock(L)`` operators, box-chain methods."""

from __future__ import annotations

from dataclasses import replace

from reprel.core.atoms import Atom, Literal
from reprel.core.state import State
from reprel.planning.htn_gtpyhop import GTPyhopPlanner, Method, Task
from reprel.planning.operators import Goal, OperatorSpec

from .boxworld import GEM, BoxWorldDomain


def _lits(*texts: str) -> tuple[Literal, ...]:
    return tuple(Literal.parse(t) for t in texts)


def _atoms(*texts: str) -> tuple[Atom, ...]:
    return tuple(Atom.parse(t) for t in texts)


PICK_KEY = OperatorSpec(
    name="pick_key",
    params=(("K", "key"),),
    preconditions=_lits("at(K,C)", "not locked(K)", "not own(K)"),
    add=_atoms("own(K)"),
    delete=_atoms("at(K,C)"),
    termination=_lits("own(K)"),
)

UNLOCK = OperatorSpec(
    name="unlock",
    params=(("L", "lock"),),
    preconditions=_lits("at(L,C)", "inside(L,K)", "color(L,Col)", "own(K2)", "color(K2,Col)"),
    add=_atoms("open(L)"),
    delete=_atoms("at(L,C)", "own(K2)", "locked(K)"),
    termination=_lits("open(L)"),
)

BOXWORLD_OPERATORS: tuple[OperatorSpec, ...] = (PICK_KEY, UNLOCK)


def _lock_holding(state: State, obj: str) -> str | None:
    for atom in state.atoms_with("inside"):
        if atom.args[1] == obj and not state.holds(Atom("open", (atom.args[0],))):
            return atom.args[0]
    return None


def _key_for(state: State, lock: str) -> str:
    colour = next(a.args[1] for a in state.atoms_with("color") if a.args[0] == lock)
    return next(
        a.args[0] for a in state.atoms_with("color") if a.args[1] == colour and a.args[0] != lock
    )


def m_achieve(state: State, goal: Goal) -> list[Task] | None:
    if state.holds(Atom("own", (GEM,))):
        return []
    return [("obtain", GEM)]


def m_obtain(state: State, obj: str) -> list[Task] | None:
    """Get hold of ``obj``: open the box it is locked in first, which needs the matching key."""
    if state.holds(Atom("own", (obj,))):
        return []
    if not state.holds(Atom("locked", (obj,))):
        return [("pick_key", obj)]
    lock = _lock_holding(state, obj)
    if lock is None:
        return None
    key = _key_for(state, lock)
    return [("obtain", key), ("unlock", lock), ("pick_key", obj)]


BOXWORLD_METHODS: dict[str, list[Method]] = {"achieve": [m_achieve], "obtain": [m_obtain]}


def make_boxworld_planner(terminal_reward: float = 1.0) -> GTPyhopPlanner:
    operators = tuple(replace(op, terminal_reward=terminal_reward) for op in BOXWORLD_OPERATORS)
    return GTPyhopPlanner(
        operators,
        BOXWORLD_METHODS,
        root_task=lambda goal: [("achieve", goal)],
        signature=BoxWorldDomain.predicates,
        name="boxworld",
    )
