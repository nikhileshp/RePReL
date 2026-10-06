import numpy as np
import pytest

from reprel.abstraction import DFOCIAbstraction, load_dfoci
from reprel.core.atoms import Atom, Literal
from reprel.core.state import State
from reprel.domains.office import OfficeConfig, OfficeDomain
from reprel.domains.office_planning import OFFICE_OPERATORS, make_office_planner
from reprel.planning.operators import OperatorInstance


def simulate(dom: OfficeDomain, s: State, plan: list[OperatorInstance]) -> State:
    for op in plan:
        nxt = op.spec.apply(s, op.binding, signature=dom.predicates)
        assert nxt is not None, f"{op} not applicable"
        s = nxt
    return s


@pytest.mark.parametrize(
    "task,expected",
    [
        ("deliver_mail", ["pickup(mail)", "deliver(mail)"]),
        ("deliver_coffee", ["pickup(coffee)", "deliver(coffee)"]),
        ("deliver_both", ["pickup(coffee)", "deliver(coffee)", "pickup(mail)", "deliver(mail)"]),
        ("visit_abcd", ["pickup(a)", "pickup(b)", "pickup(c)", "pickup(d)"]),
    ],
)
def test_plans_per_task(task: str, expected: list[str]) -> None:
    dom = OfficeDomain(OfficeConfig(task=task))
    s = dom.reset(np.random.default_rng(0))
    plan = make_office_planner().plan(s, dom.goal(s))
    assert [str(op) for op in plan] == expected
    assert dom.is_success(simulate(dom, s, plan))


def test_plan_from_mid_episode_state() -> None:
    dom = OfficeDomain(OfficeConfig(task="deliver_both"))
    s = dom.reset(np.random.default_rng(0)).with_atoms(
        add=[Atom.parse("with(mail)"), Atom.parse("delivered(coffee)")]
    )
    plan = make_office_planner().plan(s, dom.goal(s))
    assert [str(op) for op in plan] == ["deliver(mail)"]


def test_operators_match_paper() -> None:
    names = {op.name for op in OFFICE_OPERATORS}
    assert names == {"pickup", "deliver"}
    pickup = next(op for op in OFFICE_OPERATORS if op.name == "pickup")
    assert pickup.termination == (Literal.parse("with(X)"),)


def test_office_dfoci_closure_and_projection() -> None:
    spec = load_dfoci("office")
    pickup = {lit.atom.pred for lit in spec.relevant_literals("pickup")}
    assert pickup == {"at", "with", "wall", "plant", "move"}
    deliver = {lit.atom.pred for lit in spec.relevant_literals("deliver")}
    assert deliver == {"at", "with", "office", "delivered", "wall", "plant", "move"}
    dom = OfficeDomain(OfficeConfig(task="deliver_both"))
    s = dom.reset(np.random.default_rng(0)).with_atoms(add=[Atom.parse("with(coffee)")])
    ab = DFOCIAbstraction(spec, signature=dom.predicates)
    plan = make_office_planner().plan(s, dom.goal(s))
    key = ab.project(s, plan[0])  # deliver(coffee)
    dynamic = {str(a) for a in key if a.pred not in ("wall", "plant")}
    (agent_at,) = [a for a in s.atoms_with("at") if a.args[0] == "agent"]
    assert dynamic == {str(agent_at), "with(?X)", "office(l_4_4)"}
    key2 = ab.project(
        s, plan[-2]
    )  # pickup(mail): mail's location, agent location, nothing about coffee
    dynamic2 = {str(a) for a in key2 if a.pred not in ("wall", "plant")}
    assert dynamic2 == {str(agent_at), "at(?X,l_7_4)"}
