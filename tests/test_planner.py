import numpy as np
import pytest

from reprel.core.atoms import Atom, Literal
from reprel.core.state import State
from reprel.domains.taxi import TaxiConfig, TaxiDomain
from reprel.domains.taxi_planning import TAXI_OPERATORS, make_taxi_planner
from reprel.planning.operators import Goal, OperatorInstance
from reprel.planning.planner import Planner, PlanningFailure


def simulate(domain: TaxiDomain, state: State, plan: list[OperatorInstance]) -> State:
    for op in plan:
        nxt = op.spec.apply(state, op.binding, signature=domain.predicates)
        assert nxt is not None, f"{op} not applicable in {state.to_strings()}"
        state = nxt
    return state


def test_planner_implements_interface() -> None:
    assert isinstance(make_taxi_planner(), Planner)
    assert {op.name for op in TAXI_OPERATORS} == {"pickup", "drop"}


@pytest.mark.parametrize("n", [1, 2, 3, 5])
def test_plan_is_valid_and_reaches_goal(n: int) -> None:
    dom = TaxiDomain(TaxiConfig(num_passengers=n))
    s = dom.reset(np.random.default_rng(n))
    plan = make_taxi_planner().plan(s, dom.goal(s))
    assert [op.name for op in plan] == ["pickup", "drop"] * n
    assert [op.args[0] for op in plan[::2]] == [f"p{i}" for i in range(1, n + 1)]
    final = simulate(dom, s, plan)
    assert all(final.holds(lit.atom) for lit in dom.goal(s))
    assert dom.is_success(final)


def test_plan_from_state_with_passenger_in_taxi_starts_with_drop() -> None:
    dom = TaxiDomain(TaxiConfig(num_passengers=2))
    s = dom.reset(np.random.default_rng(0))
    (at_p1,) = [a for a in s.atoms_with("at") if a.args[0] == "p1"]
    s = s.with_atoms(add=[Atom.parse("in(p1,taxi)")], remove=[at_p1])
    plan = make_taxi_planner().plan(s, dom.goal(s))
    assert [str(op) for op in plan] == ["drop(p1)", "pickup(p2)", "drop(p2)"]


def test_plan_skips_delivered_passengers() -> None:
    dom = TaxiDomain(TaxiConfig(num_passengers=2))
    s = dom.reset(np.random.default_rng(1)).with_atoms(add=[Atom.parse("delivered(p1)")])
    plan = make_taxi_planner().plan(s, dom.goal(s))
    assert [str(op) for op in plan] == ["pickup(p2)", "drop(p2)"]


def test_plan_for_satisfied_goal_is_empty() -> None:
    dom = TaxiDomain(TaxiConfig(num_passengers=1))
    s = dom.reset(np.random.default_rng(2)).with_atoms(add=[Atom.parse("delivered(p1)")])
    assert make_taxi_planner().plan(s, dom.goal(s)) == []


def test_plan_subgoal_subset() -> None:
    dom = TaxiDomain(TaxiConfig(num_passengers=3))
    s = dom.reset(np.random.default_rng(3))
    plan = make_taxi_planner().plan(s, frozenset({Literal.parse("delivered(p2)")}))
    assert [str(op) for op in plan] == ["pickup(p2)", "drop(p2)"]


def test_unachievable_goal_raises_planning_failure() -> None:
    dom = TaxiDomain(TaxiConfig(num_passengers=1))
    s = dom.reset(np.random.default_rng(4))
    with pytest.raises(PlanningFailure):
        make_taxi_planner().plan(s, frozenset({Literal.parse("delivered(p9)")}))


def test_planner_is_reusable_and_deterministic() -> None:
    dom = TaxiDomain(TaxiConfig(num_passengers=2))
    planner = make_taxi_planner()
    s = dom.reset(np.random.default_rng(5))
    assert planner.plan(s, dom.goal(s)) == planner.plan(s, dom.goal(s))
    s2 = dom.reset(np.random.default_rng(6))
    assert len(planner.plan(s2, dom.goal(s2))) == 4


def test_planning_prints_nothing(capsys: pytest.CaptureFixture[str]) -> None:
    dom = TaxiDomain(TaxiConfig(num_passengers=1))
    s = dom.reset(np.random.default_rng(7))
    make_taxi_planner().plan(s, dom.goal(s))
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""


def board(s: State, p: str) -> State:
    (at_p,) = [a for a in s.atoms_with("at") if a.args[0] == p]
    return s.with_atoms(add=[Atom.parse(f"in({p},taxi)")], remove=[at_p])


def test_replan_with_other_passenger_aboard_drops_it_first() -> None:
    # The pickup agent may board whoever is waiting at the taxi's cell; the planner must cope.
    dom = TaxiDomain(TaxiConfig(num_passengers=2))
    s = board(dom.reset(np.random.default_rng(8)), "p2")
    plan = make_taxi_planner().plan(s, dom.goal(s))
    assert [str(op) for op in plan] == ["drop(p2)", "pickup(p1)", "drop(p1)"]
    assert dom.is_success(simulate(dom, s, plan))


def test_replan_with_non_goal_passenger_aboard_still_frees_the_taxi() -> None:
    dom = TaxiDomain(TaxiConfig(num_passengers=2))
    s = board(dom.reset(np.random.default_rng(9)), "p2")
    plan = make_taxi_planner().plan(s, frozenset({Literal.parse("delivered(p1)")}))
    assert [str(op) for op in plan] == ["drop(p2)", "pickup(p1)", "drop(p1)"]


def test_pickup_is_not_applicable_while_carrying() -> None:
    dom = TaxiDomain(TaxiConfig(num_passengers=2))
    s = board(dom.reset(np.random.default_rng(8)), "p2")
    pickup = next(op for op in TAXI_OPERATORS if op.name == "pickup")
    assert pickup.apply(s, {"P": "p1"}, signature=dom.predicates) is None


def test_method_exceptions_are_not_reported_as_planning_failure() -> None:
    from reprel.planning.htn_gtpyhop import GTPyhopPlanner

    def broken(state: State, goal: Goal) -> list[tuple[str, ...]]:
        raise KeyError("bug in method")

    planner = GTPyhopPlanner(
        TAXI_OPERATORS, {"achieve": [broken]}, lambda g: [("achieve", g)], name="broken"
    )
    dom = TaxiDomain(TaxiConfig(num_passengers=1))
    s = dom.reset(np.random.default_rng(0))
    with pytest.raises(RuntimeError) as excinfo:
        planner.plan(s, dom.goal(s))
    assert not isinstance(excinfo.value, PlanningFailure)


def test_backtracking_over_methods() -> None:
    """If the first applicable method leads to a dead end, the next one is tried."""
    from reprel.planning.htn_gtpyhop import GTPyhopPlanner

    def dead_end(state: State, goal: Goal) -> list[tuple[str, ...]]:
        return [("pickup", "p9")]  # p9 does not exist -> action fails

    planner = GTPyhopPlanner(
        TAXI_OPERATORS,
        {"achieve": [dead_end, lambda s, g: [("pickup", "p1"), ("drop", "p1")]]},
        lambda g: [("achieve", g)],
        signature=TaxiDomain.predicates,
        name="backtrack",
    )
    dom = TaxiDomain(TaxiConfig(num_passengers=1))
    s = dom.reset(np.random.default_rng(0))
    assert [str(op) for op in planner.plan(s, dom.goal(s))] == ["pickup(p1)", "drop(p1)"]
