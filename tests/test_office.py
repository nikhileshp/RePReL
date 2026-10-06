import numpy as np
import pytest

from reprel.core.atoms import Atom, Literal
from reprel.core.domain import make_domain
from reprel.core.state import State
from reprel.domains.office import TASKS, OfficeConfig, OfficeDomain


def A(text: str) -> Atom:
    return Atom.parse(text)


def make(task: str = "deliver_mail", seed: int = 0, **kw: object) -> tuple[OfficeDomain, State]:
    dom = OfficeDomain(OfficeConfig(task=task, **kw))  # type: ignore[arg-type]
    return dom, dom.reset(np.random.default_rng(seed))


def place(dom: OfficeDomain, s: State, loc: str) -> State:
    (at,) = [a for a in s.atoms_with("at") if a.args[0] == "agent"]
    return s.with_atoms(add=[Atom("at", ("agent", loc))], remove=[at])


def test_static_layout_items_office_plants_and_walls() -> None:
    dom, s = make()
    assert s.holds(A("at(a,l_1_1)")) and s.holds(A("at(b,l_10_1)"))
    assert s.holds(A("at(c,l_10_7)")) and s.holds(A("at(d,l_1_7)"))
    assert (
        s.holds(A("at(mail,l_7_4)"))
        and s.holds(A("at(coffee,l_3_6)"))
        and s.holds(A("at(coffee,l_8_2)"))
    )
    assert s.holds(A("office(l_4_4)")) and s.holds(A("at(office,l_4_4)"))
    assert s.holds(A("plant(l_4_1)")) and len(s.atoms_with("plant")) == 6
    # thin wall between (2,0) and (3,0); grid border at x=0
    assert s.holds(A("wall(l_2_0,right)")) and s.holds(A("wall(l_3_0,left)"))
    assert s.holds(A("wall(l_0_0,left)")) and s.holds(A("wall(l_0_0,down)"))
    assert not s.holds(A("wall(l_1_0,right)"))
    assert len(s.objects_of_type("location")) == 12 * 9


def test_reset_places_agent_on_a_cell_without_objects_and_is_deterministic() -> None:
    dom, s = make(seed=3)
    (at,) = [a for a in s.atoms_with("at") if a.args[0] == "agent"]
    assert not any(a.args[1] == at.args[1] for a in s.atoms_with("at") if a.args[0] != "agent")
    assert not s.holds(Atom("plant", (at.args[1],)))
    assert s == make(seed=3)[1] and s != make(seed=4)[1]
    assert not s.atoms_with("with") and not s.atoms_with("delivered")


def test_moves_cost_one_and_walls_cost_ten_more() -> None:
    dom, s0 = make()
    rng = np.random.default_rng(0)
    s = place(dom, s0, "l_1_0")
    t = dom.step(s, "right", rng)
    assert t.next_state.holds(A("at(agent,l_2_0)")) and t.reward == -1.0 and not t.done
    t2 = dom.step(t.next_state, "right", rng)  # wall between (2,0) and (3,0)
    assert t2.next_state == t.next_state and t2.reward == -11.0
    t3 = dom.step(s, "down", rng)  # border
    assert t3.next_state == s and t3.reward == -11.0


def test_entering_item_cell_sets_with() -> None:
    dom, s0 = make(task="visit_abcd")
    s = place(dom, s0, "l_1_2")
    t = dom.step(s, "down", np.random.default_rng(0))
    assert t.next_state.holds(A("with(a)")) and t.reward == -1.0 and not t.done


def test_office_converts_held_items_to_delivered_and_completes_task() -> None:
    dom, s0 = make(task="deliver_mail")
    s = place(dom, s0, "l_4_5").with_atoms(add=[A("with(mail)"), A("with(coffee)")])
    t = dom.step(s, "down", np.random.default_rng(0))
    nxt = t.next_state
    assert nxt.holds(A("delivered(mail)")) and nxt.holds(A("delivered(coffee)"))
    assert not nxt.holds(A("with(mail)")) and nxt.holds(A("with(office)"))
    assert t.reward == pytest.approx(99.0) and t.done and dom.is_success(nxt)


def test_plant_costs_ten_more() -> None:
    dom, s0 = make()
    s = place(dom, s0, "l_4_2")
    t = dom.step(s, "down", np.random.default_rng(0))  # plant at (4,1)
    assert t.next_state.holds(A("at(agent,l_4_1)")) and t.reward == -11.0


def test_goals_per_task() -> None:
    assert set(TASKS) == {"deliver_mail", "deliver_coffee", "deliver_both", "visit_abcd"}
    dom, s = make(task="deliver_both")
    assert dom.goal(s) == frozenset(
        {Literal.parse("delivered(mail)"), Literal.parse("delivered(coffee)")}
    )
    dom, s = make(task="visit_abcd")
    assert dom.goal(s) == frozenset(Literal.parse(f"with({x})") for x in "abcd")
    assert not dom.is_success(s)
    assert dom.is_success(s.with_atoms(add=[A(f"with({x})") for x in "abcd"]))
    with pytest.raises(ValueError):
        OfficeConfig(task="fly")


def test_registered_and_configurable() -> None:
    dom = make_domain("office", task="deliver_coffee", max_steps=50, step_reward=-2.0)
    assert isinstance(dom, OfficeDomain) and dom.max_steps == 50
    assert dom.actions == ("up", "down", "left", "right")
    s = dom.reset(np.random.default_rng(0))
    assert dom.step(place(dom, s, "l_1_2"), "up", np.random.default_rng(0)).reward == -2.0


def test_random_rollout_keeps_invariants() -> None:
    dom, s = make(task="deliver_both")
    rng = np.random.default_rng(1)
    cells = {o.name for o in s.objects if o.type == "location"}
    for _ in range(300):
        s = dom.step(s, dom.actions[int(rng.integers(4))], rng).next_state
        agent_at = [a for a in s.atoms_with("at") if a.args[0] == "agent"]
        assert len(agent_at) == 1 and agent_at[0].args[1] in cells
        for item in ("mail", "coffee"):
            assert not (s.holds(A(f"with({item})")) and s.holds(A(f"delivered({item})")))
