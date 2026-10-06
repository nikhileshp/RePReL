import numpy as np
import pytest

from reprel.core.atoms import Atom, Literal
from reprel.core.domain import make_domain
from reprel.core.state import State
from reprel.domains.taxi import TaxiConfig, TaxiDomain
from reprel.domains.taxi_layouts import EIGHT, LAYOUTS, parse_layout


def A(text: str) -> Atom:
    return Atom.parse(text)


# ---------------------------------------------------------------- layouts
def test_parse_layout_cells_depots_and_walls() -> None:
    grid = parse_layout("wwwww\nwR Gw\nw w w\nwwwww\n")
    assert grid.height == 2 and grid.width == 3
    assert grid.free_cells == {(0, 0), (0, 1), (0, 2), (1, 0), (1, 2)}
    assert grid.depots == {"R": (0, 0), "G": (0, 2)}
    assert grid.is_blocked((1, 1))
    assert grid.is_blocked((-1, 0))
    assert grid.is_blocked((0, 3))


def test_eight_layout_matches_paper_dimensions() -> None:
    grid = LAYOUTS["eight"]
    assert (grid.height, grid.width) == (8, 8)
    assert set(grid.depots) == {"R", "G", "B", "Y"}
    assert grid.depots["R"] == (0, 0) and grid.depots["G"] == (0, 7)
    assert grid.depots["Y"] == (7, 0) and grid.depots["B"] == (7, 5)


# ---------------------------------------------------------------- instances
def reset(num_passengers: int = 1, seed: int = 0, **kw: object) -> tuple[TaxiDomain, State]:
    dom = TaxiDomain(TaxiConfig(num_passengers=num_passengers, **kw))  # type: ignore[arg-type]
    return dom, dom.reset(np.random.default_rng(seed))


def test_reset_creates_passengers_with_distinct_pickup_and_destination() -> None:
    dom, s = reset(num_passengers=3)
    assert s.objects_of_type("passenger") == ("p1", "p2", "p3")
    depots = {f"l_{r}_{c}" for r, c in dom.grid.depots.values()}
    for p in ("p1", "p2", "p3"):
        (at,) = [a for a in s.atoms_with("at") if a.args[0] == p]
        (dest,) = s.atoms_with("dest") & {a for a in s.atoms if a.args[0] == p}
        assert at.args[1] in depots and dest.args[1] in depots
        assert at.args[1] != dest.args[1]
        assert not s.holds(A(f"delivered({p})"))
        assert not s.holds(A(f"in({p},taxi)"))


def test_reset_places_taxi_on_a_free_non_depot_cell() -> None:
    dom, s = reset()
    (taxi_at,) = [a for a in s.atoms_with("at") if a.args[0] == "taxi"]
    r, c = (int(x) for x in taxi_at.args[1].split("_")[1:])
    assert (r, c) in dom.grid.free_cells
    assert (r, c) not in dom.grid.depots.values()


def test_reset_is_deterministic_per_seed() -> None:
    _, a = reset(num_passengers=2, seed=42)
    _, b = reset(num_passengers=2, seed=42)
    _, c = reset(num_passengers=2, seed=43)
    assert a == b
    assert a != c


def test_wall_atoms_match_layout() -> None:
    dom, s = reset()
    # top-left corner of the eight layout: north and west blocked, east/south open
    assert s.holds(A("wall(l_0_0,north)")) and s.holds(A("wall(l_0_0,west)"))
    assert not s.holds(A("wall(l_0_0,east)")) and not s.holds(A("wall(l_0_0,south)"))
    # column 3, rows 0-2 is an internal wall column in 'eight': l_0_2 cannot go east
    assert s.holds(A("wall(l_0_2,east)"))
    assert not s.holds(A("wall(l_3_2,east)"))
    assert "l_0_3" not in {o.name for o in s.objects}


def test_goal_and_success() -> None:
    dom, s = reset(num_passengers=2)
    expected = {Literal.parse("delivered(p1)"), Literal.parse("delivered(p2)")}
    assert dom.goal(s) == frozenset(expected)
    assert not dom.is_success(s)
    done = s.with_atoms(add=[A("delivered(p1)"), A("delivered(p2)")])
    assert dom.is_success(done)


# ---------------------------------------------------------------- dynamics helpers
def place(dom: TaxiDomain, s: State, taxi: str, p1_at: str | None, p1_dest: str) -> State:
    atoms = {a for a in s.atoms if a.pred == "wall"}
    atoms.add(A(f"at(taxi,{taxi})"))
    atoms.add(A(f"dest(p1,{p1_dest})"))
    if p1_at is not None:
        atoms.add(A(f"at(p1,{p1_at})"))
    return State(frozenset(atoms), s.objects)


def test_move_changes_taxi_location_with_step_cost() -> None:
    dom, s0 = reset()
    s = place(dom, s0, "l_2_2", "l_0_0", "l_7_0")
    rng = np.random.default_rng(0)
    t = dom.step(s, "north", rng)
    assert t.next_state.holds(A("at(taxi,l_1_2)")) and not t.next_state.holds(A("at(taxi,l_2_2)"))
    assert t.reward == pytest.approx(-0.1)
    assert not t.done
    assert s.holds(A("at(taxi,l_2_2)"))  # input not mutated


def test_blocked_move_keeps_position_without_extra_penalty() -> None:
    dom, s0 = reset()
    s = place(dom, s0, "l_0_0", "l_7_0", "l_0_7")
    t = dom.step(s, "north", np.random.default_rng(0))
    assert t.next_state == s
    assert t.reward == pytest.approx(-0.1)


def test_pickup_at_passenger_location_boards_passenger() -> None:
    dom, s0 = reset()
    s = place(dom, s0, "l_0_0", "l_0_0", "l_7_0")
    t = dom.step(s, "pickup", np.random.default_rng(0))
    assert t.next_state.holds(A("in(p1,taxi)"))
    assert not t.next_state.holds(A("at(p1,l_0_0)"))
    assert t.reward == pytest.approx(9.9)


def test_pickup_away_from_passenger_is_penalised() -> None:
    dom, s0 = reset()
    s = place(dom, s0, "l_0_1", "l_0_0", "l_7_0")
    t = dom.step(s, "pickup", np.random.default_rng(0))
    assert t.next_state == s
    assert t.reward == pytest.approx(-1.1)


def test_pickup_while_carrying_is_penalised() -> None:
    dom, s0 = reset(num_passengers=2)
    s = place(dom, s0, "l_0_0", None, "l_7_0").with_atoms(
        add=[A("in(p1,taxi)"), A("at(p2,l_0_0)"), A("dest(p2,l_7_5)")]
    )
    t = dom.step(s, "pickup", np.random.default_rng(0))
    assert t.next_state == s
    assert t.reward == pytest.approx(-1.1)


def test_pickup_with_shared_depot_boards_lowest_index_passenger() -> None:
    dom, s0 = reset(num_passengers=2)
    s = place(dom, s0, "l_0_0", "l_0_0", "l_7_0").with_atoms(
        add=[A("at(p2,l_0_0)"), A("dest(p2,l_7_5)")]
    )
    t = dom.step(s, "pickup", np.random.default_rng(0))
    assert t.next_state.holds(A("in(p1,taxi)"))
    assert t.next_state.holds(A("at(p2,l_0_0)"))
    assert not t.next_state.holds(A("in(p2,taxi)"))


def test_pickup_skips_delivered_passenger_at_same_cell() -> None:
    dom, s0 = reset(num_passengers=2)
    s = place(dom, s0, "l_7_0", "l_7_0", "l_7_0").with_atoms(
        add=[A("delivered(p1)"), A("at(p2,l_7_0)"), A("dest(p2,l_0_0)")]
    )
    t = dom.step(s, "pickup", np.random.default_rng(0))
    assert t.next_state.holds(A("in(p2,taxi)"))
    assert not t.next_state.holds(A("in(p1,taxi)"))


def test_dropoff_at_destination_delivers_and_ends_episode() -> None:
    dom, s0 = reset()
    s = place(dom, s0, "l_7_0", None, "l_7_0").with_atoms(add=[A("in(p1,taxi)")])
    t = dom.step(s, "dropoff", np.random.default_rng(0))
    assert t.next_state.holds(A("delivered(p1)"))
    assert t.next_state.holds(A("at(p1,l_7_0)"))
    assert not t.next_state.holds(A("in(p1,taxi)"))
    assert t.reward == pytest.approx(19.9)
    assert t.done and dom.is_success(t.next_state)


def test_dropoff_elsewhere_or_empty_is_penalised() -> None:
    dom, s0 = reset()
    carrying = place(dom, s0, "l_3_3", None, "l_7_0").with_atoms(add=[A("in(p1,taxi)")])
    t = dom.step(carrying, "dropoff", np.random.default_rng(0))
    assert t.next_state == carrying and t.reward == pytest.approx(-1.1)
    empty = place(dom, s0, "l_7_0", "l_0_0", "l_7_0")
    t2 = dom.step(empty, "dropoff", np.random.default_rng(0))
    assert t2.next_state == empty and t2.reward == pytest.approx(-1.1)


def test_episode_not_done_until_all_passengers_delivered() -> None:
    dom, s0 = reset(num_passengers=2)
    s = place(dom, s0, "l_7_0", None, "l_7_0").with_atoms(
        add=[A("in(p1,taxi)"), A("at(p2,l_0_0)"), A("dest(p2,l_7_5)")]
    )
    t = dom.step(s, "dropoff", np.random.default_rng(0))
    assert t.next_state.holds(A("delivered(p1)"))
    assert not t.done


def test_registered_with_config_kwargs() -> None:
    dom = make_domain("taxi", num_passengers=2, max_steps=50, layout="five")
    assert isinstance(dom, TaxiDomain)
    assert dom.cfg.num_passengers == 2 and dom.max_steps == 50
    assert (dom.grid.height, dom.grid.width) == (5, 5)
    assert dom.actions == ("north", "south", "west", "east", "pickup", "dropoff")


def test_rewards_are_configurable() -> None:
    dom, s0 = reset(step_reward=-1.0, pickup_reward=0.0, illegal_reward=-5.0)
    s = place(dom, s0, "l_0_0", "l_0_0", "l_7_0")
    assert dom.step(s, "pickup", np.random.default_rng(0)).reward == pytest.approx(-1.0)
    assert dom.step(s, "dropoff", np.random.default_rng(0)).reward == pytest.approx(-6.0)


def test_random_rollout_preserves_state_invariants() -> None:
    dom = TaxiDomain(TaxiConfig(num_passengers=3))
    rng = np.random.default_rng(5)
    s = dom.reset(rng)
    free = {f"l_{r}_{c}" for r, c in dom.grid.free_cells}
    for _ in range(500):
        s = dom.step(s, dom.actions[int(rng.integers(dom.n_actions))], rng).next_state
        taxi_at = [a for a in s.atoms_with("at") if a.args[0] == "taxi"]
        assert len(taxi_at) == 1 and taxi_at[0].args[1] in free
        assert len(s.atoms_with("in")) <= 1
        for p in s.objects_of_type("passenger"):
            located = [a for a in s.atoms_with("at") if a.args[0] == p]
            aboard = s.holds(A(f"in({p},taxi)"))
            assert len(located) + int(aboard) == 1


def test_grid_is_hashable() -> None:
    assert hash(LAYOUTS["eight"]) == hash(parse_layout(EIGHT))


def test_config_rejects_non_positive_passenger_count() -> None:
    with pytest.raises(ValueError):
        TaxiConfig(num_passengers=0)
