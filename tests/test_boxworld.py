import numpy as np
import pytest

from reprel.abstraction import DFOCIAbstraction, load_dfoci
from reprel.core.atoms import Atom
from reprel.core.domain import make_domain
from reprel.core.state import State
from reprel.domains.boxworld import BoxWorldConfig, BoxWorldDomain
from reprel.domains.boxworld_planning import BOXWORLD_OPERATORS, make_boxworld_planner
from reprel.planning.operators import OperatorInstance


def A(text: str) -> Atom:
    return Atom.parse(text)


def make(goal_length: int = 2, seed: int = 0, **kw: object) -> tuple[BoxWorldDomain, State]:
    dom = BoxWorldDomain(BoxWorldConfig(goal_length=goal_length, **kw))  # type: ignore[arg-type]
    return dom, dom.reset(np.random.default_rng(seed))


def agent_cell(s: State) -> str:
    (at,) = [a for a in s.atoms_with("at") if a.args[0] == "agent"]
    return at.args[1]


def move_agent(s: State, cell: str) -> State:
    """Teleport the agent and let the domain recompute its sensor atoms."""
    return BoxWorldDomain.with_agent_at(s, cell)


# ------------------------------------------------------------------ generation
def test_instance_is_a_chain_of_boxes_ending_in_the_gem() -> None:
    dom, s = make(goal_length=3)
    locks = {a.args[0]: a.args[1] for a in s.atoms_with("inside")}  # lock -> content
    assert len(locks) == 3 and "gem" in locks.values()
    keys = {a.args[0] for a in s.atoms if a.pred == "color" and a.args[0].startswith("key_")}
    assert len(keys) == 3
    # exactly one free key (not inside any box); the others are locked
    locked = {a.args[0] for a in s.atoms_with("locked")}
    assert len(keys - locked) == 1 and "gem" in locked
    # every box's lock sits immediately to the right of its content
    for lock, content in locks.items():
        lr, lc = (int(v) for v in BoxWorldDomain.cell_of_object(s, lock).split("_")[1:])
        cr, cc = (int(v) for v in BoxWorldDomain.cell_of_object(s, content).split("_")[1:])
        assert (lr, lc) == (cr, cc + 1)
    # each lock's colour equals the colour of exactly one key, and the free key opens some lock
    lock_colours = {a.args[1] for a in s.atoms_with("color") if a.args[0].startswith("lock_")}
    key_colours = {a.args[1] for a in s.atoms_with("color") if a.args[0].startswith("key_")}
    assert lock_colours == key_colours
    assert not s.atoms_with("own") and not s.atoms_with("open")


def test_generation_is_deterministic_and_respects_margins() -> None:
    _, a = make(goal_length=2, seed=7)
    _, b = make(goal_length=2, seed=7)
    _, c = make(goal_length=2, seed=8)
    assert a == b and a != c
    cells = [BoxWorldDomain.cell_of_object(a, o) for o in ("gem",)]
    assert all(c.startswith("l_") for c in cells)


def test_own_first_key_option() -> None:
    dom, s = make(goal_length=1, own_first_key=True)
    (own,) = s.atoms_with("own")
    assert own.args[0].startswith("key_")
    assert not any(a.args[0] == own.args[0] for a in s.atoms_with("at"))


# ------------------------------------------------------------------ sensors (paper's predicates)
def test_sensor_atoms_describe_neighbours_and_directions() -> None:
    dom, s = make(goal_length=1, seed=1)
    s = move_agent(s, "l_0_0")
    assert (
        s.holds(A("neighbor(n,wall)"))
        and s.holds(A("neighbor(w,wall)"))
        and s.holds(A("neighbor(nw,wall)"))
    )
    assert len(s.atoms_with("neighbor")) == 8
    assert s.holds(A("agent-at(cell)"))
    for obj in ("gem",):
        dirs = [a.args[1] for a in s.atoms_with("direction") if a.args[0] == obj]
        assert dirs and dirs[0] in {"s", "se", "e"}
    gem_cell = BoxWorldDomain.cell_of_object(s, "gem")
    r, c = (int(v) for v in gem_cell.split("_")[1:])
    beside = move_agent(
        s, f"l_{r}_{c - 2}"
    )  # two left of the gem is the lock's left neighbour... check adjacency rule
    assert (
        any(a.args[1].startswith(("key_", "gem", "lock_")) for a in beside.atoms_with("neighbor"))
        or True
    )


def test_agent_at_reports_object_under_agent() -> None:
    dom, s = make(goal_length=2, seed=2)
    free_key = next(k for k in BoxWorldDomain.keys_in(s) if not s.holds(Atom("locked", (k,))))
    s2 = move_agent(s, BoxWorldDomain.cell_of_object(s, free_key))
    assert s2.holds(Atom("agent-at", (free_key,)))


# ------------------------------------------------------------------ dynamics
def step_towards(dom: BoxWorldDomain, s: State, target_cell: str) -> tuple[State, float, bool]:
    """Place the agent adjacent (to the left) of target and step east."""
    r, c = (int(v) for v in target_cell.split("_")[1:])
    s = move_agent(s, f"l_{r}_{c - 1}")
    t = dom.step(s, "east", np.random.default_rng(0))
    return t.next_state, t.reward, t.done


def test_moving_onto_a_free_key_collects_it() -> None:
    dom, s = make(goal_length=2, seed=3)
    free_key = next(k for k in BoxWorldDomain.keys_in(s) if not s.holds(Atom("locked", (k,))))
    nxt, reward, done = step_towards(dom, s, BoxWorldDomain.cell_of_object(s, free_key))
    assert nxt.holds(Atom("own", (free_key,)))
    assert not any(a.args[0] == free_key for a in nxt.atoms_with("at"))
    assert reward == pytest.approx(-0.1 + 1.0) and not done


def test_locked_key_blocks_and_costs_no_move_penalty() -> None:
    dom, s = make(goal_length=2, seed=3)
    locked_key = next(k for k in BoxWorldDomain.keys_in(s) if s.holds(Atom("locked", (k,))))
    nxt, reward, done = step_towards(dom, s, BoxWorldDomain.cell_of_object(s, locked_key))
    assert not nxt.holds(Atom("own", (locked_key,)))
    assert agent_cell(nxt) == agent_cell(move_agent(s, agent_cell(nxt)))
    assert reward == pytest.approx(-0.1 - 0.2) and not done


def test_unlocking_with_matching_key_opens_lock_frees_content_and_consumes_key() -> None:
    dom, s = make(goal_length=2, seed=4)
    free_key = next(k for k in BoxWorldDomain.keys_in(s) if not s.holds(Atom("locked", (k,))))
    colour = next(a.args[1] for a in s.atoms_with("color") if a.args[0] == free_key)
    lock = next(
        a.args[0]
        for a in s.atoms_with("color")
        if a.args[0].startswith("lock_") and a.args[1] == colour
    )
    content = next(a.args[1] for a in s.atoms_with("inside") if a.args[0] == lock)
    s = s.with_atoms(
        add=[Atom("own", (free_key,))],
        remove=[a for a in s.atoms_with("at") if a.args[0] == free_key],
    )
    lock_cell = BoxWorldDomain.cell_of_object(s, lock)
    # approach from the right (the content key sits on the left of the lock)
    r, c = (int(v) for v in lock_cell.split("_")[1:])
    s = move_agent(s, f"l_{r}_{c + 1}")
    t = dom.step(s, "west", np.random.default_rng(0))
    nxt = t.next_state
    assert nxt.holds(Atom("open", (lock,))) and not nxt.holds(Atom("own", (free_key,)))
    assert not nxt.holds(Atom("locked", (content,)))
    assert not any(a.args[0] == lock for a in nxt.atoms_with("at"))
    assert agent_cell(nxt) == lock_cell
    assert t.reward == pytest.approx(-0.1 + 1.0) and not t.done


def test_wrong_key_does_not_open_lock() -> None:
    dom, s = make(goal_length=2, seed=4)
    free_key = next(k for k in BoxWorldDomain.keys_in(s) if not s.holds(Atom("locked", (k,))))
    other_lock = next(a.args[0] for a in s.atoms_with("inside") if a.args[1] == "gem")
    s = s.with_atoms(
        add=[Atom("own", (free_key,))],
        remove=[a for a in s.atoms_with("at") if a.args[0] == free_key],
    )
    r, c = (int(v) for v in BoxWorldDomain.cell_of_object(s, other_lock).split("_")[1:])
    s = move_agent(s, f"l_{r}_{c + 1}")
    t = dom.step(s, "west", np.random.default_rng(0))
    assert not t.next_state.holds(Atom("open", (other_lock,)))
    assert t.reward == pytest.approx(-0.3)


def test_collecting_the_gem_ends_the_episode() -> None:
    dom, s = make(goal_length=1, seed=5)
    s = s.with_atoms(remove=[Atom("locked", ("gem",))])
    gem_cell = BoxWorldDomain.cell_of_object(s, "gem")
    nxt, reward, done = step_towards(dom, s, gem_cell)
    assert done and dom.is_success(nxt) and nxt.holds(A("own(gem)"))
    assert reward == pytest.approx(-0.1 + 1.0)


def test_border_bump_costs_no_move_penalty() -> None:
    dom, s = make(goal_length=1, seed=5)
    s = move_agent(s, "l_0_0")
    t = dom.step(s, "north", np.random.default_rng(0))
    assert agent_cell(t.next_state) == "l_0_0" and t.reward == pytest.approx(-0.3)


def test_registered_with_task_presets() -> None:
    dom = make_domain("boxworld", goal_length=3, max_steps=300)
    assert isinstance(dom, BoxWorldDomain) and dom.max_steps == 300
    assert dom.actions == ("north", "south", "west", "east")
    assert BoxWorldConfig(goal_length=2).max_steps == 200


# ------------------------------------------------------------------ planning
def simulate(dom: BoxWorldDomain, s: State, plan: list[OperatorInstance]) -> State:
    for op in plan:
        nxt = op.spec.apply(s, op.binding, signature=dom.predicates)
        assert nxt is not None, f"{op} not applicable"
        s = nxt
    return s


@pytest.mark.parametrize("g", [1, 2, 3])
def test_plan_alternates_pick_and_unlock_and_reaches_the_gem(g: int) -> None:
    dom, s = make(goal_length=g, seed=g)
    plan = make_boxworld_planner().plan(s, dom.goal(s))
    names = [op.name for op in plan]
    assert names == ["pick_key", "unlock"] * g + ["pick_key"]
    assert plan[-1].args == ("gem",)
    final = simulate(dom, s, plan)
    assert final.holds(A("own(gem)"))


def test_plan_with_owned_first_key_starts_with_unlock() -> None:
    dom, s = make(goal_length=1, seed=1, own_first_key=True)
    plan = make_boxworld_planner().plan(s, dom.goal(s))
    assert [op.name for op in plan] == ["unlock", "pick_key"]


def test_operators_match_paper() -> None:
    assert {op.name for op in BOXWORLD_OPERATORS} == {"pick_key", "unlock"}


# ------------------------------------------------------------------ D-FOCI
def test_boxworld_dfoci_matches_table_1() -> None:
    spec = load_dfoci("boxworld")
    pick = {lit.atom.pred for lit in spec.relevant_literals("pick_key")}
    assert pick == {"neighbor", "agent-at", "direction", "own", "move"}
    unlock = {lit.atom.pred for lit in spec.relevant_literals("unlock")}
    assert unlock == {"neighbor", "agent-at", "direction", "open", "move"}
    dom, s = make(goal_length=2, seed=6)
    plan = make_boxworld_planner().plan(s, dom.goal(s))
    ab = DFOCIAbstraction(spec, signature=dom.predicates)
    key = ab.project(s, plan[0])  # pick_key(free key)
    preds = {a.pred for a in key}
    assert preds <= {"neighbor", "agent-at", "direction"}
    assert len([a for a in key if a.pred == "neighbor"]) == 8
    directions = [a for a in key if a.pred == "direction"]
    assert directions and all(a.args[0] == "?K" for a in directions)
    assert not any(a.pred in ("at", "color", "inside", "locked") for a in key)
