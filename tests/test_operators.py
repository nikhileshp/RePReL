import pytest

from reprel.core.atoms import Atom, Literal, Obj
from reprel.core.state import State
from reprel.planning.operators import OperatorInstance, OperatorSpec

OBJECTS = frozenset(
    {
        Obj("taxi", "taxi"),
        Obj("p1", "passenger"),
        Obj("p2", "passenger"),
        Obj("l_0_0", "location"),
        Obj("l_0_1", "location"),
    }
)
SIGNATURE = {
    "at": ("object", "location"),
    "in": ("passenger", "taxi"),
    "dest": ("passenger", "location"),
    "delivered": ("passenger",),
}


def make_state(*atoms: str) -> State:
    return State(frozenset(Atom.parse(a) for a in atoms), OBJECTS)


PICKUP = OperatorSpec(
    name="pickup",
    params=(("P", "passenger"),),
    preconditions=tuple(
        Literal.parse(t) for t in ["at(taxi,L0)", "at(P,L)", "not in(P,taxi)", "not delivered(P)"]
    ),
    add=(Atom.parse("in(P,taxi)"), Atom.parse("at(taxi,L)")),
    delete=(Atom.parse("at(P,L)"), Atom.parse("at(taxi,L0)")),
    termination=(Literal.parse("in(P,taxi)"),),
    terminal_reward=10.0,
)


def test_spec_roles_and_instantiate() -> None:
    assert PICKUP.roles == ("P",)
    inst = PICKUP.instantiate(("p1",))
    assert inst.name == "pickup" and inst.args == ("p1",)
    assert inst.binding == {"P": "p1"}
    assert str(inst) == "pickup(p1)"
    assert inst.key == ("pickup", ("p1",))
    assert hash(inst) == hash(PICKUP.instantiate(("p1",)))
    with pytest.raises(ValueError):
        PICKUP.instantiate(("p1", "p2"))


def test_applicable_binds_existential_variables() -> None:
    s = make_state("at(taxi,l_0_1)", "at(p1,l_0_0)", "at(p2,l_0_1)")
    theta = PICKUP.applicable(s, {"P": "p1"}, signature=SIGNATURE)
    assert theta == {"P": "p1", "L0": "l_0_1", "L": "l_0_0"}
    boarded = s.with_atoms(add=[Atom.parse("in(p1,taxi)")])
    assert PICKUP.applicable(boarded, {"P": "p1"}, signature=SIGNATURE) is None


def test_apply_uses_the_found_binding() -> None:
    s = make_state("at(taxi,l_0_1)", "at(p1,l_0_0)", "at(p2,l_0_1)")
    nxt = PICKUP.apply(s, {"P": "p1"}, signature=SIGNATURE)
    assert nxt is not None
    assert nxt.atoms == frozenset(
        Atom.parse(a) for a in ["in(p1,taxi)", "at(taxi,l_0_0)", "at(p2,l_0_1)"]
    )
    assert PICKUP.apply(s, {"P": "taxi"}, signature=SIGNATURE) is None


def test_apply_when_delete_and_add_overlap_keeps_added_atom() -> None:
    s = make_state("at(taxi,l_0_0)", "at(p1,l_0_0)")
    nxt = PICKUP.apply(s, {"P": "p1"}, signature=SIGNATURE)
    assert nxt is not None and nxt.holds(Atom.parse("at(taxi,l_0_0)"))


def test_instance_termination_and_subtask_reward() -> None:
    inst = PICKUP.instantiate(("p1",))
    before = make_state("at(taxi,l_0_0)", "at(p1,l_0_0)")
    after = make_state("at(taxi,l_0_0)", "in(p1,taxi)")
    assert not inst.is_terminated(before, signature=SIGNATURE)
    assert inst.is_terminated(after, signature=SIGNATURE)
    assert inst.subtask_reward(-0.1, before, signature=SIGNATURE) == pytest.approx(-0.1)
    assert inst.subtask_reward(9.9, after, signature=SIGNATURE) == pytest.approx(19.9)


def test_instance_is_frozen() -> None:
    inst = OperatorInstance(PICKUP, ("p1",))
    with pytest.raises(AttributeError):
        inst.args = ("p2",)  # type: ignore[misc]
