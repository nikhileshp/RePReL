import pytest

from reprel.core.atoms import Atom, Obj
from reprel.core.state import State

OBJECTS = frozenset(
    {
        Obj("taxi", "taxi"),
        Obj("p1", "passenger"),
        Obj("p2", "passenger"),
        Obj("l_0_0", "location"),
        Obj("l_0_1", "location"),
    }
)


def make_state(*atoms: str) -> State:
    return State(frozenset(Atom.parse(a) for a in atoms), OBJECTS)


def test_state_is_hashable_and_value_equal() -> None:
    a = make_state("at(taxi,l_0_0)", "at(p1,l_0_1)")
    b = make_state("at(p1,l_0_1)", "at(taxi,l_0_0)")
    assert a == b
    assert hash(a) == hash(b)
    assert len({a, b}) == 1


def test_holds_and_atoms_with() -> None:
    s = make_state("at(taxi,l_0_0)", "at(p1,l_0_1)", "dest(p1,l_0_0)")
    assert s.holds(Atom.parse("at(p1,l_0_1)"))
    assert not s.holds(Atom.parse("at(p1,l_0_0)"))
    expected = {Atom.parse("at(taxi,l_0_0)"), Atom.parse("at(p1,l_0_1)")}
    assert s.atoms_with("at") == frozenset(expected)


def test_type_of_and_objects_of_type_sorted() -> None:
    s = make_state()
    assert s.type_of("p1") == "passenger"
    assert s.objects_of_type("passenger") == ("p1", "p2")
    assert s.objects_of_type("nothing") == ()
    with pytest.raises(KeyError):
        s.type_of("ghost")


def test_with_atoms_returns_new_state_and_keeps_original() -> None:
    s = make_state("at(taxi,l_0_0)")
    t = s.with_atoms(add=[Atom.parse("at(taxi,l_0_1)")], remove=[Atom.parse("at(taxi,l_0_0)")])
    assert s.holds(Atom.parse("at(taxi,l_0_0)"))
    assert t.atoms == frozenset({Atom.parse("at(taxi,l_0_1)")})
    assert t.objects == s.objects


def test_lift_renames_bound_objects_to_roles_everywhere() -> None:
    s = make_state("at(taxi,l_0_0)", "at(p1,l_0_1)", "dest(p1,l_0_0)", "at(p2,l_0_0)")
    lifted = s.lift({"P": "p1"})
    assert lifted.atoms == frozenset(
        Atom.parse(a) for a in ["at(taxi,l_0_0)", "at(?P,l_0_1)", "dest(?P,l_0_0)", "at(p2,l_0_0)"]
    )
    assert Obj("?P", "passenger") in lifted.objects
    assert Obj("p1", "passenger") not in lifted.objects


def test_lift_with_absent_object_is_a_no_op() -> None:
    s = make_state("at(taxi,l_0_0)")
    assert s.lift({"P": "p9"}) == s


def test_substitute_renames_objects() -> None:
    s = make_state("at(p1,l_0_0)")
    t = s.substitute({"p1": "p2"})
    assert t.atoms == frozenset({Atom.parse("at(p2,l_0_0)")})


def test_diff_reports_added_and_removed() -> None:
    s = make_state("at(taxi,l_0_0)", "at(p1,l_0_1)")
    t = make_state("at(taxi,l_0_1)", "at(p1,l_0_1)")
    added, removed = s.diff(t)
    assert added == frozenset({Atom.parse("at(taxi,l_0_1)")})
    assert removed == frozenset({Atom.parse("at(taxi,l_0_0)")})


def test_to_strings_sorted_and_from_strings_round_trip() -> None:
    s = make_state("dest(p1,l_0_0)", "at(taxi,l_0_0)")
    assert s.to_strings() == ["at(taxi,l_0_0)", "dest(p1,l_0_0)"]
    assert State.from_strings(s.to_strings(), OBJECTS) == s
