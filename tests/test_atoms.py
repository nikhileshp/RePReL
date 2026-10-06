import pytest

from reprel.core.atoms import Atom, Literal, Obj, is_variable, unify


def test_is_variable_uses_first_character_only() -> None:
    assert is_variable("P")
    assert is_variable("L1")
    assert not is_variable("p1")
    assert not is_variable("l_1_2")
    assert not is_variable("?P")  # lifted role marker is an object-like constant


def test_atom_parse_and_str_round_trip() -> None:
    atom = Atom.parse("at(p1,l_0_0)")
    assert atom == Atom("at", ("p1", "l_0_0"))
    assert str(atom) == "at(p1,l_0_0)"
    assert Atom.parse("at( P , L )") == Atom("at", ("P", "L"))
    assert Atom.parse("done()") == Atom("done", ())
    assert str(Atom("done", ())) == "done()"


def test_atom_parse_rejects_malformed_text() -> None:
    with pytest.raises(ValueError):
        Atom.parse("at p1 l")
    with pytest.raises(ValueError):
        Atom.parse("at(p1,l")


def test_atom_is_ground_and_variables() -> None:
    assert Atom.parse("at(p1,l_0_0)").is_ground
    lifted = Atom.parse("at(P,L1)")
    assert not lifted.is_ground
    assert lifted.variables() == ("P", "L1")


def test_atom_substitute_leaves_unbound_variables() -> None:
    atom = Atom.parse("at(P,L)")
    assert atom.substitute({"P": "p1"}) == Atom.parse("at(p1,L)")


def test_literal_parse_handles_negation() -> None:
    assert Literal.parse("in(P,T)") == Literal(Atom.parse("in(P,T)"), True)
    assert Literal.parse("not in(P,T)") == Literal(Atom.parse("in(P,T)"), False)
    assert str(Literal.parse("not in(P,T)")) == "not in(P,T)"


def test_obj_is_hashable_and_typed() -> None:
    assert Obj("p1", "passenger") == Obj("p1", "passenger")
    assert len({Obj("p1", "passenger"), Obj("p1", "passenger")}) == 1


def test_unify_binds_variables() -> None:
    theta = unify(Atom.parse("at(P,L)"), Atom.parse("at(p1,l_0_0)"))
    assert theta == {"P": "p1", "L": "l_0_0"}


def test_unify_respects_existing_binding() -> None:
    assert unify(Atom.parse("at(P,L)"), Atom.parse("at(p1,l_0_0)"), {"P": "p1"}) == {
        "P": "p1",
        "L": "l_0_0",
    }
    assert unify(Atom.parse("at(P,L)"), Atom.parse("at(p1,l_0_0)"), {"P": "p2"}) is None


def test_unify_fails_on_predicate_arity_or_constant_mismatch() -> None:
    assert unify(Atom.parse("in(P,T)"), Atom.parse("at(p1,l_0_0)")) is None
    assert unify(Atom.parse("at(P)"), Atom.parse("at(p1,l_0_0)")) is None
    assert unify(Atom.parse("at(P,l_1_1)"), Atom.parse("at(p1,l_0_0)")) is None


def test_unify_binds_same_variable_consistently() -> None:
    assert unify(Atom.parse("eq(X,X)"), Atom.parse("eq(a,b)")) is None
    assert unify(Atom.parse("eq(X,X)"), Atom.parse("eq(a,a)")) == {"X": "a"}


def test_unify_checks_types_from_signature() -> None:
    signature = {"at": ("passenger", "location")}
    types = {"p1": "passenger", "taxi": "taxi", "l_0_0": "location"}
    assert unify(
        Atom.parse("at(P,L)"), Atom.parse("at(taxi,l_0_0)"), signature=signature, types=types
    ) is None
    assert unify(
        Atom.parse("at(P,L)"), Atom.parse("at(p1,l_0_0)"), signature=signature, types=types
    ) == {"P": "p1", "L": "l_0_0"}
