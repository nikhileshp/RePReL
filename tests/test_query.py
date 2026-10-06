from reprel.core.atoms import Atom, Literal, Obj
from reprel.core.query import matches, satisfied
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
SIGNATURE = {
    "at": ("object", "location"),
    "in": ("passenger", "taxi"),
    "dest": ("passenger", "location"),
}


def make_state(*atoms: str) -> State:
    return State(frozenset(Atom.parse(a) for a in atoms), OBJECTS)


def lits(*texts: str) -> tuple[Literal, ...]:
    return tuple(Literal.parse(t) for t in texts)


def test_matches_enumerates_all_bindings_for_positive_literals() -> None:
    s = make_state("at(p1,l_0_0)", "at(p2,l_0_1)", "at(taxi,l_0_1)")
    found = list(matches(lits("at(P,L)"), s, signature=SIGNATURE))
    assert sorted(b["P"] for b in found) == ["p1", "p2", "taxi"]


def test_matches_respects_initial_binding_and_shared_variables() -> None:
    s = make_state("at(p1,l_0_0)", "at(p2,l_0_1)", "at(taxi,l_0_1)")
    found = list(matches(lits("at(P,L)", "at(taxi,L)"), s, {"P": "p2"}, signature=SIGNATURE))
    assert found == [{"P": "p2", "L": "l_0_1"}]
    assert list(matches(lits("at(P,L)", "at(taxi,L)"), s, {"P": "p1"}, signature=SIGNATURE)) == []


def test_matches_handles_negative_literals_after_binding() -> None:
    s = make_state("at(p1,l_0_0)", "at(p2,l_0_1)", "in(p2,taxi)")
    found = list(matches(lits("at(P,L)", "not in(P,taxi)"), s, signature=SIGNATURE))
    assert found == [{"P": "p1", "L": "l_0_0"}]


def test_unbound_negative_literal_means_no_atom_unifies() -> None:
    s = make_state("at(p1,l_0_0)", "in(p2,taxi)")
    assert list(matches(lits("not in(X,taxi)"), s, signature=SIGNATURE)) == []
    assert list(matches(lits("not dest(X,L)"), s, signature=SIGNATURE)) == [{}]


def test_matches_uses_types_from_state_objects() -> None:
    s = make_state("at(p1,l_0_0)", "at(taxi,l_0_1)")
    # ``in`` requires a passenger in slot 0; the taxi must not bind to P
    s2 = s.with_atoms(add=[Atom.parse("in(taxi,taxi)")])
    assert list(matches(lits("in(P,taxi)"), s2, signature=SIGNATURE)) == []


def test_satisfied_is_true_when_some_binding_exists() -> None:
    s = make_state("at(p1,l_0_0)", "in(p2,taxi)")
    assert satisfied(lits("in(P,taxi)"), s, {"P": "p2"}, signature=SIGNATURE)
    assert not satisfied(lits("in(P,taxi)"), s, {"P": "p1"}, signature=SIGNATURE)
    assert satisfied(lits(), s, signature=SIGNATURE)
