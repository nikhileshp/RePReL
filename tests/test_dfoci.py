import numpy as np
import pytest

from reprel.abstraction import make_abstraction
from reprel.abstraction.dfoci import DFOCIAbstraction, DFOCISpec, load_dfoci
from reprel.core.atoms import Atom, Literal, Obj
from reprel.core.state import State
from reprel.domains.taxi import TaxiConfig, TaxiDomain
from reprel.domains.taxi_planning import DROP, PICKUP


def preds(literals: frozenset[Literal]) -> set[str]:
    return {lit.atom.pred for lit in literals}


# ------------------------------------------------------------------ spec loading
def test_load_bundled_taxi_spec() -> None:
    spec = load_dfoci("taxi")
    assert spec.domain == "taxi" and spec.source == "hand"
    assert {s.operator for s in spec.statements if s.operator} == {"pickup", "drop"}
    assert set(spec.operators) == {"pickup", "drop"}


def test_from_dict_and_to_dict_round_trip() -> None:
    spec = load_dfoci("taxi")
    again = DFOCISpec.from_dict(spec.to_dict())
    assert again == spec


def test_source_learned_is_accepted_and_bad_source_rejected() -> None:
    data = load_dfoci("taxi").to_dict()
    data["source"] = "learned"
    assert DFOCISpec.from_dict(data).source == "learned"
    data["source"] = "guess"
    with pytest.raises(ValueError):
        DFOCISpec.from_dict(data)


def test_statement_requires_target_and_lists() -> None:
    with pytest.raises(ValueError):
        DFOCISpec.from_dict({"domain": "x", "statements": [{"influences": ["a()"]}]})


# ------------------------------------------------------------------ closure
def test_taxi_pickup_closure_matches_paper_table_1() -> None:
    spec = load_dfoci("taxi")
    relevant = spec.relevant_literals("pickup")
    assert preds(relevant) == {"at", "in", "wall", "move"}
    assert Literal.parse("at(P,L)") in relevant
    assert Literal.parse("in(P,taxi)") in relevant
    assert Literal.parse("at(taxi,L1)") in relevant
    assert "dest" not in preds(relevant) and "delivered" not in preds(relevant)


def test_taxi_drop_closure_matches_paper_table_1() -> None:
    relevant = load_dfoci("taxi").relevant_literals("drop")
    assert preds(relevant) == {"at", "in", "dest", "delivered", "wall", "move"}
    assert Literal.parse("at(P,L)") not in relevant  # passenger pickup spot is irrelevant


def test_closure_follows_chains_and_if_contexts() -> None:
    spec = DFOCISpec.from_dict(
        {
            "domain": "toy",
            "reward_parents": ["a(X)"],
            "statements": [
                {"influences": ["b(X)"], "target": "a(X)"},
                {"operator": "op(X)", "if": ["ctx(X)"], "influences": ["c(X)"], "target": "b(X)"},
                {"operator": "other(X)", "influences": ["z(X)"], "target": "b(X)"},
                {"influences": ["d(X)"], "target": "unrelated(X)"},
            ],
            "operators": {"op(X)": {"reward_parents": [], "termination_parents": ["t(X)"]}},
        }
    )
    assert preds(spec.relevant_literals("op")) == {"a", "b", "c", "ctx", "t"}
    assert preds(spec.relevant_literals("op", max_depth=0)) == {"a", "t"}
    assert preds(spec.relevant_literals("op", max_depth=1)) == {"a", "b", "t"}


def test_closure_with_fuel_keeps_fuel_for_pickup() -> None:
    data = load_dfoci("taxi").to_dict()
    data["reward_parents"].append("fuel(F)")
    data["statements"].append({"influences": ["fuel(F)", "move(Dir)"], "target": "fuel(F2)"})
    spec = DFOCISpec.from_dict(data)
    assert "fuel" in preds(spec.relevant_literals("pickup"))


def test_closure_renames_statement_args_to_operator_roles() -> None:
    spec = DFOCISpec.from_dict(
        {
            "domain": "toy",
            "statements": [{"operator": "op(X)", "influences": ["b(X)"], "target": "a(X)"}],
            "operators": {"op(X)": {"reward_parents": ["a(X)"], "termination_parents": []}},
        }
    )
    assert spec.relevant_literals("op", roles=("P",)) == frozenset(
        {Literal.parse("a(P)"), Literal.parse("b(P)")}
    )


def test_unknown_operator_raises() -> None:
    with pytest.raises(KeyError):
        load_dfoci("taxi").relevant_literals("fly")


# ------------------------------------------------------------------ projection + lifting
def taxi_state(n: int, seed: int) -> tuple[TaxiDomain, State]:
    dom = TaxiDomain(TaxiConfig(num_passengers=n))
    return dom, dom.reset(np.random.default_rng(seed))


def test_projection_keeps_only_relevant_atoms_and_lifts_arguments() -> None:
    dom, s = taxi_state(2, 0)
    abstraction = DFOCIAbstraction(load_dfoci("taxi"), signature=dom.predicates)
    key = abstraction.project(s, PICKUP.instantiate(("p1",)))
    assert abstraction.abstract(s, PICKUP.instantiate(("p1",))) == key
    non_wall = {str(a) for a in key if a.pred != "wall"}
    (p1_at,) = [a for a in s.atoms_with("at") if a.args[0] == "p1"]
    (taxi_at,) = [a for a in s.atoms_with("at") if a.args[0] == "taxi"]
    assert non_wall == {f"at(?P,{p1_at.args[1]})", str(taxi_at)}
    assert {a for a in key if a.pred == "wall"} == s.atoms_with("wall")


def test_drop_key_keeps_destination_and_in_taxi_but_not_other_passengers() -> None:
    dom, s = taxi_state(3, 1)
    (p2_at,) = [a for a in s.atoms_with("at") if a.args[0] == "p2"]
    s = s.with_atoms(add=[Atom.parse("in(p2,taxi)")], remove=[p2_at])
    key = DFOCIAbstraction(load_dfoci("taxi"), signature=dom.predicates).project(
        s, DROP.instantiate(("p2",))
    )
    non_wall = {str(a) for a in key if a.pred != "wall"}
    (dest,) = [a for a in s.atoms_with("dest") if a.args[0] == "p2"]
    (taxi_at,) = [a for a in s.atoms_with("at") if a.args[0] == "taxi"]
    assert non_wall == {"in(?P,taxi)", f"dest(?P,{dest.args[1]})", str(taxi_at)}


def test_keys_identical_across_instances_differing_in_irrelevant_passengers() -> None:
    dom1, s1 = taxi_state(1, 3)
    dom3, s3 = taxi_state(3, 4)
    # copy p1's situation and the taxi position from the 1-passenger instance into the other
    moved = {a for a in s3.atoms if a.args[0] not in {"p1", "taxi"}}
    moved |= {a for a in s1.atoms if a.args[0] in {"p1", "taxi"}}
    s3 = State(frozenset(moved), s3.objects)
    abstraction = DFOCIAbstraction(load_dfoci("taxi"), signature=dom1.predicates)
    op = PICKUP.instantiate(("p1",))
    assert abstraction.abstract(s1, op) == abstraction.abstract(s3, op)


def test_keys_identical_across_passenger_names_after_lifting() -> None:
    dom, s = taxi_state(2, 5)
    at = {a.args[0]: a for a in s.atoms_with("at")}
    dest = {a.args[0]: a for a in s.atoms_with("dest")}
    # make p2's situation equal p1's situation
    s_eq = s.with_atoms(
        add=[Atom("at", ("p2", at["p1"].args[1])), Atom("dest", ("p2", dest["p1"].args[1]))],
        remove=[at["p2"], dest["p2"]],
    )
    abstraction = DFOCIAbstraction(load_dfoci("taxi"), signature=dom.predicates)
    k1 = abstraction.project(s_eq, PICKUP.instantiate(("p1",)))
    k2 = abstraction.project(s_eq, PICKUP.instantiate(("p2",)))
    assert k1 == k2
    assert "p1" not in str(sorted(str(a) for a in k1))


def test_projection_respects_variable_types() -> None:
    # at(P,L) must not pull in the taxi's location twice or other object types
    spec = DFOCISpec.from_dict(
        {
            "domain": "toy",
            "types": {"X": "passenger"},
            "statements": [],
            "operators": {"op(X)": {"reward_parents": ["at(Y,L)"], "termination_parents": []}},
            "reward_parents": [],
        }
    )
    objects = frozenset({Obj("p1", "passenger"), Obj("taxi", "taxi"), Obj("l", "location")})
    s = State(frozenset({Atom.parse("at(p1,l)"), Atom.parse("at(taxi,l)")}), objects)
    sig = {"at": ("object", "location")}
    spec_typed = DFOCISpec.from_dict(
        {**spec.to_dict(), "types": {"X": "passenger", "Y": "passenger"}}
    )
    op = __import__("reprel.planning.operators", fromlist=["OperatorSpec"]).OperatorSpec(
        name="op", params=(("X", "passenger"),), preconditions=(), add=(), delete=(), termination=()
    )
    assert DFOCIAbstraction(spec, signature=sig).abstract(s, op.instantiate(("p1",))) == frozenset(
        {Atom.parse("at(?X,l)"), Atom.parse("at(taxi,l)")}
    )
    assert DFOCIAbstraction(spec_typed, signature=sig).abstract(s, op.instantiate(("p1",))) == (
        frozenset({Atom.parse("at(?X,l)")})
    )


def test_make_abstraction_from_registry_with_spec_name() -> None:
    dom, s = taxi_state(1, 0)
    abstraction = make_abstraction("dfoci", spec="taxi", signature=dom.predicates)
    assert isinstance(abstraction, DFOCIAbstraction)
    key = abstraction.project(s, PICKUP.instantiate(("p1",)))
    assert any(a.pred == "wall" for a in key)
