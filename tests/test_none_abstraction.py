import numpy as np

from reprel.abstraction import ABSTRACTIONS, Abstraction, make_abstraction
from reprel.abstraction.none import NoAbstraction
from reprel.domains.taxi import TaxiConfig, TaxiDomain
from reprel.domains.taxi_planning import PICKUP


def test_registry() -> None:
    assert ABSTRACTIONS["none"] is NoAbstraction
    assert isinstance(make_abstraction("none"), Abstraction)


def test_key_is_the_full_ground_state() -> None:
    dom = TaxiDomain(TaxiConfig(num_passengers=2))
    s = dom.reset(np.random.default_rng(0))
    key = NoAbstraction().abstract(s, PICKUP.instantiate(("p1",)))
    assert key == s.atoms
    assert hash(key) == hash(s.atoms)


def test_binding_is_ignored_unless_requested() -> None:
    dom = TaxiDomain(TaxiConfig(num_passengers=2))
    s = dom.reset(np.random.default_rng(0))
    plain = NoAbstraction()
    assert plain.abstract(s, PICKUP.instantiate(("p1",))) == plain.abstract(
        s, PICKUP.instantiate(("p2",))
    )
    with_args = NoAbstraction(include_binding=True)
    assert with_args.abstract(s, PICKUP.instantiate(("p1",))) != with_args.abstract(
        s, PICKUP.instantiate(("p2",))
    )
    assert with_args.abstract(s, PICKUP.instantiate(("p1",))) == (s.atoms, ("p1",))


def test_exclude_drops_predicates_from_the_key() -> None:
    dom = TaxiDomain(TaxiConfig(num_passengers=1))
    s = dom.reset(np.random.default_rng(0))
    key = NoAbstraction(exclude=("wall", "dest")).abstract(s, PICKUP.instantiate(("p1",)))
    assert isinstance(key, frozenset)
    assert not any(a.pred in ("wall", "dest") for a in key)
    assert any(a.pred == "at" for a in key)
    assert make_abstraction("none", exclude=["wall"]).abstract(
        s, PICKUP.instantiate(("p1",))
    ) == frozenset(a for a in s.atoms if a.pred != "wall")
