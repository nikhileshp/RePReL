from typing import ClassVar

import numpy as np
import pytest

from reprel.core.atoms import Atom, Literal, Obj
from reprel.core.domain import DOMAINS, Domain, Transition, make_domain, register_domain
from reprel.core.state import State


@register_domain("toy")
class ToyDomain(Domain):
    """Two-cell line: 'right' reaches the goal cell."""

    name: ClassVar[str] = "toy"
    types = {"cell": None}
    predicates = {"at": ("cell",)}
    actions = ("left", "right")
    max_steps = 10

    def __init__(self, bonus: float = 1.0) -> None:
        self.bonus = bonus

    def reset(self, rng: np.random.Generator) -> State:
        objects = frozenset({Obj("c0", "cell"), Obj("c1", "cell")})
        return State(frozenset({Atom.parse("at(c0)")}), objects)

    def step(self, state: State, action: str, rng: np.random.Generator) -> Transition:
        if action == "right":
            nxt = state.with_atoms(add=[Atom.parse("at(c1)")], remove=[Atom.parse("at(c0)")])
            return Transition(nxt, self.bonus, True, {})
        return Transition(state, -1.0, False, {})

    def goal(self, state: State) -> frozenset[Literal]:
        return frozenset({Literal.parse("at(c1)")})

    def is_success(self, state: State) -> bool:
        return state.holds(Atom.parse("at(c1)"))


def test_registry_and_factory_pass_config() -> None:
    assert DOMAINS["toy"] is ToyDomain
    dom = make_domain("toy", bonus=5.0)
    assert isinstance(dom, ToyDomain)
    assert dom.bonus == 5.0
    with pytest.raises(KeyError):
        make_domain("missing")


def test_step_returns_transition_and_does_not_mutate_input() -> None:
    dom = make_domain("toy")
    rng = np.random.default_rng(0)
    s0 = dom.reset(rng)
    t = dom.step(s0, "right", rng)
    assert t.done and t.reward == 1.0
    assert dom.is_success(t.next_state)
    assert not dom.is_success(s0)
    assert dom.goal(s0) == frozenset({Literal.parse("at(c1)")})


def test_unknown_action_is_rejected_by_validate() -> None:
    dom = make_domain("toy")
    with pytest.raises(ValueError):
        dom.validate_action("jump")
    assert dom.action_index("right") == 1
