"""Office World as a relational domain.

Objects: ``agent``, items ``a b c d mail coffee office`` (type ``item``), locations ``l_x_y``,
directions. Predicates: ``at(thing, loc)`` (items are static), ``with(item)`` (the paper's
with-agent: visited a/b/c/d, holding mail/coffee, visited office), ``delivered(item)``,
``office(loc)``, ``plant(loc)`` and ``wall(loc, dir)`` (static). Entering an item's cell sets
``with(item)``; entering the office converts held mail/coffee to ``delivered``.
Rewards (RePReL-domains): -1 per step, -10 extra for an invalid move and for standing on a
plant, +100 when the task's goal literals hold.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, ClassVar

import numpy as np

from reprel.core.atoms import Atom, Literal, Obj
from reprel.core.domain import Action, Domain, Goal, Transition, register_domain
from reprel.core.state import State

from .office_layout import DIRECTIONS, HEIGHT, ITEMS, PLANTS, WIDTH, Cell, blocked_pairs, is_blocked

AGENT = "agent"
TASKS: dict[str, tuple[str, ...]] = {
    "deliver_mail": ("delivered(mail)",),
    "deliver_coffee": ("delivered(coffee)",),
    "deliver_both": ("delivered(mail)", "delivered(coffee)"),
    "visit_abcd": ("with(a)", "with(b)", "with(c)", "with(d)"),
}
DELIVERABLE = ("mail", "coffee")


@dataclass(frozen=True)
class OfficeConfig:
    task: str = "deliver_mail"
    max_steps: int = 300
    step_reward: float = -1.0
    invalid_reward: float = -10.0
    plant_reward: float = -10.0
    terminal_reward: float = 100.0

    def __post_init__(self) -> None:
        if self.task not in TASKS:
            raise ValueError(f"unknown task {self.task!r}; known: {sorted(TASKS)}")


def loc_name(cell: Cell) -> str:
    return f"l_{cell[0]}_{cell[1]}"


def cell_of(name: str) -> Cell:
    _, x, y = name.split("_")
    return int(x), int(y)


@register_domain("office")
class OfficeDomain(Domain):
    name: ClassVar[str] = "office"
    types: ClassVar[dict[str, str | None]] = {
        "agent": None,
        "item": None,
        "location": None,
        "dir": None,
    }
    predicates: ClassVar[dict[str, tuple[str, ...]]] = {
        "at": ("object", "location"),
        "with": ("item",),
        "delivered": ("item",),
        "office": ("location",),
        "plant": ("location",),
        "wall": ("location", "dir"),
    }
    actions: ClassVar[tuple[Action, ...]] = ("up", "down", "left", "right")

    def __init__(self, config: OfficeConfig | None = None, **overrides: Any) -> None:
        self.cfg = replace(config or OfficeConfig(), **overrides)
        self.max_steps = self.cfg.max_steps
        self._pairs = blocked_pairs()
        self._goal: Goal = frozenset(Literal.parse(t) for t in TASKS[self.cfg.task])
        self._objects, self._static = self._build_static()
        self._items_at: dict[str, tuple[str, ...]] = {}
        for item, cells in ITEMS.items():
            for cell in cells:
                self._items_at[loc_name(cell)] = (*self._items_at.get(loc_name(cell), ()), item)
        self._plants = {loc_name(c) for c in PLANTS}

    def _build_static(self) -> tuple[frozenset[Obj], frozenset[Atom]]:
        cells = [(x, y) for x in range(WIDTH) for y in range(HEIGHT)]
        objects = {Obj(AGENT, "agent")}
        objects |= {Obj(d, "dir") for d in DIRECTIONS}
        objects |= {Obj(loc_name(c), "location") for c in cells}
        objects |= {Obj(item, "item") for item in ITEMS}
        atoms: set[Atom] = set()
        for item, item_cells in ITEMS.items():
            for cell in item_cells:
                atoms.add(Atom("at", (item, loc_name(cell))))
        atoms.add(Atom("office", (loc_name(ITEMS["office"][0]),)))
        atoms |= {Atom("plant", (loc_name(c),)) for c in PLANTS}
        atoms |= {
            Atom("wall", (loc_name(c), d))
            for c in cells
            for d in DIRECTIONS
            if is_blocked(c, d, self._pairs)
        }
        return frozenset(objects), frozenset(atoms)

    # ------------------------------------------------------------------ Domain API
    def reset(self, rng: np.random.Generator) -> State:
        occupied = {c for cells in ITEMS.values() for c in cells} | set(PLANTS)
        free = sorted(
            c for c in ((x, y) for x in range(WIDTH) for y in range(HEIGHT)) if c not in occupied
        )
        start = free[int(rng.integers(len(free)))]
        return State(self._static | {Atom("at", (AGENT, loc_name(start)))}, self._objects)

    def step(self, state: State, action: Action, rng: np.random.Generator) -> Transition:
        self.validate_action(action)
        here = self._agent_location(state)
        reward = self.cfg.step_reward
        if state.holds(Atom("wall", (here, action))):
            return Transition(state, reward + self.cfg.invalid_reward, False, {"success": False})
        dx, dy = DIRECTIONS[action]
        x, y = cell_of(here)
        there = loc_name((x + dx, y + dy))
        add: set[Atom] = {Atom("at", (AGENT, there))}
        remove: set[Atom] = {Atom("at", (AGENT, here))}
        for item in self._items_at.get(there, ()):
            add.add(Atom("with", (item,)))
            if item == "office":
                for held in DELIVERABLE:
                    if state.holds(Atom("with", (held,))):
                        remove.add(Atom("with", (held,)))
                        add.add(Atom("delivered", (held,)))
        if there in self._plants:
            reward += self.cfg.plant_reward
        next_state = state.with_atoms(add=add, remove=remove)
        done = self.is_success(next_state)
        if done:
            reward += self.cfg.terminal_reward
        return Transition(next_state, reward, done, {"success": done})

    def goal(self, state: State) -> Goal:
        return self._goal

    def is_success(self, state: State) -> bool:
        return all(state.holds(lit.atom) == lit.positive for lit in self._goal)

    @staticmethod
    def _agent_location(state: State) -> str:
        for atom in state.atoms_with("at"):
            if atom.args[0] == AGENT:
                return atom.args[1]
        raise ValueError("state has no agent location")
