"""Relational multi-passenger Taxi domain.

Objects: ``taxi``, passengers ``p1..pN``, locations ``l_<row>_<col>``, directions.
Predicates: ``at(thing, loc)``, ``in(passenger, taxi)``, ``dest(passenger, loc)``,
``delivered(passenger)``, ``wall(loc, dir)`` (static).
Dynamics and rewards follow the RePReL-domains Taxi environment.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, ClassVar

import numpy as np

from reprel.core.atoms import Atom, Literal, Obj
from reprel.core.domain import Action, Domain, Transition, register_domain
from reprel.core.state import State

from .taxi_layouts import DIRECTIONS, LAYOUTS, Cell, Grid

TAXI = "taxi"


@dataclass(frozen=True)
class TaxiConfig:
    num_passengers: int = 1
    layout: str = "eight"
    max_steps: int = 1000
    step_reward: float = -0.1
    pickup_reward: float = 10.0
    drop_reward: float = 20.0
    illegal_reward: float = -1.0

    def __post_init__(self) -> None:
        if self.num_passengers < 1:
            raise ValueError("num_passengers must be >= 1")


def loc_name(cell: Cell) -> str:
    return f"l_{cell[0]}_{cell[1]}"


def passenger_name(index: int) -> str:
    return f"p{index}"


def passenger_order(name: str) -> int:
    return int(name[1:])


@register_domain("taxi")
class TaxiDomain(Domain):
    name: ClassVar[str] = "taxi"
    types: ClassVar[dict[str, str | None]] = {
        "taxi": None,
        "passenger": None,
        "location": None,
        "dir": None,
    }
    predicates: ClassVar[dict[str, tuple[str, ...]]] = {
        "at": ("object", "location"),
        "in": ("passenger", "taxi"),
        "dest": ("passenger", "location"),
        "delivered": ("passenger",),
        "wall": ("location", "dir"),
    }
    actions: ClassVar[tuple[Action, ...]] = ("north", "south", "west", "east", "pickup", "dropoff")

    def __init__(self, config: TaxiConfig | None = None, **overrides: Any) -> None:
        self.cfg = replace(config or TaxiConfig(), **overrides)
        if self.cfg.layout not in LAYOUTS:
            raise ValueError(f"unknown layout {self.cfg.layout!r}; known: {sorted(LAYOUTS)}")
        self.grid: Grid = LAYOUTS[self.cfg.layout]
        self.max_steps = self.cfg.max_steps
        self._objects, self._static_atoms = self._build_static()

    # ------------------------------------------------------------------ static parts
    def _build_static(self) -> tuple[frozenset[Obj], frozenset[Atom]]:
        objects = {Obj(TAXI, "taxi")}
        objects |= {Obj(d, "dir") for d in DIRECTIONS}
        objects |= {Obj(loc_name(c), "location") for c in self.grid.free_cells}
        objects |= {
            Obj(passenger_name(i), "passenger") for i in range(1, self.cfg.num_passengers + 1)
        }
        walls = {
            Atom("wall", (loc_name(cell), d))
            for cell in self.grid.free_cells
            for d in DIRECTIONS
            if self.grid.is_blocked(self.grid.neighbour(cell, d))
        }
        return frozenset(objects), frozenset(walls)

    # ------------------------------------------------------------------ Domain API
    def reset(self, rng: np.random.Generator) -> State:
        depots = [self.grid.depots[k] for k in sorted(self.grid.depots)]
        starts = sorted(self.grid.free_cells - set(depots))
        taxi_cell = starts[int(rng.integers(len(starts)))]
        atoms = set(self._static_atoms)
        atoms.add(Atom("at", (TAXI, loc_name(taxi_cell))))
        for i in range(1, self.cfg.num_passengers + 1):
            pick, dest = rng.choice(len(depots), size=2, replace=False)
            atoms.add(Atom("at", (passenger_name(i), loc_name(depots[pick]))))
            atoms.add(Atom("dest", (passenger_name(i), loc_name(depots[dest]))))
        return State(frozenset(atoms), self._objects)

    def step(self, state: State, action: Action, rng: np.random.Generator) -> Transition:
        self.validate_action(action)
        taxi_loc = self._taxi_location(state)
        reward = self.cfg.step_reward
        if action in DIRECTIONS:
            next_state = self._move(state, taxi_loc, action)
        elif action == "pickup":
            next_state, reward = self._pickup(state, taxi_loc, reward)
        else:
            next_state, reward = self._dropoff(state, taxi_loc, reward)
        done = self.is_success(next_state)
        return Transition(next_state, reward, done, {"success": done})

    def goal(self, state: State) -> frozenset[Literal]:
        passengers = state.objects_of_type("passenger")
        return frozenset(Literal(Atom("delivered", (p,))) for p in passengers)

    def is_success(self, state: State) -> bool:
        return all(state.holds(Atom("delivered", (p,))) for p in state.objects_of_type("passenger"))

    # ------------------------------------------------------------------ dynamics
    @staticmethod
    def _taxi_location(state: State) -> str:
        for atom in state.atoms_with("at"):
            if atom.args[0] == TAXI:
                return atom.args[1]
        raise ValueError("state has no taxi location")

    def _move(self, state: State, taxi_loc: str, direction: str) -> State:
        if state.holds(Atom("wall", (taxi_loc, direction))):
            return state
        r, c = (int(x) for x in taxi_loc.split("_")[1:])
        target = loc_name(self.grid.neighbour((r, c), direction))
        return state.with_atoms(
            add=[Atom("at", (TAXI, target))], remove=[Atom("at", (TAXI, taxi_loc))]
        )

    def _carried(self, state: State) -> str | None:
        for atom in state.atoms_with("in"):
            return atom.args[0]
        return None

    def _pickup(self, state: State, taxi_loc: str, reward: float) -> tuple[State, float]:
        if self._carried(state) is not None:
            return state, reward + self.cfg.illegal_reward
        waiting = sorted(
            (
                a.args[0]
                for a in state.atoms_with("at")
                if a.args[1] == taxi_loc
                and a.args[0] != TAXI
                and not state.holds(Atom("delivered", (a.args[0],)))
            ),
            key=passenger_order,
        )
        if not waiting:
            return state, reward + self.cfg.illegal_reward
        p = waiting[0]
        next_state = state.with_atoms(
            add=[Atom("in", (p, TAXI))], remove=[Atom("at", (p, taxi_loc))]
        )
        return next_state, reward + self.cfg.pickup_reward

    def _dropoff(self, state: State, taxi_loc: str, reward: float) -> tuple[State, float]:
        p = self._carried(state)
        if p is None or not state.holds(Atom("dest", (p, taxi_loc))):
            return state, reward + self.cfg.illegal_reward
        next_state = state.with_atoms(
            add=[Atom("delivered", (p,)), Atom("at", (p, taxi_loc))],
            remove=[Atom("in", (p, TAXI))],
        )
        return next_state, reward + self.cfg.drop_reward
