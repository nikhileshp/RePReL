"""Relational Box World (Zambaldi et al. 2019 variant used by RePReL).

A chain of boxes ``[content | lock]`` leads to the gem: the free key opens lock 1, whose box
holds key 2, which opens lock 2, ... and the last box holds the gem. Colours are sampled per
episode. The agent walks on a grid; moving onto a free key collects it (replacing any owned
key), moving onto a lock whose colour matches the owned key opens it (consuming the key and
freeing the box content), moving onto a locked key or a mismatching lock is a bump.

The state carries exact positions (``at``), box structure (``inside``, ``locked``, ``color``,
``own``, ``open``) and, as in the original environment's relational observation, the agent's
sensors: ``neighbor(dir, what)`` for the 8 surrounding cells (``wall``, ``cell`` or an object),
``direction(obj, dir)`` the coarse direction of every remaining object, and ``agent-at(what)``.
Rewards (original code): step -0.1, bump -0.2 extra, key +1, gem +1 and done.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, ClassVar

import numpy as np

from reprel.core.atoms import Atom, Literal, Obj
from reprel.core.domain import Action, Domain, Goal, Transition, register_domain
from reprel.core.state import State

AGENT, GEM = "agent", "gem"
COLORS = (
    "red",
    "green",
    "blue",
    "yellow",
    "purple",
    "orange",
    "cyan",
    "magenta",
    "lime",
    "pink",
    "teal",
    "lavender",
    "brown",
    "beige",
    "maroon",
    "mint",
    "olive",
    "navy",
)
MOVES: dict[str, tuple[int, int]] = {
    "north": (-1, 0),
    "south": (1, 0),
    "west": (0, -1),
    "east": (0, 1),
}
SENSOR_DIRS: dict[str, tuple[int, int]] = {
    "n": (-1, 0),
    "ne": (-1, 1),
    "e": (0, 1),
    "se": (1, 1),
    "s": (1, 0),
    "sw": (1, -1),
    "w": (0, -1),
    "nw": (-1, -1),
}
DYNAMIC_PREDS = ("neighbor", "direction", "agent-at")


@dataclass(frozen=True)
class BoxWorldConfig:
    size: int = 10
    goal_length: int = 2
    own_first_key: bool = False
    max_steps: int | None = None  # default 100 * goal_length, as in the original tasks
    step_reward: float = -0.1
    no_move_reward: float = -0.2
    key_reward: float = 1.0
    gem_reward: float = 1.0

    def __post_init__(self) -> None:
        if self.goal_length < 1:
            raise ValueError("goal_length must be >= 1")
        if self.max_steps is None:
            object.__setattr__(self, "max_steps", 100 * self.goal_length)


def loc_name(cell: tuple[int, int]) -> str:
    return f"l_{cell[0]}_{cell[1]}"


def cell_of(name: str) -> tuple[int, int]:
    _, r, c = name.split("_")
    return int(r), int(c)


@register_domain("boxworld")
class BoxWorldDomain(Domain):
    name: ClassVar[str] = "boxworld"
    types: ClassVar[dict[str, str | None]] = {
        "agent": None,
        "key": None,
        "lock": None,
        "location": None,
        "dir": None,
        "sensed": None,
    }
    predicates: ClassVar[dict[str, tuple[str, ...]]] = {
        "at": ("object", "location"),
        "color": ("object", "object"),
        "inside": ("lock", "key"),
        "locked": ("key",),
        "own": ("key",),
        "open": ("lock",),
        "neighbor": ("dir", "object"),
        "direction": ("object", "dir"),
        "agent-at": ("object",),
    }
    actions: ClassVar[tuple[Action, ...]] = ("north", "south", "west", "east")

    def __init__(self, config: BoxWorldConfig | None = None, **overrides: Any) -> None:
        self.cfg = replace(config or BoxWorldConfig(), **overrides)
        assert self.cfg.max_steps is not None
        self.max_steps = self.cfg.max_steps
        self.size = self.cfg.size

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def cell_of_object(state: State, obj: str) -> str:
        for atom in state.atoms_with("at"):
            if atom.args[0] == obj:
                return atom.args[1]
        raise KeyError(obj)

    @staticmethod
    def keys_in(state: State) -> tuple[str, ...]:
        return tuple(sorted(o.name for o in state.objects if o.type == "key" and o.name != GEM))

    @classmethod
    def with_agent_at(cls, state: State, cell: str) -> State:
        """Move the agent to ``cell`` and recompute the sensor atoms (test helper)."""
        base = state.with_atoms(
            add=[Atom("at", (AGENT, cell))],
            remove=[
                a
                for a in state.atoms
                if a.pred in DYNAMIC_PREDS or (a.pred == "at" and a.args[0] == AGENT)
            ],
        )
        size = max(cell_of(o.name)[0] for o in state.objects if o.type == "location") + 1
        return base.with_atoms(add=cls._sensors(base, cell, size))

    # ------------------------------------------------------------------ Domain API
    def reset(self, rng: np.random.Generator) -> State:
        g, n = self.cfg.goal_length, self.size
        colours = [COLORS[i] for i in rng.choice(len(COLORS), size=g, replace=False)]
        objects = {Obj(AGENT, "agent"), Obj(GEM, "key")}
        objects |= {Obj(d, "dir") for d in SENSOR_DIRS}
        objects |= {Obj("wall", "sensed"), Obj("cell", "sensed")}
        objects |= {Obj(loc_name((r, c)), "location") for r in range(n) for c in range(n)}
        objects |= {Obj(c, "sensed") for c in COLORS}
        atoms: set[Atom] = set()
        occupied: set[tuple[int, int]] = set()

        def place(width: int) -> tuple[int, int]:
            for _ in range(10_000):
                r, c = int(rng.integers(n)), int(rng.integers(n - width + 1))
                cells = [(r, c + i) for i in range(width)]
                margin = {
                    (rr + dr, cc + dc) for rr, cc in cells for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                }
                if not (margin & occupied):
                    occupied.update(cells)
                    return r, c
            raise RuntimeError("could not place all boxes; use a larger grid")

        keys = [f"key_{col}" for col in colours]
        locks = [f"lock_{col}" for col in colours]
        for k, lk, col in zip(keys, locks, colours, strict=True):
            objects |= {Obj(k, "key"), Obj(lk, "lock")}
            atoms |= {Atom("color", (k, col)), Atom("color", (lk, col))}
        # box i: content (key i+1, or the gem for the last box) on the left of lock i
        contents = keys[1:] + [GEM]
        for lk, content in zip(locks, contents, strict=True):
            r, c = place(2)
            atoms |= {
                Atom("at", (content, loc_name((r, c)))),
                Atom("at", (lk, loc_name((r, c + 1)))),
            }
            atoms |= {Atom("inside", (lk, content)), Atom("locked", (content,))}
        if self.cfg.own_first_key:
            atoms.add(Atom("own", (keys[0],)))
        else:
            r, c = place(1)
            atoms.add(Atom("at", (keys[0], loc_name((r, c)))))
        r, c = place(1)
        agent_cell = loc_name((r, c))
        atoms.add(Atom("at", (AGENT, agent_cell)))
        base = State(frozenset(atoms), frozenset(objects))
        return base.with_atoms(add=self._sensors(base, agent_cell, n))

    def step(self, state: State, action: Action, rng: np.random.Generator) -> Transition:
        self.validate_action(action)
        here = self.cell_of_object(state, AGENT)
        r, c = cell_of(here)
        dr, dc = MOVES[action]
        nr, nc = r + dr, c + dc
        reward = self.cfg.step_reward
        add: set[Atom] = set()
        remove: set[Atom] = {a for a in state.atoms if a.pred in DYNAMIC_PREDS}
        moved, done = False, False
        if 0 <= nr < self.size and 0 <= nc < self.size:
            target = loc_name((nr, nc))
            occupant = next(
                (
                    a.args[0]
                    for a in state.atoms_with("at")
                    if a.args[1] == target and a.args[0] != AGENT
                ),
                None,
            )
            if occupant is None:
                moved = True
            elif state.type_of(occupant) == "key":
                if not state.holds(Atom("locked", (occupant,))):
                    moved = True
                    remove |= {Atom("at", (occupant, target))} | set(state.atoms_with("own"))
                    add.add(Atom("own", (occupant,)))
                    if occupant == GEM:
                        reward += self.cfg.gem_reward
                        done = True
                    else:
                        reward += self.cfg.key_reward
            else:  # a lock
                owned = next((a.args[0] for a in state.atoms_with("own")), None)
                colour = {a.args[0]: a.args[1] for a in state.atoms_with("color")}
                if owned is not None and colour.get(owned) == colour[occupant]:
                    moved = True
                    content = next(
                        a.args[1] for a in state.atoms_with("inside") if a.args[0] == occupant
                    )
                    remove |= {
                        Atom("at", (occupant, target)),
                        Atom("own", (owned,)),
                        Atom("locked", (content,)),
                    }
                    add.add(Atom("open", (occupant,)))
                    reward += self.cfg.key_reward
        if moved:
            remove.add(Atom("at", (AGENT, here)))
            add.add(Atom("at", (AGENT, loc_name((nr, nc)))))
            new_cell = loc_name((nr, nc))
        else:
            reward += self.cfg.no_move_reward
            new_cell = here
        base = state.with_atoms(add=add, remove=remove)
        next_state = base.with_atoms(add=self._sensors(base, new_cell, self.size))
        return Transition(next_state, reward, done, {"success": done})

    def goal(self, state: State) -> Goal:
        return frozenset({Literal.parse("own(gem)")})

    def is_success(self, state: State) -> bool:
        return state.holds(Atom("own", (GEM,)))

    # ------------------------------------------------------------------ sensors
    @staticmethod
    def _sensors(state: State, agent_cell: str, size: int) -> set[Atom]:
        r, c = cell_of(agent_cell)
        where = {a.args[1]: a.args[0] for a in state.atoms_with("at") if a.args[0] != AGENT}
        atoms: set[Atom] = set()
        for d, (dr, dc) in SENSOR_DIRS.items():
            rr, cc = r + dr, c + dc
            if not (0 <= rr < size and 0 <= cc < size):
                atoms.add(Atom("neighbor", (d, "wall")))
            else:
                atoms.add(Atom("neighbor", (d, where.get(loc_name((rr, cc)), "cell"))))
        atoms.add(Atom("agent-at", (where.get(agent_cell, "cell"),)))
        for cell, obj in where.items():
            orow, ocol = cell_of(cell)
            if (orow, ocol) == (r, c):
                continue
            vert = "n" if orow < r else "s" if orow > r else ""
            horiz = "w" if ocol < c else "e" if ocol > c else ""
            atoms.add(Atom("direction", (obj, vert + horiz)))
        return atoms
