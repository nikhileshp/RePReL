"""Immutable relational state: a frozenset of ground atoms over typed objects."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import cached_property

from .atoms import Atom, Obj


@dataclass(frozen=True)
class State:
    """A relational state.

    Attributes:
        atoms: Ground atoms that hold in this state.
        objects: All objects of the instance (including those with no atoms).
    """

    atoms: frozenset[Atom]
    objects: frozenset[Obj]

    # --- cached indexes (instance-local; excluded from eq/hash) ---------------------------
    @cached_property
    def _by_pred(self) -> dict[str, frozenset[Atom]]:
        index: dict[str, set[Atom]] = {}
        for atom in self.atoms:
            index.setdefault(atom.pred, set()).add(atom)
        return {pred: frozenset(atoms) for pred, atoms in index.items()}

    @cached_property
    def _type_index(self) -> dict[str, str]:
        return {o.name: o.type for o in self.objects}

    @cached_property
    def _by_type(self) -> dict[str, tuple[str, ...]]:
        index: dict[str, list[str]] = {}
        for o in self.objects:
            index.setdefault(o.type, []).append(o.name)
        return {t: tuple(sorted(names)) for t, names in index.items()}

    # --- queries -----------------------------------------------------------------------
    def holds(self, atom: Atom) -> bool:
        return atom in self.atoms

    def atoms_with(self, pred: str) -> frozenset[Atom]:
        return self._by_pred.get(pred, frozenset())

    def type_of(self, name: str) -> str:
        return self._type_index[name]

    def types(self) -> Mapping[str, str]:
        """Object name -> type name (read-only view)."""
        return self._type_index

    def objects_of_type(self, type_name: str) -> tuple[str, ...]:
        return self._by_type.get(type_name, ())

    # --- constructors ------------------------------------------------------------------
    def with_atoms(self, add: Iterable[Atom] = (), remove: Iterable[Atom] = ()) -> State:
        return State((self.atoms - frozenset(remove)) | frozenset(add), self.objects)

    def substitute(self, theta: Mapping[str, str]) -> State:
        """Rename objects (old name -> new name) in atoms and in the object set."""
        atoms = frozenset(a.substitute(theta) for a in self.atoms)
        objects = frozenset(Obj(theta.get(o.name, o.name), o.type) for o in self.objects)
        return State(atoms, objects)

    def lift(self, binding: Mapping[str, str]) -> State:
        """Rename bound objects to their roles: ``{"P": "p1"}`` turns ``p1`` into ``?P``."""
        return self.substitute({obj: f"?{role}" for role, obj in binding.items()})

    def diff(self, other: State) -> tuple[frozenset[Atom], frozenset[Atom]]:
        """Return ``(added, removed)`` going from ``self`` to ``other``."""
        return other.atoms - self.atoms, self.atoms - other.atoms

    # --- serialisation -----------------------------------------------------------------
    def to_strings(self) -> list[str]:
        return sorted(str(a) for a in self.atoms)

    @classmethod
    def from_strings(cls, atoms: Iterable[str], objects: Iterable[Obj]) -> State:
        return cls(frozenset(Atom.parse(a) for a in atoms), frozenset(objects))
