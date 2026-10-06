"""Immutable relational state: a frozenset of ground atoms over typed objects."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

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

    # --- queries -----------------------------------------------------------------------
    def holds(self, atom: Atom) -> bool:
        return atom in self.atoms

    def atoms_with(self, pred: str) -> frozenset[Atom]:
        return frozenset(a for a in self.atoms if a.pred == pred)

    def type_of(self, name: str) -> str:
        for obj in self.objects:
            if obj.name == name:
                return obj.type
        raise KeyError(name)

    def types(self) -> dict[str, str]:
        """Object name -> type name."""
        return {o.name: o.type for o in self.objects}

    def objects_of_type(self, type_name: str) -> tuple[str, ...]:
        return tuple(sorted(o.name for o in self.objects if o.type == type_name))

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
