"""Ground and lifted atoms, literals, typed objects, and first-order unification.

Conventions (matching the RePReL paper): identifiers starting with an uppercase letter are
variables (``P``, ``L1``); everything else is an object constant (``p1``, ``l_0_0``, ``?P``).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

Substitution = Mapping[str, str]

ANY_TYPE = "object"
_ATOM_RE = re.compile(r"^\s*([A-Za-z_][\w\-]*)\s*\((.*)\)\s*$")


def is_variable(name: str) -> bool:
    """Return True if ``name`` denotes a logical variable (first character uppercase)."""
    return bool(name) and name[0].isupper()


@dataclass(frozen=True)
class Obj:
    """A typed domain object."""

    name: str
    type: str


@dataclass(frozen=True)
class Atom:
    """A predicate applied to argument names (objects or variables)."""

    pred: str
    args: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.args, tuple):
            object.__setattr__(self, "args", tuple(self.args))

    @classmethod
    def parse(cls, text: str) -> Atom:
        """Parse ``"pred(a,b)"`` into an Atom. Whitespace around arguments is ignored."""
        match = _ATOM_RE.match(text)
        if match is None:
            raise ValueError(f"malformed atom: {text!r}")
        pred, body = match.group(1), match.group(2).strip()
        args = tuple(a.strip() for a in body.split(",")) if body else ()
        if any(not a for a in args):
            raise ValueError(f"malformed atom: {text!r}")
        return cls(pred, args)

    def __str__(self) -> str:
        return f"{self.pred}({','.join(self.args)})"

    @property
    def is_ground(self) -> bool:
        return not any(is_variable(a) for a in self.args)

    def variables(self) -> tuple[str, ...]:
        """Distinct variables in argument order."""
        seen: list[str] = []
        for a in self.args:
            if is_variable(a) and a not in seen:
                seen.append(a)
        return tuple(seen)

    def substitute(self, theta: Substitution) -> Atom:
        """Replace arguments found in ``theta``; unmapped arguments are left as they are."""
        return Atom(self.pred, tuple(theta.get(a, a) for a in self.args))


@dataclass(frozen=True)
class Literal:
    """A positive or negated atom."""

    atom: Atom
    positive: bool = True

    @classmethod
    def parse(cls, text: str) -> Literal:
        stripped = text.strip()
        if stripped.startswith("not "):
            return cls(Atom.parse(stripped[4:]), False)
        return cls(Atom.parse(stripped), True)

    def __str__(self) -> str:
        return str(self.atom) if self.positive else f"not {self.atom}"

    def substitute(self, theta: Substitution) -> Literal:
        return Literal(self.atom.substitute(theta), self.positive)


def type_compatible(actual: str, required: str) -> bool:
    """True if an object of type ``actual`` may fill a slot of type ``required``.

    Types are flat for now; subtype hierarchies from ``Domain.types`` are not consulted.
    """
    return required == ANY_TYPE or actual == required


def unify(
    pattern: Atom,
    ground: Atom,
    theta: Substitution | None = None,
    *,
    types: Mapping[str, str] | None = None,
    signature: Mapping[str, tuple[str, ...]] | None = None,
) -> dict[str, str] | None:
    """Unify a (possibly lifted) ``pattern`` with a ground atom.

    Args:
        pattern: Atom that may contain variables.
        ground: Atom with object arguments only.
        theta: Existing variable binding to extend (not mutated).
        types: Object name -> type name, used with ``signature`` for type checks.
        signature: Predicate -> argument types; a variable may only bind to an object whose
            type is compatible with the slot's declared type.

    Returns:
        The extended binding, or None if unification fails.
    """
    if pattern.pred != ground.pred or len(pattern.args) != len(ground.args):
        return None
    binding: dict[str, str] = dict(theta or {})
    slot_types = signature.get(pattern.pred) if signature is not None else None
    for i, (p, g) in enumerate(zip(pattern.args, ground.args, strict=True)):
        if is_variable(p):
            bound = binding.get(p)
            if bound is None:
                if slot_types is not None and types is not None:
                    actual = types.get(g, ANY_TYPE)
                    if not type_compatible(actual, slot_types[i]):
                        return None
                binding[p] = g
            elif bound != g:
                return None
        elif p != g:
            return None
    return binding
