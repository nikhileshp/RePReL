"""Conjunctive queries over a State: enumerate substitutions satisfying a literal list."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence

from .atoms import Literal, Substitution, is_variable, unify
from .state import State

Signature = Mapping[str, tuple[str, ...]]


def matches(
    literals: Sequence[Literal],
    state: State,
    theta: Substitution | None = None,
    *,
    signature: Signature | None = None,
) -> Iterator[dict[str, str]]:
    """Yield every substitution extending ``theta`` under which all ``literals`` hold.

    Positive literals are matched against the state's atoms (binding variables, typed by
    ``signature`` and the state's objects). Negative literals are evaluated once their
    variables are bound; a negative literal with unbound variables holds iff no atom
    unifies with it (negation as failure). Positive literals are processed first so
    negative ones see maximal bindings.
    """
    ordered = sorted(literals, key=lambda lit: not lit.positive)
    types = state.types()
    yield from _match(ordered, 0, state, dict(theta or {}), types, signature)


def _match(
    literals: Sequence[Literal],
    i: int,
    state: State,
    theta: dict[str, str],
    types: Mapping[str, str],
    signature: Signature | None,
) -> Iterator[dict[str, str]]:
    if i == len(literals):
        yield theta
        return
    lit = literals[i]
    pattern = lit.atom.substitute(theta)
    candidates = state.atoms_with(pattern.pred)
    if lit.positive:
        for atom in sorted(candidates, key=str):
            extended = unify(pattern, atom, theta, types=types, signature=signature)
            if extended is not None:
                yield from _match(literals, i + 1, state, extended, types, signature)
        return
    if any(is_variable(a) for a in pattern.args):
        blocked = any(
            unify(pattern, atom, theta, types=types, signature=signature) is not None
            for atom in candidates
        )
    else:
        blocked = pattern in candidates
    if not blocked:
        yield from _match(literals, i + 1, state, theta, types, signature)


def satisfied(
    literals: Sequence[Literal],
    state: State,
    theta: Substitution | None = None,
    *,
    signature: Signature | None = None,
) -> bool:
    """True if at least one substitution satisfies ``literals``."""
    return next(matches(literals, state, theta, signature=signature), None) is not None
