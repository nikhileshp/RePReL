"""Dynamic First-Order Conditional Influence (D-FOCI) statements and the abstraction they induce.

A statement ``operator: if C then X1 -+1-> X2`` says that, while ``operator`` runs, literal
``X2`` at the next step is directly influenced only by ``X1`` (and the context ``C``).
The relevant literals for an operator are the ancestor closure of its reward and
termination parents over the statements. Projecting a state onto the relevant literals
and renaming the operator's bound arguments to roles gives a lifted, hashable key.

YAML format (variables start uppercase; ``types`` optionally types them). Atoms contain
commas, so lists must use block style (or quoted strings)::

    domain: taxi
    source: hand            # hand | learned
    types: {P: passenger}
    reward_parents:         # parents of the step reward R
      - at(taxi,L1)
      - move(Dir)
    statements:
      - operator: pickup(P)   # optional; omitted = holds for every operator
        if: []                # context literals
        influences:
          - at(taxi,L1)
          - at(P,L)
          - in(P,taxi)
        target: in(P,taxi)
    operators:
      pickup(P):
        reward_parents: [in(P,taxi)]
        termination_parents: [in(P,taxi)]
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from reprel.core.atoms import Atom, Literal, is_variable, unify
from reprel.core.query import Signature
from reprel.core.state import State
from reprel.planning.operators import OperatorInstance

from .abstraction import Abstraction, register_abstraction

SOURCES = ("hand", "learned")


def _parse_head(text: str) -> tuple[str, tuple[str, ...]]:
    atom = Atom.parse(text) if "(" in text else Atom(text.strip(), ())
    return atom.pred, atom.args


def _lits(items: Iterable[str] | None) -> tuple[Literal, ...]:
    return tuple(Literal.parse(t) for t in (items or ()))


@dataclass(frozen=True)
class Statement:
    influences: tuple[Literal, ...]
    target: Literal
    operator: str | None = None
    operator_args: tuple[str, ...] = ()
    context: tuple[Literal, ...] = ()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Statement:
        if "target" not in data or "influences" not in data:
            raise ValueError(f"statement needs 'influences' and 'target': {data!r}")
        operator: str | None = None
        args: tuple[str, ...] = ()
        if data.get("operator"):
            operator, args = _parse_head(data["operator"])
        return cls(
            influences=_lits(data["influences"]),
            target=Literal.parse(data["target"]),
            operator=operator,
            operator_args=args,
            context=_lits(data.get("if")),
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.operator:
            out["operator"] = f"{self.operator}({','.join(self.operator_args)})"
        if self.context:
            out["if"] = [str(lit) for lit in self.context]
        out["influences"] = [str(lit) for lit in self.influences]
        out["target"] = str(self.target)
        return out


@dataclass(frozen=True)
class OperatorParents:
    args: tuple[str, ...]
    reward_parents: tuple[Literal, ...] = ()
    termination_parents: tuple[Literal, ...] = ()


@dataclass(frozen=True)
class DFOCISpec:
    domain: str
    statements: tuple[Statement, ...]
    operators: Mapping[str, OperatorParents]
    reward_parents: tuple[Literal, ...] = ()
    types: Mapping[str, str] = field(default_factory=dict)
    source: str = "hand"

    # ------------------------------------------------------------------ I/O
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DFOCISpec:
        source = data.get("source", "hand")
        if source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}, got {source!r}")
        operators: dict[str, OperatorParents] = {}
        for head, parents in (data.get("operators") or {}).items():
            name, args = _parse_head(head)
            parents = parents or {}
            operators[name] = OperatorParents(
                args=args,
                reward_parents=_lits(parents.get("reward_parents")),
                termination_parents=_lits(parents.get("termination_parents")),
            )
        return cls(
            domain=str(data.get("domain", "")),
            statements=tuple(Statement.from_dict(s) for s in data.get("statements") or ()),
            operators=operators,
            reward_parents=_lits(data.get("reward_parents")),
            types=dict(data.get("types") or {}),
            source=source,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> DFOCISpec:
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(yaml.safe_load(fh) or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "source": self.source,
            "types": dict(self.types),
            "reward_parents": [str(lit) for lit in self.reward_parents],
            "statements": [s.to_dict() for s in self.statements],
            "operators": {
                f"{name}({','.join(op.args)})": {
                    "reward_parents": [str(lit) for lit in op.reward_parents],
                    "termination_parents": [str(lit) for lit in op.termination_parents],
                }
                for name, op in self.operators.items()
            },
        }

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)

    # ------------------------------------------------------------------ closure
    def relevant_literals(
        self, operator: str, roles: Sequence[str] | None = None, max_depth: int | None = None
    ) -> frozenset[Literal]:
        """Ancestor closure of the operator's reward/termination parents (plus global R parents).

        Statement variables that name operator arguments are renamed to ``roles``
        (positionally); all other variables stay statement-local existentials.
        """
        if operator not in self.operators:
            raise KeyError(f"no D-FOCI parents for operator {operator!r}")
        parents = self.operators[operator]
        rename = self._renaming(parents.args, roles)
        relevant: set[Literal] = {lit.substitute(rename) for lit in self.reward_parents}
        relevant |= {lit.substitute(rename) for lit in parents.reward_parents}
        relevant |= {lit.substitute(rename) for lit in parents.termination_parents}
        applicable = [
            (s, self._renaming(s.operator_args, roles if roles else parents.args))
            for s in self.statements
            if s.operator is None or s.operator == operator
        ]
        depth = 0
        while max_depth is None or depth < max_depth:
            added: set[Literal] = set()
            for statement, theta in applicable:
                target = statement.target.substitute(theta)
                if any(_compatible(target.atom, lit.atom) for lit in relevant):
                    for lit in (*statement.influences, *statement.context):
                        renamed = lit.substitute(theta)
                        if renamed not in relevant:
                            added.add(renamed)
            if not added:
                break
            relevant |= added
            depth += 1
        return frozenset(relevant)

    @staticmethod
    def _renaming(args: Sequence[str], roles: Sequence[str] | None) -> dict[str, str]:
        if not roles or not args:
            return {}
        if len(args) != len(roles):
            raise ValueError(f"operator args {args} do not match roles {tuple(roles)}")
        return dict(zip(args, roles, strict=True))


def _compatible(a: Atom, b: Atom) -> bool:
    """Two lifted atoms may denote the same ground atom (variables match anything)."""
    if a.pred != b.pred or len(a.args) != len(b.args):
        return False
    return all(
        is_variable(x) or is_variable(y) or x == y for x, y in zip(a.args, b.args, strict=True)
    )


def load_dfoci(name: str) -> DFOCISpec:
    """Load a bundled spec from ``reprel/dfoci/<name>.yaml`` (or a path to a YAML file)."""
    path = Path(name)
    if path.suffix in {".yaml", ".yml"} and path.exists():
        return DFOCISpec.from_yaml(path)
    resource = resources.files("reprel.dfoci").joinpath(f"{name}.yaml")
    with resources.as_file(resource) as p:
        return DFOCISpec.from_yaml(p)


@register_abstraction("dfoci")
class DFOCIAbstraction(Abstraction):
    """Project the state onto an operator's relevant literals and lift its arguments.

    Args:
        spec: A :class:`DFOCISpec` or the name/path of a spec for :func:`load_dfoci`.
        signature: Predicate -> argument types from the domain, for typed unification.
        max_depth: Optional bound on the closure depth (paper: 2; default unbounded).
    """

    def __init__(
        self,
        spec: DFOCISpec | str,
        *,
        signature: Signature | None = None,
        max_depth: int | None = None,
    ) -> None:
        self.spec = spec if isinstance(spec, DFOCISpec) else load_dfoci(spec)
        self.signature = signature
        self.max_depth = max_depth
        self._relevant: dict[tuple[str, tuple[str, ...]], frozenset[Literal]] = {}

    def relevant(self, op: OperatorInstance) -> frozenset[Literal]:
        key = (op.name, op.spec.roles)
        if key not in self._relevant:
            self._relevant[key] = self.spec.relevant_literals(
                op.name, op.spec.roles, self.max_depth
            )
        return self._relevant[key]

    def abstract(self, state: State, op: OperatorInstance) -> Hashable:
        return self.project(state, op)

    def project(self, state: State, op: OperatorInstance) -> frozenset[Atom]:
        """The lifted relevant atoms (same as :meth:`abstract`, with a precise type)."""
        binding = op.binding
        types = state.types()
        var_types = dict(self.spec.types)
        for role in binding:
            var_types.pop(role, None)
        kept: set[Atom] = set()
        for lit in self.relevant(op):
            if not lit.positive:
                continue
            pattern = lit.atom.substitute(binding)
            for atom in state.atoms_with(pattern.pred):
                theta = unify(pattern, atom, types=types, signature=self.signature)
                if theta is None:
                    continue
                if all(
                    types.get(theta[v]) == var_types[v]
                    for v in pattern.variables()
                    if v in var_types
                ):
                    kept.add(atom)
        return frozenset(
            a.substitute({obj: f"?{role}" for role, obj in binding.items()}) for a in kept
        )
