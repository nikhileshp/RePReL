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

from reprel.core.atoms import ANY_TYPE, Atom, Literal, is_variable, type_compatible, unify
from reprel.core.query import Signature
from reprel.core.state import State
from reprel.planning.operators import OperatorInstance

from .abstraction import Abstraction, register_abstraction

SOURCES = ("hand", "learned")
TOP_KEYS = {"domain", "source", "types", "reward_parents", "statements", "operators"}
STATEMENT_KEYS = {"operator", "if", "influences", "target"}
OPERATOR_KEYS = {"reward_parents", "termination_parents"}
_APART = "__"  # separator used to rename statement-local variables apart


def _check_keys(data: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"unknown key(s) {sorted(unknown)} in {where}; allowed: {sorted(allowed)}")


def _parse_head(text: str) -> tuple[str, tuple[str, ...]]:
    atom = Atom.parse(text) if "(" in text else Atom(text.strip(), ())
    return atom.pred, atom.args


def _lits(items: Iterable[str] | None, where: str = "") -> tuple[Literal, ...]:
    out = []
    for text in items or ():
        try:
            out.append(Literal.parse(text))
        except ValueError as exc:
            raise ValueError(
                f"{exc} in {where or 'literal list'}: atoms contain commas, so YAML lists must "
                "use block style (one '- item' per line) or quote each atom"
            ) from exc
    return tuple(out)


def _base_var(name: str) -> str:
    """Original variable name, before apart-renaming (``L1__s0`` -> ``L1``)."""
    return name.split(_APART, 1)[0]


def _rename_apart(lit: Literal, roles: Mapping[str, str], tag: str) -> Literal:
    """Map operator-argument variables to roles; every other variable becomes ``V__tag``."""
    theta = {v: roles.get(v, f"{v}{_APART}{tag}") for v in lit.atom.variables()}
    return lit.substitute(theta)


@dataclass(frozen=True)
class Statement:
    influences: tuple[Literal, ...]
    target: Literal
    operator: str | None = None
    operator_args: tuple[str, ...] = ()
    context: tuple[Literal, ...] = ()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Statement:
        _check_keys(data, STATEMENT_KEYS, f"statement {data!r}")
        if "target" not in data or "influences" not in data:
            raise ValueError(f"statement needs 'influences' and 'target': {data!r}")
        operator: str | None = None
        args: tuple[str, ...] = ()
        if data.get("operator"):
            operator, args = _parse_head(data["operator"])
        return cls(
            influences=_lits(data["influences"], "influences"),
            target=Literal.parse(data["target"]),
            operator=operator,
            operator_args=args,
            context=_lits(data.get("if"), "if"),
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
        _check_keys(data, TOP_KEYS, "D-FOCI spec")
        source = data.get("source", "hand")
        if source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}, got {source!r}")
        operators: dict[str, OperatorParents] = {}
        for head, parents in (data.get("operators") or {}).items():
            name, args = _parse_head(head)
            parents = parents or {}
            _check_keys(parents, OPERATOR_KEYS, f"operators[{head}]")
            operators[name] = OperatorParents(
                args=args,
                reward_parents=_lits(parents.get("reward_parents"), f"{head}.reward_parents"),
                termination_parents=_lits(
                    parents.get("termination_parents"), f"{head}.termination_parents"
                ),
            )
        return cls(
            domain=str(data.get("domain", "")),
            statements=tuple(Statement.from_dict(s) for s in data.get("statements") or ()),
            operators=operators,
            reward_parents=_lits(data.get("reward_parents"), "reward_parents"),
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

    def variable_type(self, name: str) -> str | None:
        """Declared type of a (possibly apart-renamed) variable, if any."""
        return self.types.get(_base_var(name))

    # ------------------------------------------------------------------ closure
    def relevant_literals(
        self, operator: str, roles: Sequence[str] | None = None, max_depth: int | None = None
    ) -> frozenset[Literal]:
        """Ancestor closure of the operator's reward/termination parents (plus global R parents).

        Variables naming operator arguments are renamed to ``roles`` (positionally; the
        operator header's own names when ``roles`` is None). Every other variable is
        existential and renamed apart per statement (``L1`` -> ``L1__s3``) so that a
        statement-local name can never be captured by a role or by another statement.
        """
        if operator not in self.operators:
            raise KeyError(f"no D-FOCI parents for operator {operator!r}")
        parents = self.operators[operator]
        role_names = tuple(roles) if roles else parents.args
        if len(role_names) != len(parents.args):
            raise ValueError(f"operator args {parents.args} do not match roles {role_names}")
        op_roles = dict(zip(parents.args, role_names, strict=True))
        relevant: set[Literal] = {_rename_apart(lit, {}, "r") for lit in self.reward_parents}
        relevant |= {_rename_apart(lit, op_roles, "o") for lit in parents.reward_parents}
        relevant |= {_rename_apart(lit, op_roles, "o") for lit in parents.termination_parents}
        applicable: list[tuple[Statement, dict[str, str], str]] = []
        for i, s in enumerate(self.statements):
            if s.operator is None:
                applicable.append((s, {}, f"s{i}"))
            elif s.operator == operator:
                if len(s.operator_args) != len(role_names):
                    raise ValueError(f"statement {s} args do not match operator {operator}")
                applicable.append((s, dict(zip(s.operator_args, role_names, strict=True)), f"s{i}"))
        depth = 0
        while max_depth is None or depth < max_depth:
            added: set[Literal] = set()
            for statement, stmt_roles, tag in applicable:
                target = _rename_apart(statement.target, stmt_roles, tag)
                if any(_compatible(target.atom, lit.atom) for lit in relevant):
                    for lit in (*statement.influences, *statement.context):
                        renamed = _rename_apart(lit, stmt_roles, tag)
                        if renamed not in relevant:
                            added.add(renamed)
            if not added:
                break
            relevant |= added
            depth += 1
        return frozenset(relevant)


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


@dataclass(frozen=True)
class _Pattern:
    """A relevant literal prepared for projection under one operator's roles."""

    atom: Atom  # roles left as variables; existentials canonical V0, V1..
    var_types: tuple[tuple[str, str], ...]  # (variable, required type) for typed existentials
    simple: bool  # all args are distinct variables -> type filter suffices


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
        self._patterns: dict[tuple[str, tuple[str, ...]], tuple[_Pattern, ...]] = {}

    def relevant(self, op: OperatorInstance) -> frozenset[Literal]:
        key = (op.name, op.spec.roles)
        if key not in self._relevant:
            self._relevant[key] = self.spec.relevant_literals(
                op.name, op.spec.roles, self.max_depth
            )
        return self._relevant[key]

    def patterns(self, op: OperatorInstance) -> tuple[_Pattern, ...]:
        """Deduplicated projection patterns for the operator (cached per operator name)."""
        key = (op.name, op.spec.roles)
        if key not in self._patterns:
            roles = set(op.spec.roles)
            unique: dict[tuple[Atom, tuple[tuple[str, str], ...]], _Pattern] = {}
            for lit in self.relevant(op):
                canon: dict[str, str] = {}
                var_types: list[tuple[str, str]] = []
                for v in lit.atom.variables():
                    if v in roles:
                        continue
                    canon[v] = f"V{len(canon)}"
                    declared = self.spec.variable_type(v)
                    if declared is not None:
                        var_types.append((canon[v], declared))
                atom = lit.atom.substitute(canon)
                simple = all(is_variable(a) and a not in roles for a in atom.args) and len(
                    set(atom.args)
                ) == len(atom.args)
                unique.setdefault(
                    (atom, tuple(var_types)), _Pattern(atom, tuple(var_types), simple)
                )
            self._patterns[key] = tuple(unique.values())
        return self._patterns[key]

    def abstract(self, state: State, op: OperatorInstance) -> Hashable:
        return self.project(state, op)

    def project(self, state: State, op: OperatorInstance) -> frozenset[Atom]:
        """The lifted relevant atoms (same as :meth:`abstract`, with a precise type)."""
        binding = op.binding
        types = state.types()
        kept: set[Atom] = set()
        for pattern in self.patterns(op):
            bound = pattern.atom.substitute(binding)
            candidates = state.atoms_with(bound.pred)
            if not candidates:
                continue
            if pattern.simple:
                positions = [(bound.args.index(v), t) for v, t in pattern.var_types]
                kept.update(
                    atom
                    for atom in candidates
                    if len(atom.args) == len(bound.args)
                    and all(
                        type_compatible(types.get(atom.args[i], ANY_TYPE), t) for i, t in positions
                    )
                )
                continue
            for atom in candidates:
                theta = unify(bound, atom, types=types, signature=self.signature)
                if theta is not None and all(
                    type_compatible(types.get(theta[v], ANY_TYPE), t) for v, t in pattern.var_types
                ):
                    kept.add(atom)
        lift = {obj: f"?{role}" for role, obj in binding.items()}
        objs = set(lift)
        return frozenset(a.substitute(lift) if objs.intersection(a.args) else a for a in kept)
