"""GTPyhop-backed HTN planner.

OperatorSpecs become GTPyhop actions automatically (precondition check by unification,
add/delete effects). Task methods are plain callables ``(state: State, *args) -> list[task]
| None`` written against our immutable :class:`State`; a task is a tuple
``("task_name", arg1, ...)`` naming either a compound task or an operator.
"""

from __future__ import annotations

import contextlib
import io
import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any

os.environ.setdefault("GTPYHOP_QUIET", "true")
os.environ.setdefault("GTPYHOP_WARN_GLOBALS", "false")

import gtpyhop  # noqa: E402

from reprel.core.query import Signature
from reprel.core.state import State

from .operators import Goal, OperatorInstance, OperatorSpec
from .planner import Planner, PlanningFailure

Task = tuple[Any, ...]
Method = Callable[..., Sequence[Task] | None]


class _PlanningState(gtpyhop.State):  # type: ignore[misc]
    """GTPyhop state wrapper holding one immutable relational State in ``rs``."""

    def __init__(self, rs: State, name: str = "s") -> None:
        super().__init__(name)
        self.rs = rs

    def copy(self, new_name: str | None = None) -> _PlanningState:
        return _PlanningState(self.rs, new_name or self.__name__)


class GTPyhopPlanner(Planner):
    """HTN planner over OperatorSpecs and task methods.

    Args:
        operators: Operator specifications; each becomes a GTPyhop action of the same name.
        methods: Compound task name -> ordered list of method callables.
        root_task: Builds the initial todo list from a goal, e.g. ``lambda g: [("achieve", g)]``.
        signature: Predicate signature used for typed unification.
        name: GTPyhop domain name (must be unique per process).
    """

    def __init__(
        self,
        operators: Sequence[OperatorSpec],
        methods: Mapping[str, Sequence[Method]],
        root_task: Callable[[Goal], Sequence[Task]],
        *,
        signature: Signature | None = None,
        name: str = "reprel",
    ) -> None:
        self._specs = {op.name: op for op in operators}
        self._root_task = root_task
        self._signature = signature
        self._domain = gtpyhop.Domain(name)
        gtpyhop.declare_actions(*(self._make_action(op) for op in operators))
        for task, fns in methods.items():
            gtpyhop.declare_task_methods(
                task, *(self._make_method(task, i, fn) for i, fn in enumerate(fns))
            )
        self._session = gtpyhop.PlannerSession(
            domain=self._domain, verbose=0, structured_logging=False, memory_tracking=False
        )

    @property
    def operators(self) -> Mapping[str, OperatorSpec]:
        return self._specs

    def plan(self, state: State, goal: Goal) -> list[OperatorInstance]:
        # GTPyhop's session prints when it saves/restores the global verbosity level.
        with contextlib.redirect_stdout(io.StringIO()):
            result = self._session.find_plan(_PlanningState(state), list(self._root_task(goal)))
        if not result.success or result.plan is None:
            raise PlanningFailure(result.error or "no plan found")
        return [self._specs[step[0]].instantiate(tuple(step[1:])) for step in result.plan]

    # ------------------------------------------------------------------ adapters
    def _make_action(self, spec: OperatorSpec) -> Callable[..., _PlanningState | bool]:
        def action(gs: _PlanningState, *args: str) -> _PlanningState | bool:
            if len(args) != len(spec.roles):
                return False
            nxt = spec.apply(
                gs.rs, dict(zip(spec.roles, args, strict=True)), signature=self._signature
            )
            if nxt is None:
                return False
            gs.rs = nxt
            return gs

        action.__name__ = spec.name
        return action

    @staticmethod
    def _make_method(task: str, index: int, fn: Method) -> Callable[..., Sequence[Task] | None]:
        def method(gs: _PlanningState, *args: Any) -> Sequence[Task] | None:
            return fn(gs.rs, *args)

        method.__name__ = f"{task}_m{index}_{getattr(fn, '__name__', 'method')}"
        return method
