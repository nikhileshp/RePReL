"""RePReL execution: plan, run each operator with its subtask agent, replan on failure."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from reprel.abstraction.abstraction import Abstraction
from reprel.agents.agent_pool import AgentPool
from reprel.core.domain import Domain
from reprel.core.state import State
from reprel.logging.transitions import TransitionLogger, TransitionRecord
from reprel.planning.operators import OperatorInstance
from reprel.planning.planner import Planner, PlanningFailure

from .results import EpisodeResult


@dataclass(frozen=True)
class ExecutorConfig:
    """Failure handling. The original RePReL had no operator budget and never replanned."""

    max_operator_steps: int | None = None
    replan_on_timeout: bool = True
    max_replans: int = 10


class RePReLExecutor:
    """Algorithm 1 of the paper with explicit failure handling.

    One episode: reset, plan, and for each grounded operator run its agent on abstract keys
    until the operator's termination condition holds. Only the active operator's agent is
    updated. The episode ends on environment ``done`` or the domain's step limit. If an
    operator exceeds ``max_operator_steps`` the executor replans from the current state.
    """

    def __init__(
        self,
        domain: Domain,
        planner: Planner,
        abstraction: Abstraction,
        pool: AgentPool,
        config: ExecutorConfig | None = None,
    ) -> None:
        self.domain = domain
        self.planner = planner
        self.abstraction = abstraction
        self.pool = pool
        self.config = config or ExecutorConfig()
        self.episode_counter = 0

    def run_episode(
        self,
        rng: np.random.Generator,
        epsilon: float,
        learn: bool,
        logger: TransitionLogger | None = None,
        initial_state: State | None = None,
    ) -> EpisodeResult:
        domain, cfg = self.domain, self.config
        state = initial_state if initial_state is not None else domain.reset(rng)
        goal = domain.goal(state)
        episode = self.episode_counter
        self.episode_counter += 1
        plan = self._plan(state, goal)
        steps, env_return, failures, replans = 0, 0.0, 0, 0
        run: list[str] = []
        idx = 0
        episode_over = False
        while not episode_over and idx < len(plan) and steps < domain.max_steps:
            op = plan[idx]
            if op.is_terminated(state, signature=domain.predicates):
                idx += 1
                continue
            run.append(str(op))
            agent = self.pool.get(op.name)
            key = self.abstraction.abstract(state, op)
            op_steps = 0
            while True:
                action_idx = agent.act(key, rng, epsilon)
                action = domain.actions[action_idx]
                tr = domain.step(state, action, rng)
                steps += 1
                op_steps += 1
                terminated = op.is_terminated(tr.next_state, signature=domain.predicates)
                reward = op.subtask_reward(tr.reward, tr.next_state, signature=domain.predicates)
                next_key = self.abstraction.abstract(tr.next_state, op)
                if learn:
                    agent.update(key, action_idx, reward, next_key, terminal=terminated or tr.done)
                if logger is not None:
                    logger.log(
                        TransitionRecord(
                            episode,
                            steps - 1,
                            op.name,
                            op.args,
                            state,
                            action,
                            tr.reward,
                            tr.done,
                            tr.next_state,
                            terminated,
                        )
                    )
                env_return += tr.reward
                state, key = tr.next_state, next_key
                if terminated:
                    idx += 1
                    break
                if tr.done or steps >= domain.max_steps:
                    episode_over = True
                    break
                if cfg.max_operator_steps is not None and op_steps >= cfg.max_operator_steps:
                    failures += 1
                    if cfg.replan_on_timeout and replans < cfg.max_replans:
                        replans += 1
                        plan = self._plan(state, goal)
                        idx = 0
                    else:
                        episode_over = True
                    break
        return EpisodeResult(
            env_return=env_return,
            env_steps=steps,
            success=domain.is_success(state),
            subtask_failures=failures,
            replans=replans,
            operators_run=tuple(run),
        )

    def _plan(self, state: State, goal: frozenset) -> list[OperatorInstance]:  # type: ignore[type-arg]
        try:
            return self.planner.plan(state, goal)
        except PlanningFailure:
            return []
