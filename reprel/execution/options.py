"""Option-based baselines reconstructed from the paper's description of Taxi experiments.

Both baselines use one option per *location* (``reach(L)``), with full-state keys and no
abstraction. Every option's table is updated at every step from the shared experience
(off-policy), each with its own pseudo-reward: the environment reward plus a terminal bonus
when the taxi arrives at that option's location.

* Taskable RL (``trl``, Illanes et al. 2020, "seq" variant): the HTN plan supplies the
  sequence of operators; the option executed for an operator is the one for its target
  location (the passenger's pickup depot, or their destination). The option runs until the
  operator's termination condition holds, exactly like a RePReL subtask.
* Option-based HRL (``hrl``): no planner. A tabular meta-controller (SMDP Q-learning over the
  full state) chooses among the reach options and the primitive ``pickup``/``dropoff``
  actions. A reach option runs until the taxi is at its location (at least one step).

The exact tabular definitions used for the paper's Taxi figures were never released; this is
a faithful reading of the text ("4 options for each location R, G, B, and Y"; "Taskable RL
updates all the subtask policies for every step").
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from reprel.agents.agent_pool import AgentPool
from reprel.agents.q_learning import QLearningAgent
from reprel.core.atoms import Atom
from reprel.core.domain import Domain
from reprel.core.state import State
from reprel.logging.transitions import TransitionLogger, TransitionRecord
from reprel.planning.operators import OperatorInstance
from reprel.planning.planner import Planner, PlanningFailure

from .results import EpisodeResult


@dataclass(frozen=True)
class ReachOption:
    """Subgoal option: terminates when ``goal`` holds; pseudo-reward adds ``terminal_reward``."""

    name: str
    goal: Atom
    terminal_reward: float

    def is_terminated(self, state: State) -> bool:
        return state.holds(self.goal)

    def reward(self, env_reward: float, next_state: State) -> float:
        return env_reward + (self.terminal_reward if self.is_terminated(next_state) else 0.0)


OperatorTarget = Callable[[OperatorInstance, State], str]


def _update_all_options(
    options: Sequence[ReachOption],
    pool: AgentPool,
    key: frozenset[Atom],
    action_idx: int,
    env_reward: float,
    next_state: State,
    env_done: bool,
) -> None:
    next_key = next_state.atoms
    for option in options:
        terminated = option.is_terminated(next_state)
        pool.get(option.name).update(
            key,
            action_idx,
            option.reward(env_reward, next_state),
            next_key,
            terminal=terminated or env_done,
        )


class TRLExecutor:
    """Planner-provided operator sequence executed by location options (full-state keys)."""

    def __init__(
        self,
        domain: Domain,
        planner: Planner,
        options: Sequence[ReachOption],
        operator_target: OperatorTarget,
        pool: AgentPool,
    ) -> None:
        self.domain = domain
        self.planner = planner
        self.options = tuple(options)
        self._by_target = {o.goal.args[-1]: o for o in self.options}
        self.operator_target = operator_target
        self.pool = pool
        self.episode_counter = 0

    def run_episode(
        self,
        rng: np.random.Generator,
        epsilon: float,
        learn: bool,
        logger: TransitionLogger | None = None,
        initial_state: State | None = None,
    ) -> EpisodeResult:
        domain = self.domain
        state = initial_state if initial_state is not None else domain.reset(rng)
        episode = self.episode_counter
        if learn or logger is not None:
            self.episode_counter += 1
        try:
            plan = self.planner.plan(state, domain.goal(state))
            planning_failures = 0
        except PlanningFailure:
            plan, planning_failures = [], 1
        steps, env_return = 0, 0.0
        run: list[str] = []
        episode_over = False
        for op in plan:
            if episode_over:
                break
            if op.is_terminated(state, signature=domain.predicates):
                continue
            option = self._by_target[self.operator_target(op, state)]
            run.append(f"{op}->{option.name}")
            agent = self.pool.get(option.name)
            while True:
                key = state.atoms
                action_idx = agent.act(key, rng, epsilon)
                tr = domain.step(state, domain.actions[action_idx], rng)
                steps += 1
                if learn:
                    _update_all_options(
                        self.options, self.pool, key, action_idx, tr.reward, tr.next_state, tr.done
                    )
                terminated = op.is_terminated(tr.next_state, signature=domain.predicates)
                if logger is not None:
                    logger.log(
                        TransitionRecord(
                            episode,
                            steps - 1,
                            op.name,
                            op.args,
                            state,
                            domain.actions[action_idx],
                            tr.reward,
                            tr.done,
                            tr.next_state,
                            terminated,
                        )
                    )
                env_return += tr.reward
                state = tr.next_state
                if tr.done or steps >= domain.max_steps:
                    episode_over = True
                    break
                if terminated:
                    break
        return EpisodeResult(
            env_return,
            steps,
            domain.is_success(state),
            planning_failures=planning_failures,
            operators_run=tuple(run),
        )


class HRLExecutor:
    """Options + primitive pickup/dropoff under a learned tabular meta-controller (SMDP Q)."""

    def __init__(
        self,
        domain: Domain,
        options: Sequence[ReachOption],
        pool: AgentPool,
        *,
        meta_alpha: float,
        gamma: float,
        primitives: Sequence[str] = ("pickup", "dropoff"),
    ) -> None:
        self.domain = domain
        self.options = tuple(options)
        self.pool = pool
        self.gamma = gamma
        self.primitives = tuple(p for p in primitives if p in domain.actions)
        self.meta = QLearningAgent(len(self.options) + len(self.primitives), meta_alpha, gamma)
        self.episode_counter = 0

    def run_episode(
        self,
        rng: np.random.Generator,
        epsilon: float,
        learn: bool,
        logger: TransitionLogger | None = None,
        initial_state: State | None = None,
    ) -> EpisodeResult:
        domain = self.domain
        state = initial_state if initial_state is not None else domain.reset(rng)
        episode = self.episode_counter
        if learn or logger is not None:
            self.episode_counter += 1
        steps, env_return = 0, 0.0
        run: list[str] = []
        while steps < domain.max_steps:
            meta_key = state.atoms
            choice = self.meta.act(meta_key, rng, epsilon)
            start_state = state
            smdp_return, k, done = 0.0, 0, False
            if choice >= len(self.options):
                action = self.primitives[choice - len(self.options)]
                run.append(action)
                action_idx = domain.action_index(action)
                tr = domain.step(state, action, rng)
                steps += 1
                if learn:
                    _update_all_options(
                        self.options,
                        self.pool,
                        state.atoms,
                        action_idx,
                        tr.reward,
                        tr.next_state,
                        tr.done,
                    )
                self._log(logger, episode, steps, action, state, tr, False)
                smdp_return, k, done = tr.reward, 1, tr.done
                env_return += tr.reward
                state = tr.next_state
            else:
                option = self.options[choice]
                run.append(option.name)
                agent = self.pool.get(option.name)
                while True:  # at least one step, then until the option's goal holds
                    key = state.atoms
                    action_idx = agent.act(key, rng, epsilon)
                    tr = domain.step(state, domain.actions[action_idx], rng)
                    steps += 1
                    if learn:
                        _update_all_options(
                            self.options,
                            self.pool,
                            key,
                            action_idx,
                            tr.reward,
                            tr.next_state,
                            tr.done,
                        )
                    terminated = option.is_terminated(tr.next_state)
                    self._log(
                        logger, episode, steps, domain.actions[action_idx], state, tr, terminated
                    )
                    smdp_return += self.gamma**k * tr.reward
                    k += 1
                    env_return += tr.reward
                    state = tr.next_state
                    done = tr.done
                    if terminated or done or steps >= domain.max_steps:
                        break
            if learn:
                # SMDP Q-learning: bootstrap with gamma^k from the state where the option ended
                row = self.meta._row(meta_key)
                nxt = self.meta.q.get(state.atoms)
                bootstrap = 0.0 if done or nxt is None else float(nxt.max())
                row[choice] += self.meta.alpha * (
                    smdp_return + (self.gamma**k) * bootstrap - row[choice]
                )
            if done:
                break
            del start_state
        return EpisodeResult(env_return, steps, domain.is_success(state), operators_run=tuple(run))

    @staticmethod
    def _log(
        logger: TransitionLogger | None,
        episode: int,
        steps: int,
        action: str,
        state: State,
        tr: object,
        terminated: bool,
    ) -> None:
        if logger is None:
            return
        from reprel.core.domain import Transition

        assert isinstance(tr, Transition)
        logger.log(
            TransitionRecord(
                episode,
                steps - 1,
                "hrl",
                (),
                state,
                action,
                tr.reward,
                tr.done,
                tr.next_state,
                terminated,
            )
        )
