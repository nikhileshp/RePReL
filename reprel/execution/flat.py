"""Flat Q-learning baseline: one agent, full ground state, environment reward only."""

from __future__ import annotations

import numpy as np

from reprel.agents.q_learning import Agent
from reprel.core.domain import Domain
from reprel.core.state import State
from reprel.logging.transitions import TransitionLogger, TransitionRecord

from .results import EpisodeResult


class FlatExecutor:
    def __init__(self, domain: Domain, agent: Agent) -> None:
        self.domain = domain
        self.agent = agent
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
        key = state.atoms
        while steps < domain.max_steps:
            action_idx = self.agent.act(key, rng, epsilon)
            action = domain.actions[action_idx]
            tr = domain.step(state, action, rng)
            steps += 1
            next_key = tr.next_state.atoms
            if learn:
                self.agent.update(key, action_idx, tr.reward, next_key, terminal=tr.done)
            if logger is not None:
                logger.log(
                    TransitionRecord(
                        episode,
                        steps - 1,
                        "flat",
                        (),
                        state,
                        action,
                        tr.reward,
                        tr.done,
                        tr.next_state,
                        tr.done,
                    )
                )
            env_return += tr.reward
            state, key = tr.next_state, next_key
            if tr.done:
                break
        return EpisodeResult(env_return, steps, domain.is_success(state), operators_run=("flat",))
