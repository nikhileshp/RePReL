"""Greedy evaluation episodes and the training loop with periodic evaluation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np

from reprel.agents.schedules import EpsilonSchedule
from reprel.logging.transitions import TransitionLogger

from .results import EpisodeResult, EvalResult, TrainingPoint


class EpisodeRunner(Protocol):
    def run_episode(
        self,
        rng: np.random.Generator,
        epsilon: float,
        learn: bool,
        logger: TransitionLogger | None = None,
    ) -> EpisodeResult: ...


def evaluate(
    runner: EpisodeRunner, rng: np.random.Generator, episodes: int, epsilon: float
) -> EvalResult:
    """Run ``episodes`` episodes without learning and summarise them."""
    if episodes < 1:
        raise ValueError("episodes must be >= 1")
    results = [runner.run_episode(rng, epsilon, learn=False) for _ in range(episodes)]
    returns = np.array([r.env_return for r in results])
    successes = [r.env_steps for r in results if r.success]
    return EvalResult(
        return_mean=float(returns.mean()),
        return_std=float(returns.std()),
        success_rate=len(successes) / episodes,
        mean_steps_to_success=float(np.mean(successes)) if successes else float("nan"),
        episodes=episodes,
    )


def train(
    runner: EpisodeRunner,
    rng: np.random.Generator,
    *,
    total_steps: int,
    eval_every: int,
    eval_episodes: int,
    eval_epsilon: float,
    schedule: EpsilonSchedule,
    eval_rng: np.random.Generator | None = None,
    logger: TransitionLogger | None = None,
    on_eval: Callable[[TrainingPoint], None] | None = None,
) -> list[TrainingPoint]:
    """Train for ``total_steps`` environment steps, evaluating every ``eval_every`` steps.

    Evaluation happens at the first episode boundary after each multiple of ``eval_every``
    and after the final episode, using a separate RNG so it never perturbs training.
    """
    eval_rng = eval_rng if eval_rng is not None else np.random.default_rng(rng.integers(2**32))
    history: list[TrainingPoint] = []
    steps, episode, next_eval = 0, 0, eval_every
    while steps < total_steps:
        epsilon = schedule.value(episode)
        result = runner.run_episode(rng, epsilon, learn=True, logger=logger)
        steps += result.env_steps
        episode += 1
        if steps >= next_eval or steps >= total_steps:
            point = TrainingPoint(
                steps, episode, epsilon, evaluate(runner, eval_rng, eval_episodes, eval_epsilon)
            )
            history.append(point)
            if on_eval is not None:
                on_eval(point)
            while next_eval <= steps:
                next_eval += eval_every
    return history
