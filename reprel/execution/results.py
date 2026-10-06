"""Result records shared by the executors, evaluation, and training."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EpisodeResult:
    env_return: float
    env_steps: int
    success: bool
    subtask_failures: int = 0
    replans: int = 0
    planning_failures: int = 0
    operators_run: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class EvalResult:
    return_mean: float
    return_std: float
    success_rate: float
    mean_steps_to_success: float
    episodes: int


@dataclass(frozen=True)
class TrainingPoint:
    env_steps: int
    episodes: int
    epsilon: float
    eval: EvalResult
