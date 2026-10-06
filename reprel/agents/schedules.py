"""Exploration-rate schedules, evaluated per episode."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


class EpsilonSchedule(ABC):
    @abstractmethod
    def value(self, episode: int) -> float:
        """Exploration probability for the given (0-based) training episode."""


@dataclass(frozen=True)
class LinearSchedule(EpsilonSchedule):
    """Linear decay from ``start`` to ``end`` over ``episodes`` episodes, then constant.

    The original RePReL code intended 0.75 -> 0.01 over 20 000 episodes.
    """

    start: float = 0.75
    end: float = 0.01
    episodes: int = 20_000

    def value(self, episode: int) -> float:
        if episode >= self.episodes:
            return self.end
        return self.start + (self.end - self.start) * episode / self.episodes


@dataclass(frozen=True)
class ConstantSchedule(EpsilonSchedule):
    epsilon: float

    def value(self, episode: int) -> float:
        return self.epsilon
