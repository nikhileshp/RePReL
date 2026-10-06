"""Deterministic seeding. All randomness flows through the returned Generator."""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int) -> np.random.Generator:
    """Seed the stdlib and legacy NumPy RNGs and return a fresh ``Generator`` for ``seed``."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return np.random.default_rng(seed)
