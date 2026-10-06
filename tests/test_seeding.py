import random

import numpy as np

from reprel.seeding import seed_everything


def test_same_seed_gives_same_generator_stream() -> None:
    a = seed_everything(123)
    b = seed_everything(123)
    assert isinstance(a, np.random.Generator)
    assert a.integers(0, 1000, size=5).tolist() == b.integers(0, 1000, size=5).tolist()


def test_different_seeds_differ() -> None:
    a = seed_everything(1).integers(0, 10**9)
    b = seed_everything(2).integers(0, 10**9)
    assert a != b


def test_seed_everything_also_seeds_stdlib_random() -> None:
    seed_everything(7)
    x = random.random()
    seed_everything(7)
    assert random.random() == x
