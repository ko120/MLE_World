"""Seed the search process independently for each harness repetition."""

import random


def seed_runtime(value):
    if value is None:
        return None
    seed = int(value)
    if str(seed) != str(value) or not 0 <= seed < 2**32:
        raise ValueError("AIDE_RUN_SEED must be an integer in [0, 2**32)")
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    return seed
