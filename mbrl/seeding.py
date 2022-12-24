import random

import numpy as np
import torch


def create_seed():
    return np.random.randint(0, 2**32)


class Seeding:
    SEED = create_seed()

    @classmethod
    def maybe_generate_seed(cls, seed=None):
        if seed:
            cls.SEED = seed

    @classmethod
    def set_seed(cls, seed=None):
        cls.maybe_generate_seed(seed)
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cls.SEED)
        np.random.seed(cls.SEED)
        random.seed(cls.SEED)


def np_random_seeding(seed=None):
    if seed is not None and not (isinstance(seed, int) and 0 <= seed):
        raise ValueError("Seed must be a non-negative integer or omitted, not {}".format(seed))

    if seed is None:
        seed = create_seed()

    rng = np.random.RandomState()
    rng.seed(seed)
    return rng, seed
