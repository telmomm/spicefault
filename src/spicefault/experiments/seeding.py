"""Random streams of a campaign."""

from __future__ import annotations

import numpy as np


def sample_stream(seed: int, *key: int) -> np.random.Generator:
    """Independent stream for the sample identified by `key`, whatever the execution order.

    The key is positional, typically (condition index, replica): the stream of a
    condition then depends on its place in the fault list.
    """
    return np.random.default_rng(np.random.SeedSequence(int(seed), spawn_key=key))
