"""Random streams of an experiment.

Every sample has its own stream, derived from the master seed and a key that
identifies the sample. Nothing else feeds the draws, so a sample is the same
whatever the number of workers, the order of execution or the chunking.
"""

from __future__ import annotations

import hashlib

import numpy as np

SCHEMES = ("positional", "content", "common")


def sample_stream(seed: int, *key: int) -> np.random.Generator:
    """Independent stream for the sample identified by `key`, whatever the execution order."""
    return np.random.default_rng(np.random.SeedSequence(int(seed), spawn_key=key))


def text_key(text: str) -> int:
    """A stable 64-bit integer for a piece of text, the same on every platform and run."""
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def seed_key(scheme: str, fault_index: int, fault_id: str, replica: int) -> tuple[int, ...]:
    """Key of the stream of one sample.

    - `positional`: (fault index, replica). The stream of a fault depends on its place
      in the fault list: adding or reordering faults changes the samples of the others.
    - `content`: (hash of the fault identifier, replica). A fault has the same samples
      in every experiment that contains it.
    - `common`: (replica,). Replica r is the same drawn circuit for every fault and for
      the healthy case (common random numbers): comparisons between conditions are
      paired, and the conditions are no longer statistically independent.
    """
    if scheme == "positional":
        return (fault_index, replica)
    if scheme == "content":
        return (text_key(fault_id), replica)
    if scheme == "common":
        return (replica,)
    raise ValueError(f"unknown seeding scheme {scheme!r}; expected one of {SCHEMES}")
