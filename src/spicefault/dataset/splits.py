"""Train/test splits that keep related fault samples together."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def split_by_replica(
    samples: pd.DataFrame, test_fraction: float, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Return row positions stratified by fault ID, keeping each replica's conditions together."""
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")
    if "fault_id" not in samples:
        raise KeyError("samples must contain a 'fault_id' column")
    rng = np.random.default_rng(seed)
    test = []
    fault_ids = samples["fault_id"].to_numpy()
    replicas = samples["replica"].to_numpy() if "replica" in samples else None
    for fault_id in pd.unique(fault_ids):
        indices = np.flatnonzero(fault_ids == fault_id)
        if replicas is not None:
            groups = [
                indices[replicas[indices] == replica] for replica in pd.unique(replicas[indices])
            ]
        else:
            groups = [np.asarray([index]) for index in indices]
        if len(groups) < 2:
            continue
        count = min(len(groups) - 1, max(1, round(len(groups) * test_fraction)))
        chosen = rng.permutation(len(groups))[:count]
        test.extend(np.concatenate([groups[i] for i in chosen]))
    test_indices = np.sort(np.asarray(test, dtype=np.int64))
    mask = np.ones(len(samples), dtype=bool)
    mask[test_indices] = False
    return np.flatnonzero(mask), test_indices


def split_by_magnitude(
    samples: pd.DataFrame,
    test_magnitudes: Sequence[float],
    magnitude_column: str = "fault_magnitude",
) -> tuple[np.ndarray, np.ndarray]:
    """Hold out every sample whose complete fault magnitude is listed."""
    if magnitude_column not in samples:
        raise KeyError(f"samples must contain a {magnitude_column!r} column")
    magnitudes = np.asarray(test_magnitudes, dtype=float)
    if magnitudes.ndim == 0:
        magnitudes = magnitudes.reshape(1)
    if magnitudes.ndim != 1:
        raise ValueError("test_magnitudes must be a one-dimensional sequence")
    test_mask = np.isin(samples[magnitude_column].to_numpy(dtype=float), magnitudes)
    return np.flatnonzero(~test_mask), np.flatnonzero(test_mask)