"""Reading and writing of a dataset directory.

A dataset is a directory with:
- samples.parquet  one row per simulation;
- waveforms.npy    float32 [n_stored, n_points] (optional): one row per sample whose
                   operating condition stores a waveform, in sample order. When every
                   condition stores one, that is one row per sample;
- metadata.json    the definition of what was simulated: enough to regenerate it;
- manifest.json    the record of the run: versions, counts, fingerprints of the files;
- any other file the application adds, such as the source netlist.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from .manifest import MANIFEST

SAMPLES, WAVEFORMS, METADATA = "samples.parquet", "waveforms.npy", "metadata.json"


class StoredWaveforms:
    """Waveforms aligned with the samples, read from a file that holds only some rows.

    A campaign in which some operating conditions store no waveform writes only the
    rows of the others. This reads them as the full array would be read: indexed by
    sample, with NaN where nothing is stored, without holding those rows in memory.
    `stored` is the array of the file and `rows` the sample of each of its rows.
    Only the samples can be indexed in place; `np.asarray` gives the full array.
    """

    def __init__(self, stored: np.ndarray, rows: np.ndarray, n_samples: int):
        if len(stored) != len(rows):
            raise ValueError(
                f"{len(stored)} stored waveforms for {len(rows)} samples that store one"
            )
        self.stored, self.rows = stored, np.asarray(rows, dtype=np.int64)
        self.shape = (int(n_samples), *stored.shape[1:])
        self.dtype = stored.dtype
        self._position = np.full(n_samples, -1, dtype=np.int64)
        self._position[self.rows] = np.arange(len(self.rows))

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, key):
        key, rest = (key[0], key[1:]) if isinstance(key, tuple) else (key, ())
        position = self._position[key]
        if np.ndim(position) == 0:
            if position < 0:
                found = np.full(self.shape[1:], np.nan, dtype=self.dtype)
            else:
                found = np.asarray(self.stored[int(position)])
            return found[rest] if rest else found
        found = np.full((len(position), *self.shape[1:]), np.nan, dtype=self.dtype)
        has = position >= 0
        found[has] = self.stored[position[has]]
        return found[(slice(None), *rest)] if rest else found

    def __array__(self, dtype=None, copy=None):
        full = self[:]
        return full if dtype is None else full.astype(dtype)


def assemble(
    stems: list[Path], out_dir: str | Path, waveform_rows: Sequence[bool] | None = None
) -> tuple[pd.DataFrame, np.ndarray | None]:
    """Join the chunks written by a campaign into `out_dir`; returns (samples, waveforms).

    With `waveform_rows`, one flag per sample, only the waveforms of the flagged samples
    are written, in sample order.
    """
    out_dir = Path(out_dir)
    df = pd.concat([pd.read_parquet(s.with_suffix(".parquet")) for s in stems], ignore_index=True)
    df.to_parquet(out_dir / SAMPLES, index=False)
    if not all(s.with_suffix(".npy").exists() for s in stems):
        return df, None
    parts, start = [], 0
    for stem in stems:
        part = np.load(stem.with_suffix(".npy"))
        stop = start + len(part)
        if waveform_rows is not None:
            part = part[np.asarray(waveform_rows[start:stop], dtype=bool)]
        parts.append(part)
        start = stop
    waveforms = np.concatenate(parts)
    np.save(out_dir / WAVEFORMS, waveforms)
    return df, waveforms


def read_waveforms(path: str | Path, samples: pd.DataFrame, manifest: dict, mmap: bool = False):
    """The waveforms of a dataset aligned with `samples`, or None if it stores none.

    `manifest` is the record of the dataset: its `waveform_conditions`, if any, are the
    operating conditions whose samples have a row in the file.
    """
    file = Path(path) / WAVEFORMS
    if not file.exists():
        return None
    stored = np.load(file, mmap_mode="r" if mmap else None)
    conditions = manifest.get("waveform_conditions")
    if conditions is None:
        return stored
    rows = np.flatnonzero(samples["condition"].isin(conditions).to_numpy())
    return StoredWaveforms(stored, rows, len(samples))


def load_metadata(path: str | Path) -> dict:
    """The definition of the campaign that wrote the dataset."""
    return json.loads((Path(path) / METADATA).read_text())


def load_dataset(
    path: str | Path, drop_failed: bool = True, ok_column: str = "sim_ok"
) -> tuple[pd.DataFrame, np.ndarray | None, dict]:
    """Return (samples, waveforms, manifest). Failed simulations are dropped on request."""
    path = Path(path)
    df = pd.read_parquet(path / SAMPLES)
    manifest = json.loads((path / MANIFEST).read_text())
    waveforms = read_waveforms(path, df, manifest)
    if waveforms is not None:
        waveforms = np.asarray(waveforms)
    if drop_failed and ok_column in df:
        keep = df[ok_column].to_numpy()
        df = df[keep].reset_index(drop=True)
        waveforms = waveforms[keep] if waveforms is not None else None
    return df, waveforms, manifest
