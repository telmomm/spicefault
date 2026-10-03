"""Reading and writing of a dataset directory.

A dataset is a directory with:
- samples.parquet  one row per simulation;
- waveforms.npy    float32 [n_samples, n_points], row-aligned with the table (optional);
- manifest.json    configuration, software versions and counts.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SAMPLES, WAVEFORMS, MANIFEST = "samples.parquet", "waveforms.npy", "manifest.json"


def assemble(stems: list[Path], out_dir: str | Path) -> tuple[pd.DataFrame, np.ndarray | None]:
    """Join the chunks written by a campaign into `out_dir`; returns (samples, waveforms)."""
    out_dir = Path(out_dir)
    df = pd.concat([pd.read_parquet(s.with_suffix(".parquet")) for s in stems], ignore_index=True)
    df.to_parquet(out_dir / SAMPLES, index=False)
    if not all(s.with_suffix(".npy").exists() for s in stems):
        return df, None
    waveforms = np.concatenate([np.load(s.with_suffix(".npy")) for s in stems])
    np.save(out_dir / WAVEFORMS, waveforms)
    return df, waveforms


def write_manifest(out_dir: str | Path, manifest: dict) -> None:
    (Path(out_dir) / MANIFEST).write_text(json.dumps(manifest, indent=2))


def load_dataset(
    path: str | Path, drop_failed: bool = True, ok_column: str = "sim_ok"
) -> tuple[pd.DataFrame, np.ndarray | None, dict]:
    """Return (samples, waveforms, manifest). Failed simulations are dropped on request."""
    path = Path(path)
    df = pd.read_parquet(path / SAMPLES)
    waveforms = np.load(path / WAVEFORMS) if (path / WAVEFORMS).exists() else None
    manifest = json.loads((path / MANIFEST).read_text())
    if drop_failed and ok_column in df:
        keep = df[ok_column].to_numpy()
        df = df[keep].reset_index(drop=True)
        waveforms = waveforms[keep] if waveforms is not None else None
    return df, waveforms, manifest
