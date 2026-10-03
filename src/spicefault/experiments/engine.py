"""Parallel, resumable execution of a list of independent simulation tasks.

The application supplies the tasks and a `worker(task, context)` function returning
(row, waveform): `row` is a dict of scalars describing the simulation and its
results, `waveform` a 1-D array or None. The worker must be a module-level function
and derive any random numbers from the task itself (see `seeding.sample_stream`),
so the results do not depend on the number of processes.
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .. import __version__
from ..dataset.store import assemble, write_manifest
from ..simulation.ngspice import ngspice_version

Worker = Callable[[Any, Any], "tuple[dict, np.ndarray | None]"]

_WORKER: Worker | None = None
_CONTEXT: Any = None


def _init_worker(worker: Worker, context: Any) -> None:
    global _WORKER, _CONTEXT
    _WORKER, _CONTEXT = worker, context


def _call(task: Any) -> tuple[dict, np.ndarray | None]:
    return _WORKER(task, _CONTEXT)


def config_key(config: Any) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def run_chunks(
    tasks: Sequence[Any],
    worker: Worker,
    context: Any,
    parts_dir: str | Path,
    n_points: int | None = None,
    workers: int = 1,
    chunk: int = 2000,
    progress: bool = True,
) -> list[Path]:
    """Simulate `tasks` in chunks, one pair of files per chunk; finished chunks are kept.

    This is what makes a long campaign resumable: a chunk is written only once it is
    complete, and a chunk already on disk is not simulated again. Rows get a
    `sample_id` equal to the position of their task. With `n_points`, waveforms are
    stored as float32 [n, n_points], with NaN where the worker returned none.
    """
    parts_dir = Path(parts_dir)
    parts_dir.mkdir(parents=True, exist_ok=True)
    starts = range(0, len(tasks), chunk)
    stems = [parts_dir / f"part_{start:07d}" for start in starts]
    done, t0 = 0, time.time()
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init_worker, initargs=(worker, context)
    ) as pool:
        for stem, start in zip(stems, starts, strict=True):
            if stem.with_suffix(".parquet").exists():
                continue
            subset = tasks[start : start + chunk]
            rows = []
            if n_points is not None:
                waveforms = np.full((len(subset), n_points), np.nan, dtype=np.float32)
            for i, (row, wave) in enumerate(pool.map(_call, subset, chunksize=16)):
                rows.append({"sample_id": start + i, **row})
                if wave is not None and n_points is not None:
                    waveforms[i] = wave
                done += 1
                if progress and done % 500 == 0:
                    rate = done / (time.time() - t0)
                    print(f"{start + i + 1}/{len(tasks)} simulations ({rate:.0f}/s)", flush=True)
            if n_points is not None:
                np.save(stem.with_suffix(".npy"), waveforms)
            # the table is written last and renamed, so its presence marks a complete chunk
            tmp = stem.with_suffix(".tmp")
            pd.DataFrame(rows).to_parquet(tmp, index=False)
            tmp.rename(stem.with_suffix(".parquet"))
    return stems


def run_campaign(
    tasks: Sequence[Any],
    worker: Worker,
    context: Any,
    out_dir: str | Path,
    config: Any,
    n_points: int | None = None,
    workers: int = 1,
    chunk: int = 2000,
    progress: bool = True,
    summary: Callable[[pd.DataFrame], dict] | None = None,
) -> Path:
    """Simulate every task and write the dataset to `out_dir`.

    `config` is whatever defines the campaign; it must be JSON-serialisable and is
    copied into the manifest. If a previous run on the same folder and configuration
    was interrupted, it resumes after the last complete chunk. `summary` adds
    application counts, computed from the samples, to the manifest.
    """
    out_dir = Path(out_dir)
    parts_dir = out_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    key_file = parts_dir / "config.sha256"
    key = config_key({"config": config, "chunk": chunk, "n_tasks": len(tasks)})
    if key_file.exists() and key_file.read_text() != key:
        raise ValueError(
            f"{parts_dir} holds a partial run made with another configuration; "
            "delete it or choose another output folder"
        )
    resumed = any(parts_dir.glob("part_*.parquet"))
    key_file.write_text(key)

    t0 = time.time()
    stems = run_chunks(tasks, worker, context, parts_dir, n_points, workers, chunk, progress)
    df, _ = assemble(stems, out_dir)
    manifest = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "spicefault_version": __version__,
        "simulator": ngspice_version(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "platform": platform.platform(),
        "n_samples": len(df),
        "workers": workers,
        "chunk": chunk,
        "resumed": resumed,
        "elapsed_last_run_s": round(time.time() - t0, 1),
        **(summary(df) if summary is not None else {}),
        "config": config,
    }
    write_manifest(out_dir, manifest)
    shutil.rmtree(parts_dir)
    return out_dir
