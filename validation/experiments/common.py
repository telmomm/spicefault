"""What the experiment scripts share: where data and results live, and how a dataset
becomes an analysis.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from spicefault import Dataset
from spicefault.reliability import ReliabilityAnalysis
from validation.study import Study

DATA = Path("data") / "validation"
RESULTS = Path("results")
ALPHA, BETA, THRESHOLD = 0.01, 0.1, 3.0


def analysis_of(study: Study, folder: Path, seed: int = 0, **kwargs) -> ReliabilityAnalysis:
    """The analysis of a campaign of a study.

    Compliance is decided on the simulated values; detection works on the values as
    the instrument reads them, with its noise (the measurement model of the study).
    """
    dataset = Dataset(folder)
    problems = dataset.verify()
    if problems:
        raise ValueError(f"{folder} is not intact: {problems}")
    samples = dataset.samples
    observed = study.observed(samples, seed)
    observed["compliant"] = study.compliant(samples)
    return ReliabilityAnalysis(
        observed,
        study.features,
        compliant="compliant",
        noise_floor=study.noise,
        faults=dataset.metadata["faults"],
        seed=seed,
        **kwargs,
    )


def save(experiment: str, name: str, result: dict, tables: dict[str, pd.DataFrame]) -> Path:
    """Write a result as JSON, and its tables as CSV, under results/<experiment>/."""
    folder = RESULTS / experiment
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.json").write_text(json.dumps(result, indent=2, default=_plain))
    for label, table in tables.items():
        table.to_csv(folder / f"{name}_{label}.csv")
    return folder / f"{name}.json"


def _plain(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")
