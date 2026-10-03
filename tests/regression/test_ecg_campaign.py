"""A reduced ECG campaign gives the same dataset through both engines, for any worker count."""

import copy

import numpy as np
import pandas as pd
import pytest
from ecgfd.dataset import build_tasks, simulate_parts
from ecgfd.specs import with_nominal_gain

import ecg_adapter
from spicefault.dataset import load_dataset
from spicefault.experiments import run_campaign, run_chunks

pytestmark = [pytest.mark.ecgfd, pytest.mark.ngspice]

N_POINTS = 1000


def read(stems):
    df = pd.concat([pd.read_parquet(s.with_suffix(".parquet")) for s in stems], ignore_index=True)
    return df, np.concatenate([np.load(s.with_suffix(".npy")) for s in stems])


@pytest.fixture(scope="module")
def study(cfg):
    """(configuration, tasks): three cases of every fault kind, healthy included."""
    cfg = copy.deepcopy(cfg)
    cfg["dataset"] = {"n_healthy": 3, "n_per_fault": 1}
    cfg = with_nominal_gain(cfg)
    by_kind = {}
    for task in build_tasks(cfg):
        by_kind.setdefault(task[1].kind, []).append(task)
    assert len(by_kind) == {"integrated": 12, "reference": 9}[cfg["circuit"]]
    # spread within each kind, so that different components and magnitudes are hit
    return cfg, [t for tasks in by_kind.values() for t in tasks[:: max(len(tasks) // 3, 1)][:3]]


@pytest.fixture(scope="module")
def baseline(study, tmp_path_factory):
    cfg, tasks = study
    parts = tmp_path_factory.mktemp("baseline")
    return read(simulate_parts(tasks, cfg, parts, jobs=4, chunk=2000, progress=False))


@pytest.mark.parametrize("workers, chunk", [(1, 2000), (4, 7)])
def test_chunks_are_identical_to_the_baseline(study, baseline, tmp_path, workers, chunk):
    cfg, tasks = study
    stems = run_chunks(
        tasks, ecg_adapter.simulate_task, cfg, tmp_path, N_POINTS, workers, chunk, progress=False
    )
    df, waveforms = read(stems)
    assert df["sim_ok"].all()
    pd.testing.assert_frame_equal(df, baseline[0], check_exact=True)
    assert np.array_equal(waveforms, baseline[1])


def test_campaign_dataset_matches_the_baseline(study, baseline, tmp_path):
    cfg, tasks = study
    out = run_campaign(
        tasks, ecg_adapter.simulate_task, cfg, tmp_path / "data", config=cfg, n_points=N_POINTS,
        workers=4, chunk=11, progress=False,
        summary=lambda df: {"n_failed": int((~df["sim_ok"]).sum())},
    )
    df, waveforms, manifest = load_dataset(out)
    pd.testing.assert_frame_equal(df, baseline[0], check_exact=True)
    assert np.array_equal(waveforms, baseline[1])
    assert manifest["config"] == cfg and manifest["n_failed"] == 0
