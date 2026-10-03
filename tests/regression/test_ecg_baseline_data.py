"""Cases of the published ECG dataset `data/v1`, simulated again through spicefault.

Tolerances are those of docs/EXPERIMENT_PLAN.md, section 3: realised parameters,
labels and waveforms must be equal; scalar features and specification values may
differ by 1e-9 relative. The dataset location comes from ECGFD_DATA.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ecgfd.config import REPO_ROOT
from ecgfd.dataset import build_tasks

import ecg_adapter
from spicefault.experiments import run_chunks

pytestmark = [pytest.mark.ecgfd, pytest.mark.ngspice, pytest.mark.baseline_data]

DATA = Path(os.environ.get("ECGFD_DATA", REPO_ROOT / "data" / "v1"))
PER_KIND = 3
RTOL, ATOL = 1e-9, 1e-15


@pytest.fixture(scope="module")
def stored(cfg):
    folder = DATA / cfg["circuit"]
    if not (folder / "samples.parquet").exists():
        pytest.skip(f"no baseline dataset in {folder}")
    manifest = json.loads((folder / "manifest.json").read_text())
    df = pd.read_parquet(folder / "samples.parquet")
    waveforms = np.load(folder / "waveforms.npy", mmap_mode="r")
    return manifest, df, waveforms


def test_regenerated_cases_match_the_dataset(stored, tmp_path):
    manifest, df, waveforms = stored
    cfg = manifest["config"]
    tasks = build_tasks(cfg)
    assert len(tasks) == len(df) == manifest["n_samples"]

    # a few cases of every fault kind, healthy included
    picked = df.groupby("kind", sort=True).sample(PER_KIND, random_state=0)
    ids = sorted(picked["sample_id"])
    stems = run_chunks(
        [tasks[i] for i in ids], ecg_adapter.simulate_task, cfg, tmp_path, waveforms.shape[1],
        workers=4, progress=False,
    )
    ours = pd.read_parquet(stems[0].with_suffix(".parquet")).drop(columns="sample_id")
    theirs = df.set_index("sample_id").loc[ids].reset_index(drop=True)
    assert list(ours.columns) == list(theirs.columns)

    numeric = [c for c in ours if c.startswith(("dc_", "acd_", "acc_", "zlo_", "spec_"))]
    exact = [c for c in ours if c not in numeric]
    assert any(c.startswith("p_") for c in exact) and "compliant" in exact
    pd.testing.assert_frame_equal(ours[exact], theirs[exact], check_exact=True)
    pd.testing.assert_frame_equal(
        ours[numeric], theirs[numeric], check_exact=False, rtol=RTOL, atol=ATOL
    )
    assert np.array_equal(np.load(stems[0].with_suffix(".npy")), waveforms[ids])

    # reported for the record: the tolerance is not expected to be used on the same platform
    worst = float(np.nanmax(np.abs(ours[numeric].to_numpy() - theirs[numeric].to_numpy())))
    print(f"\n{cfg['circuit']}: {len(ids)} cases, largest difference {worst}")
