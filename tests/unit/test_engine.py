import json

import numpy as np
import pandas as pd
import pytest
from engine_workers import draw, scalar_only

from spicefault.dataset import load_dataset
from spicefault.experiments import run_campaign, run_chunks

TASKS = [(index, replica) for index in range(5) for replica in range(20)]
CONFIG = {"seed": 42, "n_points": 8}


def read(stems):
    df = pd.concat([pd.read_parquet(s.with_suffix(".parquet")) for s in stems], ignore_index=True)
    return df, np.concatenate([np.load(s.with_suffix(".npy")) for s in stems])


def test_results_do_not_depend_on_workers_or_chunking(tmp_path):
    kwargs = {"n_points": 8, "progress": False}
    df1, w1 = read(run_chunks(TASKS, draw, CONFIG, tmp_path / "a", workers=1, chunk=100, **kwargs))
    df4, w4 = read(run_chunks(TASKS, draw, CONFIG, tmp_path / "b", workers=4, chunk=7, **kwargs))
    pd.testing.assert_frame_equal(df1, df4)
    assert np.array_equal(w1, w4, equal_nan=True)
    assert list(df1["sample_id"]) == list(range(len(TASKS)))
    assert w1.dtype == np.float32
    assert np.isnan(w1[~df1["sim_ok"].to_numpy()]).all()
    assert not np.isnan(w1[df1["sim_ok"].to_numpy()]).any()


def test_resumed_run_repeats_only_incomplete_chunks(tmp_path):
    log = tmp_path / "log"
    log.mkdir()
    context = {**CONFIG, "log": str(log)}
    kwargs = {"n_points": 8, "workers": 2, "chunk": 25, "progress": False}
    reference = read(run_chunks(TASKS, draw, CONFIG, tmp_path / "ref", **kwargs))

    stems = run_chunks(TASKS, draw, context, tmp_path / "parts", **kwargs)
    assert len(list(log.iterdir())) == len(TASKS)
    # an interruption leaves a chunk without its table, possibly with its waveforms
    stems[2].with_suffix(".parquet").unlink()
    for file in log.iterdir():
        file.unlink()
    df, waveforms = read(run_chunks(TASKS, draw, context, tmp_path / "parts", **kwargs))

    assert len(list(log.iterdir())) == 25  # only the third chunk was simulated again
    pd.testing.assert_frame_equal(df, reference[0])
    assert np.array_equal(waveforms, reference[1], equal_nan=True)


def test_campaign_writes_a_dataset_with_its_provenance(tmp_path):
    out = run_campaign(
        TASKS, draw, CONFIG, tmp_path / "data", config=CONFIG, n_points=8, workers=2, chunk=30,
        progress=False, summary=lambda df: {"n_failed": int((~df["sim_ok"]).sum())},
    )
    assert sorted(p.name for p in out.iterdir()) == [
        "manifest.json", "samples.parquet", "waveforms.npy"
    ]
    df, waveforms, manifest = load_dataset(out, drop_failed=False)
    assert len(df) == len(waveforms) == manifest["n_samples"] == len(TASKS)
    assert manifest["n_failed"] == 10 and manifest["config"] == CONFIG
    assert manifest["workers"] == 2 and manifest["chunk"] == 30 and manifest["resumed"] is False
    assert {"spicefault_version", "simulator", "python", "numpy", "platform"} <= set(manifest)
    kept, kept_waveforms, _ = load_dataset(out)
    assert len(kept) == len(kept_waveforms) == len(TASKS) - 10


def test_campaign_resumes_and_refuses_another_configuration(tmp_path):
    out = tmp_path / "data"
    run_chunks(TASKS[:30], draw, CONFIG, out / "parts", n_points=8, chunk=30, progress=False)
    with pytest.raises(ValueError, match="another configuration"):
        # the leftover chunk has no record of the configuration that produced it
        (out / "parts" / "config.sha256").write_text("something else")
        run_campaign(TASKS, draw, CONFIG, out, config=CONFIG, n_points=8, chunk=30, progress=False)
    (out / "parts" / "config.sha256").unlink()
    run_campaign(TASKS, draw, CONFIG, out, config=CONFIG, n_points=8, chunk=30, progress=False)
    assert json.loads((out / "manifest.json").read_text())["resumed"] is True
    assert not (out / "parts").exists()


def test_campaign_without_waveforms(tmp_path):
    out = run_campaign([1, 2, 3], scalar_only, 2.0, tmp_path / "data", config={}, progress=False)
    df, waveforms, _ = load_dataset(out)
    assert waveforms is None and list(df["x"]) == [2.0, 4.0, 6.0]
