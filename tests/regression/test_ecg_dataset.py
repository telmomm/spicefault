"""A campaign over the whole ECG fault catalogue, written as a dataset by spicefault.

One sample per fault, in service. The dataset is checked against the published
`data/v1`, where the same samples are the first replica of each fault, and against
itself: integrity, traceability and reproduction from the folder.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ecgfd.config import REPO_ROOT
from ecgfd.dataset import sample_rng
from ecgfd.faults import fault_catalogue
from ecgfd.sampling import sample_instance

from ecg_adapter import service_experiment
from spicefault import FaultCampaign

pytestmark = [pytest.mark.ecgfd, pytest.mark.ngspice, pytest.mark.baseline_data]

DATA = Path(os.environ.get("ECGFD_DATA", REPO_ROOT / "data" / "v1"))
N_HEALTHY = 3


@pytest.fixture(scope="module")
def study(cfg, tmp_path_factory):
    folder = DATA / cfg["circuit"]
    if not (folder / "samples.parquet").exists():
        pytest.skip(f"no baseline dataset in {folder}")
    config = json.loads((folder / "manifest.json").read_text())["config"]
    reduced = {**config, "dataset": {"n_healthy": N_HEALTHY, "n_per_fault": 1}}
    experiment = service_experiment(reduced)
    campaign = FaultCampaign.from_experiment(experiment, tmp_path_factory.mktemp("ecg") / "data")
    campaign.run(workers=6, progress=False)
    return config, experiment, campaign.dataset(), folder


def test_whole_catalogue_matches_the_published_dataset(study):
    config, experiment, dataset, folder = study
    n_per_fault, n_healthy = config["dataset"]["n_per_fault"], config["dataset"]["n_healthy"]
    catalogue = fault_catalogue(config)
    assert len(dataset) == N_HEALTHY + len(catalogue) and dataset.ok.all()
    assert dataset.verify() == []

    # replica 0 of fault k is row n_healthy + (k - 1) * n_per_fault of data/v1
    ids = list(range(N_HEALTHY)) + [n_healthy + k * n_per_fault for k in range(len(catalogue))]
    baseline = pd.read_parquet(folder / "samples.parquet").iloc[ids].reset_index(drop=True)
    assert list(dataset.samples["fault_id"]) == list(baseline["condition"])
    # the value injected by an `ina_cmrr` fault depends on the drawn sign (built per sample)
    comparable = (baseline["kind"] != "ina_cmrr").to_numpy()
    ours = dataset.samples.loc[comparable, dataset.features].reset_index(drop=True)
    theirs = baseline.loc[comparable, dataset.features].reset_index(drop=True)
    pd.testing.assert_frame_equal(ours, theirs, check_exact=False, rtol=1e-9, atol=1e-15)
    waveforms = np.load(folder / "waveforms.npy", mmap_mode="r")
    assert np.array_equal(dataset.waveforms[comparable], waveforms[ids][comparable])
    assert comparable.sum() >= len(ids) - 2
    assert list(dataset.samples["electrode_kind"]) == list(baseline["electrode_kind"])


def test_dataset_is_traceable_and_reproducible(study):
    config, experiment, dataset, _ = study
    assert dataset.labels == ["p_U1_cmrr_db", "p_U1_cmrr_sign", "electrode_type", "electrode_kind"][
        -len(dataset.labels) :
    ]
    # joint variations hold functions: the experiment is passed to reproduce the samples
    with pytest.raises(NotImplementedError, match="holds a function"):
        dataset.experiment()
    report = dataset.reproduce(n=15, experiment=experiment)
    assert report[["definition", "parameters", "status"]].all().all()
    assert report["max_abs_diff"].max() == 0.0 and report["waveform_abs_diff"].max() == 0.0

    # the netlist of a sample is rebuilt from the stored values, without the functions
    for sample_id in (0, 5, 40, len(dataset) - 1):
        sample = experiment.plan()[sample_id]
        assert dataset.netlist(sample_id) == experiment.realise(sample).netlist
    provenance = dataset.provenance(40)
    fault = fault_catalogue(config)[40 - N_HEALTHY]
    assert (provenance.fault_id, provenance.seed) == (fault.id, config["seed"])
    assert provenance.seed_key == f"{40 - N_HEALTHY + 1}/0"
    drawn = sample_instance(config, sample_rng(config, 40 - N_HEALTHY + 1, 0))
    assert provenance.parameters["p_R1_value"] == drawn.passives["R1"]


def test_dataset_feeds_analysis_and_models(study):
    _, _, dataset, _ = study
    x, y = dataset.to_ml(target="fault_location")
    assert x.shape == (len(dataset), len(dataset.features)) and len(set(y)) > 20
    waveforms, kinds = dataset.to_ml(target="fault_type", waveforms=True)
    assert waveforms.shape == (len(dataset), 1000) and "open" in set(kinds)
    shift = dataset.analysis(healthy_split=None).standardised_shift()
    assert len(shift) == len(dataset) - N_HEALTHY
