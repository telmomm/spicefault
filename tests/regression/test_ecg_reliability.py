"""The reliability metrics against the testability code of the ECG study, on `data/v1`.

`ecgfd.ambiguity` and `ecgfd.evaluation` are the reference implementations these
metrics were generalised from. The dataset location comes from ECGFD_DATA.
"""

import json
import os
from pathlib import Path

import ecgfd.ambiguity as reference
import numpy as np
import pandas as pd
import pytest
from ecgfd.config import REPO_ROOT
from ecgfd.evaluation import escape_rate, false_reject_rate

from ecg_adapter import service_experiment
from spicefault.reliability import (
    ReliabilityAnalysis,
    collinear_groups,
    local_sensitivity,
    normalised_sensitivity,
)
from spicefault.reliability import testability_rank as rank

pytestmark = [pytest.mark.ecgfd, pytest.mark.baseline_data]

DATA = Path(os.environ.get("ECGFD_DATA", REPO_ROOT / "data" / "v1"))
ALPHA = 1.0 - 0.99  # the coverage of the baseline, written the way it computes the tail


@pytest.fixture(scope="module")
def study(cfg):
    folder = DATA / cfg["circuit"]
    if not (folder / "samples.parquet").exists():
        pytest.skip(f"no baseline dataset in {folder}")
    config = json.loads((folder / "manifest.json").read_text())["config"]
    df = pd.read_parquet(folder / "samples.parquet")
    df = df[df["sim_ok"]].reset_index(drop=True)
    features = [c for c in df.columns if c.startswith(("dc_", "acd_", "acc_", "zlo_"))]
    analysis = ReliabilityAnalysis(
        df, features, fault="condition", compliant="compliant", healthy_split=None
    )
    return config, df, features, analysis


def test_fault_dictionary_and_pairwise_separation(study):
    _, df, features, analysis = study
    centre, spread = reference.fault_dictionary(df, features)
    ours_centre, ours_spread = analysis.dictionary()
    pd.testing.assert_frame_equal(ours_centre, centre, check_exact=True)
    pd.testing.assert_frame_equal(ours_spread, spread, check_exact=True)

    d = reference.separability(centre, spread)
    pd.testing.assert_frame_equal(analysis.separation(), d, check_exact=True)
    shift = analysis.standardised_shift()["d_prime"]
    assert np.allclose(shift, d.loc["healthy", shift.index], rtol=1e-12, equal_nan=True)


def test_ambiguity_structure(study):
    _, df, features, analysis = study
    d = reference.separability(*reference.fault_dictionary(df, features))
    component_of = dict(zip(df["condition"], df["target"], strict=True))
    faults_only = d.drop(index="healthy", columns="healthy")
    del component_of["healthy"]

    ours = analysis.ambiguity(threshold=3.0, component_of=component_of)
    assert ours.groups == reference.ambiguity_groups(d, 3.0)
    assert ours.undetectable == list(d.index[(d.loc["healthy"] < 3.0) & (d.index != "healthy")])
    assert ours.confusable == reference.confusable_components(faults_only, component_of, 3.0)
    assert ours.component_groups == reference.component_groups(faults_only, component_of, 3.0)
    assert len(ours.groups) > 1 and any(ours.confusable.values())


def test_limit_test_detection_escape_and_false_reject(study):
    _, df, features, analysis = study
    rate, false_alarm = reference.envelope_detection(df, features, 0.99)
    ours = analysis.detectability(alpha=ALPHA)
    assert not ours.held_out  # as the baseline: thresholds and false alarms on the same samples
    assert ours.false_alarm == pytest.approx(false_alarm, abs=1e-15)
    assert np.allclose(ours.table["p_detect"], rate.reindex(ours.table.index), rtol=0, atol=1e-15)

    flagged = analysis.flags(ALPHA)
    if hasattr(reference, "envelope_flags"):  # later versions expose the decision per case
        limits = reference.envelope_limits(df[df["kind"] == "healthy"], features, 0.99)
        assert np.array_equal(flagged, reference.envelope_flags(df, limits))
    bad = ~df["compliant"].to_numpy(dtype=bool)
    faulty = (df["kind"] != "healthy").to_numpy()
    coverage = analysis.diagnostic_coverage(alpha=ALPHA, n_boot=20)
    # every fault has the same number of samples, so equal weights pool the samples
    assert coverage["escape_rate"] == pytest.approx(escape_rate(bad[faulty], flagged[faulty]))
    assert coverage["false_reject_rate"] == pytest.approx(false_reject_rate(bad, flagged))
    assert coverage["interval"][0] <= coverage["diagnostic_coverage"] <= coverage["interval"][1]

    probability = analysis.failure_probability()["p_failure"]
    expected = bad.astype(float)
    assert np.allclose(probability, pd.Series(expected).groupby(df["condition"], sort=False).mean())


@pytest.mark.ngspice
def test_local_sensitivity_and_what_it_bounds(study):
    config, df, features, analysis = study
    baseline = reference.sensitivity_matrix(config)  # every passive, and the pulse features too
    experiment = service_experiment(config)
    passives = list(baseline.columns)[:6]
    ours = local_sensitivity(
        experiment.circuit, experiment.measurements, experiment.config,
        parameters=[(name, "value") for name in passives],
    )
    assert list(ours.index) == features
    pd.testing.assert_frame_equal(ours, baseline.loc[features, passives], check_exact=True)

    spread = analysis.dictionary()[1].loc["healthy"]
    z = normalised_sensitivity(baseline.loc[features], spread)
    theirs = reference.normalised_sensitivity(baseline.loc[features], spread)
    pd.testing.assert_frame_equal(z, theirs, check_exact=True)
    assert rank(z) == reference.testability_rank(z) and rank(z) >= 2
    assert collinear_groups(z) == reference.collinear_groups(z)
