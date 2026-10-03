"""The experiment scripts run from campaign to result. Sizes are tiny: the figures mean nothing."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repository root

from validation.experiments import common, e_variability, f_conditions, g_separability  # noqa: E402

pytestmark = pytest.mark.ngspice

SCALES = (0.0, 1.0, 2.0)


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    root = tmp_path_factory.mktemp("experiments")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(common, "RESULTS", root / "results")
        yield root / "data", root / "results"


@pytest.fixture(scope="module")
def variability(workspace):
    data, _ = workspace
    e_variability.run("sallen_key", data, SCALES, samples_per_fault=4, healthy=120, workers=4)
    return e_variability.analyse("sallen_key", data, SCALES, alpha=0.05)


def test_experiment_e(variability, workspace):
    _, results = workspace
    result = variability
    assert result["n_faults"] == 58 and set(result["per_scale"]) == {"0", "1", "2"}
    exact, declared, wide = (result["per_scale"][k] for k in ("0", "1", "2"))
    # without tolerance every healthy circuit is the nominal one, and it complies
    assert exact["yield"] == 1.0 and exact["failed_simulations"] == 0
    assert exact["mean_detection_probability"] >= declared["mean_detection_probability"]
    assert declared["mean_detection_probability"] >= wide["mean_detection_probability"]
    assert exact["faults_detected"] >= wide["faults_detected"]
    assert wide["yield"] <= declared["yield"] <= 1.0
    assert 0.0 <= declared["diagnostic_coverage"] <= 1.0
    saved = json.loads((results / "e_variability" / "sallen_key.json").read_text())
    assert saved["scales"] == list(SCALES) and saved["declared_tolerances"][0]["tolerance"] == 0.05
    for table in ("detection", "standardised_shift", "failure_probability"):
        assert (results / "e_variability" / f"sallen_key_{table}.csv").exists()


def test_experiment_g(variability, workspace):
    data, results = workspace
    result = g_separability.analyse("sallen_key", data)
    measured, predicted = result["from_measurements"], result["predicted_by_local_sensitivity"]
    assert measured["n_features"] == 10 and result["from_waveform"]["n_features"] == 10
    assert measured["n_conditions"] == 58
    visible = measured["among_detectable_conditions"]
    hidden = len(measured["conditions_not_separated_from_healthy"])
    assert visible["n_conditions"] == 58 - hidden
    components = {"R1", "R2", "R3", "R4", "R5", "C1", "C2", "XU1"}
    assert set(visible["confusable_components"]) <= components
    assert set(visible["confusable_components"]) | set(
        measured["components_with_no_detectable_fault"]
    ) == components
    assert predicted["n_components"] == 7 and 1 <= predicted["testability_rank"] <= 7
    # the gain resistors only act through their ratio: local sensitivity must say so
    assert any({"R4", "R5"} <= set(group) for group in predicted["collinear_groups"])
    assert (results / "g_separability" / "sallen_key_sensitivity.csv").exists()


def test_experiment_f(workspace):
    data, results = workspace
    f_conditions.run(data, "one_factor", samples_per_fault=2, healthy=40, workers=4)
    result = f_conditions.analyse(data, "one_factor", alpha=0.05)
    assert result["n_faults"] == 54 and result["reference_condition"] == "nominal"
    assert list(result["per_condition"]) == [
        "nominal", "low_line", "high_line", "heavy_load", "light_load", "cold", "hot",
    ]
    assert all(0.0 <= c["yield"] <= 1.0 for c in result["per_condition"].values())
    assert 0.0 <= result["largest_range_of_detection_probability"] <= 1.0
    assert (results / "f_conditions" / "regulator_one_factor_detection.csv").exists()
