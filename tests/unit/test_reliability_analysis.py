import json
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from spicefault import Circuit, Measurement, SimulationConfig
from spicefault.faults import ParametricFault, ShortCircuit
from spicefault.reliability import (
    LimitTest,
    ReliabilityAnalysis,
    detectability_across,
    local_sensitivity,
    robustness,
)

PHI, QUANTILE = NormalDist().cdf, NormalDist().inv_cdf
N_HEALTHY, N_FAULT = 40000, 4000
SHIFTS = {"R1:parametric:+0.5": 1.0, "R1:parametric:+1": 2.0, "R1:parametric:+2": 4.0}


def population(shifts=SHIFTS, spread=1.0, seed=0, n_fault=N_FAULT) -> pd.DataFrame:
    """Healthy: x ~ N(0, spread), y ~ N(0, 1). Each fault shifts x by `shift`, in sigmas of 1."""
    rng = np.random.default_rng(seed)
    parts = [pd.DataFrame({"fault_id": "healthy", "x": rng.normal(0, spread, N_HEALTHY)})]
    for fault_id, shift in shifts.items():
        parts.append(pd.DataFrame({"fault_id": fault_id, "x": rng.normal(shift, spread, n_fault)}))
    df = pd.concat(parts, ignore_index=True)
    df["y"] = rng.normal(0, 1, len(df))
    df["sim_ok"] = True
    return df


RECORDS = [ParametricFault("R1", deviation=d).metadata() for d in (0.5, 1, 2)]


def analysis(df=None, features=("x",), **kwargs) -> ReliabilityAnalysis:
    return ReliabilityAnalysis(population() if df is None else df, list(features), **kwargs)


# --- M1 ---------------------------------------------------------------------------------


def test_detection_probability_matches_the_normal_model():
    result = analysis().detectability(alpha=0.01)
    assert result.held_out and result.n_calibration == result.n_evaluation == N_HEALTHY // 2
    assert result.false_alarm == pytest.approx(0.01, abs=0.003)
    assert result.false_alarm_interval[0] < result.false_alarm < result.false_alarm_interval[1]
    z = QUANTILE(1 - 0.005)
    for fault_id, shift in SHIFTS.items():
        row = result.table.loc[fault_id]
        expected = PHI(shift - z) + PHI(-shift - z)
        assert row["p_detect"] == pytest.approx(expected, abs=0.03)
        assert row["ci_low"] < row["p_detect"] < row["ci_high"]
        assert row["ci_high"] - row["ci_low"] < 0.04 and row["n_ok"] == N_FAULT
        assert row["bound_low"] == row["bound_high"] == row["p_detect"]  # nothing failed


def test_bonferroni_keeps_the_false_alarm_rate_with_several_features():
    two = analysis(features=("x", "y")).detectability(alpha=0.01)
    assert two.false_alarm == pytest.approx(0.01, abs=0.003)
    # the irrelevant feature costs some detection: its share of alpha is wasted
    one = analysis().detectability(alpha=0.01)
    assert (two.table["p_detect"] <= one.table["p_detect"] + 0.01).all()


def test_in_sample_false_alarm_is_optimistic():
    """Limits set on few healthy samples sit near their extremes. Judged on those same
    samples the false-alarm rate looks close to alpha; on new ones it is higher.
    """
    in_sample, held_out = [], []
    for seed in range(150):
        df = population(seed=seed).groupby("fault_id").head(120)
        fit_on_60 = df.drop(df.index[60:120])  # 60 healthy samples, used for both purposes
        in_sample.append(
            analysis(fit_on_60, ("x", "y"), healthy_split=None).detectability(0.05).false_alarm
        )
        split = analysis(df, ("x", "y"), seed=seed).detectability(0.05)  # 60 and 60
        assert split.held_out and (split.n_calibration, split.n_evaluation) == (60, 60)
        held_out.append(split.false_alarm)
    assert np.mean(in_sample) == pytest.approx(0.05, abs=0.02)
    assert np.mean(held_out) > np.mean(in_sample) + 0.02


def test_failed_simulations_give_bounds():
    df = population(n_fault=100)
    failing = df.index[df["fault_id"] == "R1:parametric:+1"][:40]
    df.loc[failing, "sim_ok"] = False
    df.loc[failing, ["x", "y"]] = np.nan
    a = analysis(df)
    assert a.counts().loc["R1:parametric:+1"].to_dict() == {"n": 100, "n_ok": 60, "n_failed": 40}
    row = a.detectability().table.loc["R1:parametric:+1"]
    assert (row["n"], row["n_ok"]) == (100, 60)
    assert row["bound_low"] == row["detected"] / 100
    assert row["bound_high"] == (row["detected"] + 40) / 100
    assert row["bound_low"] < row["p_detect"] < row["bound_high"]
    response = a.response("R1:parametric:+1")
    assert (response.n_total, response.n_ok, response.n_failed) == (100, 60, 40)
    with pytest.raises(ValueError, match="not finite"):
        analysis(df.assign(sim_ok=True))


def test_a_custom_detector_can_be_passed():
    class OneSided:
        def fit(self, healthy):
            self.limit = np.quantile(healthy[:, 0], 0.99)
            return self

        def flag(self, x):
            return x[:, 0] > self.limit

    one_sided = analysis().detectability(detector=OneSided())
    two_sided = analysis().detectability(alpha=0.01)
    assert one_sided.false_alarm == pytest.approx(0.01, abs=0.003)
    # all faults shift upwards: the one-sided rule detects more at the same false-alarm rate
    assert (one_sided.table["p_detect"] > two_sided.table["p_detect"]).all()
    assert isinstance(LimitTest(0.01).fit(np.zeros((10, 2))), LimitTest)


# --- M2, M3 -----------------------------------------------------------------------------


def test_standardised_shift_and_auc_match_the_normal_model():
    a = analysis(features=("x", "y"))
    shift = a.standardised_shift()
    assert shift["d_prime"].tolist() == pytest.approx([1.0, 2.0, 4.0], rel=0.05)
    assert set(shift["feature"]) == {"x"}
    per_feature = a.standardised_shift(per_feature=True)
    assert per_feature["y"].max() < 0.1 and list(per_feature.columns) == ["x", "y"]
    assert np.allclose(shift["d_prime"], a.separation().loc["healthy", shift.index])

    table = a.auc()
    expected = [PHI(s / 2**0.5) for s in SHIFTS.values()]
    assert table["auc"].tolist() == pytest.approx(expected, abs=0.01)
    assert set(table["feature"]) == {"x"} and set(table["n_features"]) == {2}


def test_noise_floor_bounds_the_spread():
    exact = population(spread=0.0)  # no variation and no noise: degenerate clouds
    assert np.isinf(analysis(exact).standardised_shift()["d_prime"]).all() is np.False_
    floored = analysis(exact, noise_floor={"x": 0.5}).standardised_shift()["d_prime"]
    assert floored.tolist() == pytest.approx([2.0, 4.0, 8.0])


# --- M5, M6 -----------------------------------------------------------------------------


def labelled() -> pd.DataFrame:
    """A small table with known counts: x far above 0 is detected, `compliant` is given."""
    rows = [("healthy", 0.0, True)] * 60 + [("healthy", 0.1, True)] * 39 + [("healthy", 0.2, False)]
    # A: 10 samples, 8 fail, 6 of the failing ones are detected
    rows += [("A", 9.0, False)] * 6 + [("A", 0.05, False)] * 2 + [("A", 9.0, True)] * 2
    # B: 10 samples, 2 fail, none of them detected
    rows += [("B", 0.05, False)] * 2 + [("B", 0.05, True)] * 8
    # C: never fails
    rows += [("C", 9.0, True)] * 10
    return pd.DataFrame(rows, columns=["fault_id", "x", "compliant"])


def test_failure_probability_and_yield():
    a = analysis(labelled(), compliant="compliant", healthy_split=None)
    table = a.failure_probability()
    assert table["p_failure"].to_dict() == {"healthy": 0.01, "A": 0.8, "B": 0.2, "C": 0.0}
    assert table["failures"].tolist() == [1, 8, 2, 0]
    assert table.loc["C", "ci_low"] == 0.0 and 0.25 < table.loc["C", "ci_high"] < 0.31
    with pytest.raises(ValueError, match="needs the `compliant` column"):
        analysis(labelled(), healthy_split=None).failure_probability()


def test_diagnostic_coverage_escape_and_false_reject():
    a = analysis(labelled(), compliant="compliant", healthy_split=None)
    result = a.diagnostic_coverage(alpha=0.1)
    # detected and failing: 6 of 10 in A, none in B and C; failing: 8, 2 and 0 of 10
    assert result["diagnostic_coverage"] == pytest.approx(0.6 / (0.8 + 0.2 + 0.0))
    assert result["escape_rate"] == pytest.approx(0.4)
    assert result["interval"][0] < 0.6 < result["interval"][1]
    assert result["n_failed_samples"] == 10 and result["weights"] == "equal"
    # compliant: 99 healthy (none of them flagged; the flagged one does not comply) and
    # 2 + 8 + 10 fault samples, of which the 2 of A and the 10 of C are flagged
    assert result["n_compliant_samples"] == 119 and result["n_compliant_healthy"] == 99
    assert result["false_reject_rate"] == pytest.approx(12 / 119)

    # if B occurs nine times as often as A, most failures are the undetected ones
    weighted = a.diagnostic_coverage(alpha=0.1, weights={"A": 1.0, "B": 9.0, "C": 1.0})
    assert weighted["diagnostic_coverage"] == pytest.approx(0.6 / (0.8 + 9 * 0.2))
    assert weighted["weights"] == "given"


# --- M7 ---------------------------------------------------------------------------------


def test_severity_response_and_minimum_detectable():
    a = analysis(faults=RECORDS)
    response = a.severity_response(alpha=0.01)
    assert response["magnitude"].tolist() == [0.5, 1.0, 2.0]
    assert response["severity"].tolist() == [0.5, 1.0, 2.0]
    assert set(response["components"]) == {"R1"}
    assert response["p_detect"].is_monotonic_increasing

    found = a.minimum_detectable(alpha=0.01, beta=0.1)
    assert len(found) == 1
    row = found.iloc[0]
    assert (row["fault_type"], row["components"], row["direction"]) == ("parametric", "R1", "+")
    assert row["minimum_detectable"] == 2.0 and row["monotone"] and row["grid"] == [0.5, 1.0, 2.0]
    # a stricter requirement than any simulated magnitude meets
    assert np.isnan(a.minimum_detectable(alpha=0.01, beta=1e-6)["minimum_detectable"].iloc[0])
    with pytest.raises(ValueError, match="needs the fault records"):
        analysis().severity_response()


@pytest.mark.ngspice
def test_local_sensitivity_of_a_divider():
    circuit = Circuit("divider\nV1 in 0 dc 1\nR1 in out 10k\nR2 out 0 30k\n.end\n")
    measurements = [
        Measurement.value("v(out)", name="vout"),
        Measurement.value("v(in)", name="vin"),
    ]
    s = local_sensitivity(circuit, measurements, SimulationConfig(("op",)))
    assert list(s.index) == ["vout", "vin"] and list(s.columns) == ["R1", "R2"]
    # vout = R2 / (R1 + R2): d vout / d ln R = -+ R1 R2 / (R1 + R2)^2, per 1 %
    expected = 10e3 * 30e3 / 40e3**2 * 0.01
    assert s.loc["vout"].tolist() == pytest.approx([-expected, expected], rel=1e-3)
    assert s.loc["vin"].abs().max() < 1e-9
    source = local_sensitivity(circuit, measurements, SimulationConfig(("op",)), [("V1", "dc")])
    assert source.loc["vout", "V1.dc"] == pytest.approx(0.75 * 0.01, rel=1e-6)


# --- M8, M11 ----------------------------------------------------------------------------


def test_robustness_against_the_tolerance_scale():
    """The same shifts, with a healthy spread that grows with the tolerance scale."""
    by_scale = {k: analysis(population(spread=k, seed=3)) for k in (2.0, 0.5, 1.0)}
    table = robustness(by_scale, alpha=0.01, beta=0.1)
    assert list(table.columns) == [0.5, 1.0, 2.0, "loss", "critical_tolerance"]
    assert (table[0.5] >= table[1.0]).all() and (table[1.0] >= table[2.0]).all()
    assert (table["loss"] > 0.05).all()
    z = QUANTILE(1 - 0.005)
    assert table.loc["R1:parametric:+1", 0.5] == pytest.approx(PHI(4 - z), abs=0.03)
    # shifts of 1, 2 and 4: detected with 90 % up to a spread of about shift / (z + 1.28)
    critical = table["critical_tolerance"]
    assert np.isnan(critical["R1:parametric:+0.5"])
    assert critical["R1:parametric:+1"] == 0.5 and critical["R1:parametric:+2"] == 1.0


def test_operating_condition_dependence():
    nominal, hot = population(seed=4), population(spread=2.0, seed=5)
    both = pd.concat([nominal.assign(condition="nominal"), hot.assign(condition="hot")])
    parts = analysis(both.reset_index(drop=True)).by("condition")
    assert list(parts) == ["nominal", "hot"]
    assert parts["hot"].counts()["n"].tolist() == [N_HEALTHY, N_FAULT, N_FAULT, N_FAULT]
    table = detectability_across(parts, alpha=0.01, beta=0.1)
    assert list(table.columns) == ["nominal", "hot", "worst_case", "range", "lost"]
    assert np.allclose(table["worst_case"], table["hot"])
    assert np.allclose(table["range"], table["nominal"] - table["hot"])
    # the largest fault is detected at the nominal condition and no longer when hot
    assert table["lost"].to_dict() == {
        "R1:parametric:+0.5": [], "R1:parametric:+1": [], "R1:parametric:+2": ["hot"],
    }


# --- M9 ---------------------------------------------------------------------------------


def test_ambiguity_of_the_analysis():
    shifts = {"R1:parametric:+0.5": 1.0, "R1:parametric:+2": 8.0, "R2:short": 8.5, "C1:short": 30.0}
    records = [
        ParametricFault("R1", deviation=0.5).metadata(),
        ParametricFault("R1", deviation=2).metadata(),
        ShortCircuit("R2").metadata(),
        ShortCircuit("C1").metadata(),
    ]
    a = analysis(population(shifts, seed=6), faults=records)
    result = a.ambiguity(threshold=3.0)
    assert result.undetectable == ["R1:parametric:+0.5"]
    assert result.groups == [
        ["healthy", "R1:parametric:+0.5"], ["R1:parametric:+2", "R2:short"], ["C1:short"],
    ]
    assert result.confusable == {"C1": [], "R1": ["R2"], "R2": ["R1"]}
    assert result.component_groups == [["R1", "R2"], ["C1"]]
    assert analysis(population(shifts, seed=6)).ambiguity().confusable is None
    stage = dict.fromkeys(shifts, "stage 1")
    assert a.ambiguity(component_of=stage).component_groups == [["stage 1"]]


# --- from a campaign --------------------------------------------------------------------


def test_analysis_of_a_campaign_dataset(tmp_path):
    from campaign_backend import Divider

    from spicefault import FaultCampaign, Simulator
    from spicefault.variation import tolerances

    circuit = Circuit("divider\nV1 in 0 dc 1\nR1 in out 10k\nR2 out 0 10k\n.end\n", "divider")
    faults = [ParametricFault("R1", deviation=d) for d in (0.005, 0.05, 0.2)]
    campaign = FaultCampaign(
        circuit, [*faults, ShortCircuit("R2")], out_dir=tmp_path / "data", samples_per_fault=60,
        healthy_samples=600, variations=tolerances(circuit, {"R": 0.005}),
        simulator=Simulator(Divider()), measurements=[Measurement.value("v(out)", name="vout")],
    )
    campaign.run(workers=2, progress=False)
    a = ReliabilityAnalysis.from_dataset(tmp_path / "data")
    assert a.features == ["vout"] and len(a.fault_ids) == 4
    table = a.detectability(alpha=0.02).table
    assert table.loc["R1:parametric:+0.2", "p_detect"] == 1.0
    assert table.loc["R1:parametric:+0.005", "p_detect"] < 0.6
    assert table.loc["R2:short", "p_detect"] == 1.0
    found = a.minimum_detectable(alpha=0.02).set_index("fault_type")
    assert found.loc["parametric", "minimum_detectable"] == 0.05
    assert json.dumps(a.response("R2:short").record)  # the record came with the dataset
