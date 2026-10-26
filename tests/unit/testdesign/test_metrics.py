import math

import numpy as np
import pandas as pd
import pytest

from spicefault.testdesign import metrics as mt


def synthetic(n_healthy=400, n_fault=60, seed=0):
    """Two tests. Fault `far` moves test a by 10 sigma, `twin` moves it by -10 sigma,
    `hidden` changes nothing, and `other` moves test b by 10 sigma."""
    rng = np.random.default_rng(seed)
    rows = []
    shifts = {
        "healthy": (0, 0),
        "far": (10, 0),
        "twin": (-10, 0),
        "hidden": (0, 0),
        "other": (0, 10),
    }
    for fault, (da, db) in shifts.items():
        n = n_healthy if fault == "healthy" else n_fault
        for replica in range(n):
            rows.append(
                {
                    "fault_id": fault,
                    "replica": replica,
                    "a": rng.normal() + da,
                    "b": rng.normal() + db,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def matrix():
    return synthetic()


def test_split_is_by_replica_and_disjoint(matrix):
    cal, ev = mt.split_healthy(matrix, 0.5)
    assert len(cal) == len(ev) == 200
    assert set(cal).isdisjoint(ev)
    assert matrix.loc[cal, "replica"].max() < matrix.loc[ev, "replica"].min()


def test_detection_and_false_alarm(matrix):
    d = mt.detect(matrix, ["a", "b"], mt.GaussianLimits(0.01))
    p = d.per_fault.p
    assert p["far"] == p["twin"] == p["other"] == 1.0
    assert p["hidden"] < 0.1
    assert d.false_alarm["n"] == 200 and d.false_alarm["p"] < 0.05
    row = d.per_fault.loc["far"]
    assert row.ci_low == pytest.approx((0.025) ** (1 / 60), rel=1e-6)  # exact CP bound with x = n
    only_a = mt.detect(matrix, ["a"], mt.GaussianLimits(0.01))
    assert only_a.per_fault.p["other"] < 0.1


def test_gaussian_limits_hit_the_target_false_alarm_rate():
    rng = np.random.default_rng(1)
    healthy, fresh = rng.normal(size=(2000, 4)), rng.normal(size=(200000, 4))
    rate = mt.GaussianLimits(0.01, cover_sample=False).fit(healthy).flag(fresh).mean()
    assert 0.006 < rate < 0.015
    # with cover_sample the limits never fall inside the calibration range
    heavy = rng.standard_t(2, size=(200, 3))
    limits = mt.GaussianLimits(0.01).fit(heavy)
    assert (limits.low <= heavy.min(axis=0)).all() and (limits.high >= heavy.max(axis=0)).all()
    assert not limits.flag(heavy).any()


def test_phase_is_read_around_its_fault_free_direction():
    rng = np.random.default_rng(2)

    def angle(x):
        return (x + 180.0) % 360.0 - 180.0

    rows = [
        {"fault_id": "healthy", "replica": r, "ac|out|phase@1": angle(180 + rng.normal(0, 5))}
        for r in range(400)
    ]
    rows += [
        {"fault_id": "shift", "replica": r, "ac|out|phase@1": angle(120 + rng.normal(0, 5))}
        for r in range(50)
    ]
    m = pd.DataFrame(rows)
    d = mt.detect(m, ["ac|out|phase@1"], mt.GaussianLimits(0.01))
    assert d.false_alarm["p"] < 0.03  # a naive reading of +-180 would flag about half of them
    assert d.per_fault.p["shift"] == 1.0
    assert abs(abs(d.angles["ac|out|phase@1"]) - 180) < 2


def test_metric_values_by_hand():
    faults = mt.fault_table(
        ["s1", "s2", "m1", "safe"],
        rate=pd.Series({"s1": 2.0, "s2": 1.0, "m1": 1.0, "safe": 4.0}),
        iec=pd.Series({"s1": "dangerous", "s2": "dangerous", "m1": "dangerous", "safe": "safe"}),
        iso=pd.Series(
            {"s1": "single_point", "s2": "single_point", "m1": "multiple_point", "safe": "safe"}
        ),
    )
    p = pd.Series({"s1": 1.0, "s2": 0.5, "m1": 0.0, "safe": 0.0})
    v = mt.metric_values(p, faults)
    assert v["dc"] == pytest.approx((2 + 0.5) / 4)
    assert v["sff"] == pytest.approx((4 + 2.5) / 8)
    assert v["spfm"] == pytest.approx(1 - 0.5 / 8)
    assert v["lfm"] == pytest.approx(1 - 1.0 / (8 - 0.5))
    assert v["dc_w"] == pytest.approx(2.5 / 8)
    assert math.isnan(
        mt.metric_values(p[["s1"]], faults[["rate", "severity", "iec", "iso"]].loc[["s1"]])["lfm"]
    )
    with pytest.raises(ValueError):
        mt.fault_table(["x"], iso="latent")


def test_bounds_bracket_the_estimate_and_hold_simultaneously(matrix):
    d = mt.detect(matrix, ["a", "b"], mt.GaussianLimits(0.01))
    faults = mt.fault_table(d.per_fault.index)
    for method in ("bonferroni", "jeffreys"):
        c = mt.coverage(d, faults, 0.95, method)
        for metric in ("dc_w", "dc", "spfm"):
            assert c.loc[metric, "lower"] <= c.loc[metric, "estimate"] <= c.loc[metric, "upper"]
    b = mt.coverage(d, faults, 0.95, "bonferroni")
    j = mt.coverage(d, faults, 0.95, "jeffreys")
    assert b.loc["dc", "lower"] <= j.loc["dc", "lower"]  # Bonferroni is the conservative one
    # a fully detected fault: its simultaneous lower bound is (0.05 / K) ** (1 / n)
    full = d.per_fault.loc[["far"]]
    one = mt.Detection(d.tests, full, d.false_alarm, d.alarms, d.detector, 0.95)
    assert mt.coverage(one, faults, 0.95).loc["dc", "lower"] == pytest.approx(
        0.05 ** (1 / 60), rel=1e-6
    )


def test_targets():
    assert mt.asil_reached(0.995, 0.95) == "D"
    assert mt.asil_reached(0.98, 0.70) == "B"
    assert mt.asil_reached(0.98) == "C"
    assert mt.asil_reached(0.85) == "QM-A"
    assert mt.sil_reached(0.95, hft=0, element_type="B") == 2
    assert mt.sil_reached(0.50, hft=0, element_type="B") == 0
    assert mt.sil_reached(0.995, hft=1, element_type="A") == 4


def test_localisation_and_ambiguity(matrix):
    d = mt.detect(matrix, ["a", "b"], mt.GaussianLimits(0.01))
    loc = mt.localisation(matrix, d)
    assert loc["h_prior"] == pytest.approx(2.0)  # four equally likely faults
    assert (
        loc["h_given_tests"] < 0.2
    )  # far, twin and other are told apart; hidden is the fault-free signature
    sets = mt.ambiguity_sets(matrix, d)
    assert sets[0] == ["hidden"]
    only_a = mt.detect(matrix, ["a"], mt.GaussianLimits(0.01))
    sets_a = mt.ambiguity_sets(matrix, only_a)
    assert sets_a[0] == ["hidden", "other"]
    assert mt.localisation(matrix, only_a)["h_given_tests"] == pytest.approx(0.5, abs=0.1)
