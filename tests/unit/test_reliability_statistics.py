from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from spicefault.reliability import (
    LimitTest,
    ambiguity_groups,
    auc,
    bootstrap_interval,
    clopper_pearson_interval,
    collinear_groups,
    component_groups,
    confusable_components,
    normalised_sensitivity,
    pairwise_shift,
    robust_spread,
    wilson_interval,
)
from spicefault.reliability import testability_rank as rank  # not a test of this module

PHI = NormalDist().cdf


def test_wilson_interval():
    low, high = wilson_interval(100, 200)
    assert (low, high) == pytest.approx((0.4313, 0.5687), abs=1e-4)  # half-width about 0.07
    assert wilson_interval(0, 200) == pytest.approx((0.0, 0.01885), abs=1e-5)
    assert wilson_interval(200, 200)[1] == 1.0
    assert all(np.isnan(wilson_interval(0, 0)))
    narrow, wide = wilson_interval(50, 100, 0.8), wilson_interval(50, 100, 0.99)
    assert wide[0] < narrow[0] < 0.5 < narrow[1] < wide[1]


def test_clopper_pearson_interval():
    # no miss in 200: the miss probability is below 1 - 0.025 ** (1 / 200)
    bound = 0.025 ** (1 / 200)
    assert clopper_pearson_interval(0, 200) == pytest.approx((0.0, 1 - bound), abs=1e-9)
    assert clopper_pearson_interval(200, 200) == pytest.approx((bound, 1.0), abs=1e-9)
    assert clopper_pearson_interval(5, 20) == pytest.approx((0.08657, 0.49105), abs=1e-5)
    # exact, hence never narrower than Wilson at the extremes
    assert clopper_pearson_interval(1, 50)[1] > wilson_interval(1, 50)[1]


def test_robust_spread_ignores_extreme_values():
    rng = np.random.default_rng(0)
    x = rng.normal(5.0, 2.0, size=(20000, 3))
    assert robust_spread(x) == pytest.approx([2.0, 2.0, 2.0], rel=0.03)
    x[:200] = 1e6  # a fault that saturates in 1 % of the cases
    assert robust_spread(x) == pytest.approx([2.0, 2.0, 2.0], rel=0.05)
    assert x.std(axis=0).min() > 1e4


def test_auc():
    assert auc(np.array([3.0, 4.0]), np.array([1.0, 2.0])) == 1.0
    assert auc(np.array([1.0, 2.0]), np.array([3.0, 4.0])) == 0.0
    assert auc(np.array([1.0, 2.0]), np.array([1.0, 2.0])) == 0.5  # ties count half
    assert auc(np.array([2.0, 2.0, 5.0]), np.array([2.0, 1.0])) == pytest.approx(5 / 6)
    rng = np.random.default_rng(1)
    healthy, fault = rng.normal(0, 1, 20000), rng.normal(1.0, 1, 20000)
    assert auc(fault, healthy) == pytest.approx(PHI(1 / 2**0.5), abs=0.005)


def test_bootstrap_interval_of_a_mean():
    data = np.random.default_rng(2).normal(10.0, 2.0, 400)
    low, high = bootstrap_interval(lambda rng: rng.choice(data, len(data)).mean(), 2000, seed=0)
    assert low < data.mean() < high
    assert high - low == pytest.approx(2 * 1.96 * 2.0 / 20, rel=0.15)
    assert bootstrap_interval(lambda rng: rng.choice(data, len(data)).mean(), 50, seed=3) == (
        bootstrap_interval(lambda rng: rng.choice(data, len(data)).mean(), 50, seed=3)
    )


def test_limit_test_keeps_the_false_alarm_rate():
    rng = np.random.default_rng(3)
    calibration, evaluation = rng.normal(size=(200000, 4)), rng.normal(size=(200000, 4))
    test = LimitTest(alpha=0.02).fit(calibration)
    z = NormalDist().inv_cdf(1 - 0.02 / 8)
    assert test.high == pytest.approx([z] * 4, abs=0.05)
    assert test.low == pytest.approx([-z] * 4, abs=0.05)
    assert test.flag(evaluation).mean() == pytest.approx(0.02, abs=0.002)
    assert test.flag(np.array([[0, 0, 0, 0], [0, 9, 0, 0], [0, 0, -9, 0]])).tolist() == [
        False, True, True,
    ]


# --- ambiguity structure --------------------------------------------------------------

CENTRE = pd.DataFrame(
    {"a": [0.0, 0.0, 10.0, 10.5, 30.0, 0.5], "b": [0.0, 20.0, 0.0, 0.0, 0.0, 0.0]},
    index=["healthy", "R1:open", "R2:open", "R3:open", "C1:open", "C2:small"],
)
SPREAD = pd.DataFrame(1.0, index=CENTRE.index, columns=CENTRE.columns)
OWNER = {"R1:open": "R1", "R2:open": "R2", "R3:open": "R3", "C1:open": "C1", "C2:small": "C2"}


def test_pairwise_shift_takes_the_most_separating_feature():
    d = pairwise_shift(CENTRE, SPREAD)
    assert d.loc["healthy", "R1:open"] == 20.0 and d.loc["R2:open", "R3:open"] == 0.5
    assert d.loc["R1:open", "R2:open"] == 20.0 and (np.diag(d) == 0).all()
    assert np.array_equal(d, d.T)
    # the noise floor is a lower bound of the spread
    wide = pairwise_shift(CENTRE, SPREAD * 0.0, floor=pd.Series({"a": 2.0, "b": 4.0}))
    assert wide.loc["healthy", "R1:open"] == 5.0 and wide.loc["healthy", "C1:open"] == 15.0


def test_ambiguity_groups_and_confusable_components():
    d = pairwise_shift(CENTRE, SPREAD)
    assert ambiguity_groups(d) == [
        ["healthy", "C2:small"], ["R2:open", "R3:open"], ["R1:open"], ["C1:open"],
    ]
    faults = d.drop(index="healthy", columns="healthy")
    assert confusable_components(faults, OWNER) == {
        "C1": [], "C2": [], "R1": [], "R2": ["R3"], "R3": ["R2"],
    }
    assert component_groups(faults, OWNER) == [["R2", "R3"], ["C1"], ["C2"], ["R1"]]
    # a wider threshold chains conditions: groups are an upper bound, partners are direct
    assert ambiguity_groups(d, threshold=12.0)[0] == ["healthy", "R2:open", "R3:open", "C2:small"]
    assert confusable_components(faults, OWNER, 12.0)["C2"] == ["R2", "R3"]


# --- what local sensitivity bounds ----------------------------------------------------


def test_normalised_sensitivity_rank_and_collinear_groups():
    # two measurements; R1 and R2 move them along the same direction, C1 along another,
    # and C2 hardly at all
    sensitivity = pd.DataFrame(
        {
            "R1": [1.0, 2.0, 0.3],
            "R2": [-0.5, -1.0, 9.9],
            "C1": [2.0, -1.0, 0.0],
            "C2": [0.01, 0.0, 5.0],
        },
        index=["m1", "m2", "flat"],
    )
    spread = pd.Series({"m1": 2.0, "m2": 4.0, "flat": 0.0})
    z = normalised_sensitivity(sensitivity, spread)
    assert list(z.index) == ["m1", "m2"]  # a measurement without spread is dropped
    assert z["R1"].tolist() == [0.5, 0.5] and z["C1"].tolist() == [1.0, -0.25]
    assert rank(z) == 2  # two directions for three sensitive components
    assert rank(z, deviation_pct=1.0) == 0 and rank(z.iloc[:0]) == 0
    groups, insensitive = collinear_groups(z)
    assert groups == [["R1", "R2"], ["C1"]] and insensitive == ["C2"]
