import math

import numpy as np
import pandas as pd
import pytest

from spicefault.testdesign import metrics as mt
from spicefault.testdesign.select import baselines as bl


def synthetic(n=60, seed=0):
    """Three tests. Test a separates the faults into {healthy, b} and {a, c} by 10 sigma;
    test b separates b from healthy; test c is noise. So {a, b} isolate everything but
    a from c, which nothing separates."""
    rng = np.random.default_rng(seed)
    means = {"healthy": (0, 0, 0), "a": (10, 0, 0), "b": (0, 10, 0), "c": (10, 0, 0)}
    rows = [
        {"fault_id": f, "replica": r, **{t: rng.normal(m) for t, m in zip("abc", mu, strict=True)}}
        for f, mu in means.items()
        for r in range(n)
    ]
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def matrix():
    return synthetic()


def test_lscv_prefers_a_sensible_bandwidth():
    x = np.random.default_rng(1).normal(size=200)
    rule = bl.bandwidth(x, "silverman")
    assert bl.lscv(x, rule) < bl.lscv(x, rule / 20)
    assert bl.lscv(x, rule) < bl.lscv(x, rule * 20)
    h = bl.bandwidth(x, "gwo")
    assert rule / 10 <= h <= rule * 3
    assert (
        bl.lscv(x, h) <= bl.lscv(x, rule) + 1e-6
    )  # GWO does at least as well as the rule of thumb


def test_grey_wolf_finds_a_minimum():
    assert bl.grey_wolf(lambda x: (x - 0.3) ** 2, -1, 1) == pytest.approx(0.3, abs=1e-3)


def test_bhattacharyya():
    rng = np.random.default_rng(2)
    x, y = rng.normal(0, 1, 400), rng.normal(0, 1, 400)
    far = rng.normal(20, 1, 400)
    h = bl.bandwidth(x, "silverman")
    assert bl.bhattacharyya_kde(x, y, h, h) > 0.95
    assert bl.bhattacharyya_kde(x, far, h, h) < 1e-6
    assert bl.bhattacharyya_normal(x, y) > 0.98
    # closed form for equal spreads: exp(-(mean difference)^2 / (8 sigma^2)); here sigma^2 = 2
    assert bl.bhattacharyya_normal(np.array([-1.0, 1.0]), np.array([0.0, 2.0])) == pytest.approx(
        math.exp(-1 / 16), rel=1e-6
    )
    assert bl.bhattacharyya_kde(np.zeros(5), np.zeros(5), 0.0, 0.0) == 1.0


@pytest.mark.parametrize("method,threshold", [("kde", 0.95), ("normal", 0.95), ("nominal", 5.0)])
def test_pair_table_and_selection(matrix, method, threshold):
    t = bl.pair_table(
        matrix, ["a", "b", "c"], method=method, threshold=threshold, bandwidth_method="silverman"
    )
    assert len(t.table) == 6  # 4 conditions, 6 pairs
    assert not t.table.loc[("a", "c")].any()
    assert t.table.loc[("healthy", "a"), "a"] and not t.table.loc[("healthy", "a"), "b"]
    assert not t.table["c"].any()
    chosen = bl.select_pairs(t)
    assert sorted(chosen) == ["a", "b"]
    iso = bl.isolation(t, chosen)
    assert iso == {"remaining_pairs": 1, "isolated": 2, "conditions": 4}


def test_integer_codes_entropy_and_genetic(matrix):
    codes = bl.integer_codes(matrix, ["a", "b", "c"], half_width=2.5)
    assert codes.loc["a", "a"] == codes.loc["c", "a"] != codes.loc["healthy", "a"]
    assert codes["c"].nunique() == 1
    assert sorted(bl.starzyk_entropy(codes)) == ["a", "b"]
    assert bl.isolated_by_codes(codes, ["a", "b"]) == 2
    # the fitness of Golonek and Rutkowski weighs a test left out as much as a condition
    # isolated: here {a, b} (2 isolated + 1 left out) ties with no test at all (0 + 3)
    assert sorted(bl.genetic(codes, population=20, generations=30)) in ([], ["a", "b"])


def test_zilch_greedy(matrix):
    detects = bl.detection_table(matrix, ["a", "b", "c"], lambda: mt.GaussianLimits(0.01))
    assert detects.loc["a", "a"] and detects.loc["b", "b"] and not detects["c"].any()
    assert bl.zilch_greedy(detects) == ["a", "b"] or bl.zilch_greedy(detects) == ["b", "a"]


# Table II of Golonek and Rutkowski (2007), the same dictionary as Table II of Starzyk et al. (2004)
PUBLISHED = """
S0  3 2 2 3 3 3 4 4 1 1 1
S1  0 0 0 0 0 0 0 0 1 1 7
S2  3 2 2 3 4 3 4 4 1 1 0
S3  1 0 0 0 0 0 0 0 1 1 7
S4  2 3 3 4 6 5 6 6 1 1 2
S5  1 1 1 1 1 1 1 1 1 1 5
S6  0 0 0 2 3 2 3 3 1 1 8
S7  3 2 2 3 5 4 5 5 1 1 4
S8  3 2 2 3 0 0 0 0 1 1 7
S9  3 2 2 3 6 6 7 7 1 1 6
S10 3 2 2 3 2 0 0 0 1 1 7
S11 3 2 2 3 3 3 4 2 1 1 4
S12 3 2 2 3 3 2 0 8 5 1 8
S13 3 2 2 3 3 2 2 8 4 1 8
S14 3 2 2 3 3 3 4 4 1 1 0
S15 3 2 2 3 3 3 4 4 2 1 8
S16 3 2 2 3 3 3 4 4 3 1 8
S17 3 2 2 3 3 3 4 4 2 2 3
S18 3 2 2 3 3 3 4 4 0 0 1
"""


def published_codes():
    rows = [line.split() for line in PUBLISHED.strip().splitlines()]
    return pd.DataFrame(
        [[int(v) for v in r[1:]] for r in rows],
        index=[r[0] for r in rows],
        columns=[f"V{i}" for i in range(1, 12)],
    )


def test_starzyk_on_the_published_dictionary():
    # Starzyk et al. select n11, n9, n5, n1 in that order and isolate every condition
    codes = published_codes()
    assert bl.starzyk_entropy(codes) == ["V11", "V9", "V5", "V1"]
    assert bl.isolated_by_codes(codes, ["V1", "V5", "V9", "V11"]) == 19
    # the index of V10 in the first iteration, 20.9 in Table III, is 17 log10 17
    assert bl.entropy_index(codes, ["V10"]) / np.log(10) == pytest.approx(20.9, abs=0.05)


def test_genetic_on_the_published_dictionary():
    # Golonek and Rutkowski find V1, V5, V9 and V11 (Eq. 12)
    found = [sorted(bl.genetic(published_codes(), seed=s)) for s in range(5)]
    assert sum(f == ["V1", "V11", "V5", "V9"] for f in found) >= 4
