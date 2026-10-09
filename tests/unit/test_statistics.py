import math

import numpy as np
import pandas as pd
import pytest
from campaign_backend import Divider

from spicefault import (
    Campaign,
    Circuit,
    Measurement,
    OperatingCondition,
    Simulator,
    Specification,
)
from spicefault.statistics import (
    describe,
    ecdf,
    quantile_interval,
    samples_for_half_width,
    yield_report,
    zero_failure_bound,
)
from spicefault.variation import tolerances

CIRCUIT = Circuit("divider\nV1 in 0 dc 1\nR1 in out 10k\nR2 out 0 10k\n.end\n", "divider")


def test_quantile_interval_is_distribution_free_and_covers_as_often_as_it_says():
    # the median of 10: ranks 2 and 9, from the Binomial(10, 0.5) tails 11/1024 <= 0.025
    assert quantile_interval(10, 0.5) == (2, 9)
    assert quantile_interval(100, 0.5) == (40, 61)
    # the smallest of n values is below the quantile q with probability 1 - (1 - q)^n:
    # 100 samples cannot bracket the 1 % quantile, 1000 can
    assert quantile_interval(100, 0.01) is None and quantile_interval(100, 0.99) is None
    low, high = quantile_interval(1000, 0.01)
    assert 1 <= low < 10 < high <= 20
    assert quantile_interval(0, 0.5) is None
    with pytest.raises(ValueError, match="between 0 and 1"):
        quantile_interval(10, 1.0)

    # coverage on a skewed population whose quantiles are known: exponential, q = 0.9
    rng = np.random.default_rng(0)
    truth, n, trials = -math.log(0.1), 200, 2000
    low, high = quantile_interval(n, 0.9, confidence=0.9)
    x = np.sort(rng.exponential(size=(trials, n)), axis=1)
    covered = ((x[:, low - 1] <= truth) & (truth <= x[:, high - 1])).mean()
    assert 0.90 <= covered <= 0.95  # at least the confidence; conservative, as it is discrete


def test_describe_counts_failures_and_gives_no_quantile_it_cannot_support():
    rng = np.random.default_rng(1)
    n = 4000
    samples = pd.DataFrame({
        "fault_id": ["healthy"] * n + ["R1:open"] * 50,
        "condition": "nominal",
        "sim_ok": [True] * (n - 40) + [False] * 40 + [True] * 50,
        "gain": np.concatenate([rng.normal(10.0, 2.0, n), rng.normal(0.0, 1.0, 50)]),
        "unused": np.nan,
    })  # fmt: skip
    samples.loc[~samples["sim_ok"], "gain"] = np.nan
    table = describe(samples, ["gain", "unused"], quantiles=(0.001, 0.5, 0.975))
    assert list(table.index) == [("healthy", "nominal", "gain"), ("R1:open", "nominal", "gain")]
    healthy = table.loc[("healthy", "nominal", "gain")]
    assert healthy["n"] == n - 40 and healthy["n_failed"] == 40
    assert healthy["mean_low"] < 10.0 < healthy["mean_high"]
    width = healthy["mean_high"] - healthy["mean_low"]
    assert width == pytest.approx(2 * 1.96 * 2 / n**0.5, rel=0.05)
    assert healthy["std"] == pytest.approx(2.0, rel=0.05)
    assert healthy["q0.5_low"] < 10.0 < healthy["q0.5_high"]
    assert healthy["q0.975_low"] < 10.0 + 1.96 * 2.0 < healthy["q0.975_high"]
    assert healthy["q0.001_low"] <= healthy["q0.001"] <= healthy["q0.001_high"]
    assert healthy["minimum"] <= healthy["q0.001_low"]
    # 50 samples say nothing about the 0.1 % quantile: not the smallest value, but nothing
    faulty = table.loc[("R1:open", "nominal", "gain")]
    assert faulty["n"] == 50 and np.isnan(faulty[["q0.001", "q0.001_low", "q0.001_high"]]).all()
    assert np.isfinite(faulty[["q0.5", "q0.5_low", "q0.5_high"]]).all()

    x, p = ecdf([3.0, 1.0, np.nan, 2.0])
    assert x.tolist() == [1.0, 2.0, 3.0] and p.tolist() == pytest.approx([1 / 3, 2 / 3, 1.0])


def test_yield_and_its_interval_against_a_population_with_a_known_yield():
    # a standard normal between -1.5 and 2.0: yield Phi(2) - Phi(-1.5); a second, independent
    # specification with yield 0.9
    rng = np.random.default_rng(2)
    n = 20000
    values = pd.DataFrame({"gain": rng.normal(size=n), "noise": rng.uniform(size=n)})
    limits = [Specification("gain", -1.5, 2.0), Specification("noise", maximum=0.9)]
    report = yield_report(values, limits)
    exact = 0.9772499 - 0.0668072
    assert list(report.index) == ["gain", "noise", "all"]
    for name, truth in (("gain", exact), ("noise", 0.9), ("all", exact * 0.9)):
        row = report.loc[name]
        assert row["ci_low"] < truth < row["ci_high"]
        assert row["yield"] == pytest.approx(truth, abs=0.01)
        assert row["yield_min"] == row["yield_max"] == row["yield"] and row["n_failed"] == 0
    gain = report.loc["gain"]
    assert gain["margin_sigma"] == pytest.approx(1.5, abs=0.05)  # the nearer limit
    assert gain["minimum"] < -1.5 and gain["maximum"] > 2.0 and gain["minimum_limit"] == -1.5
    assert report.loc["noise", "margin_sigma"] == pytest.approx(0.4 / (1 / 12**0.5), rel=0.05)
    both = ((values["gain"] < -1.5) | (values["gain"] > 2.0)) & (values["noise"] > 0.9)
    assert gain["only_this"] == gain["n"] - gain["passed"] - both.sum()

    # the interval covers the truth as often as it says, for both estimators
    trials, size, truth = 500, 300, 0.9
    for interval, least in (("wilson", 0.92), ("clopper-pearson", 0.95)):
        covered = 0
        for _ in range(trials):
            sample = pd.DataFrame({"noise": rng.uniform(size=size)})
            row = yield_report(sample, limits[1:], interval=interval).loc["all"]
            covered += row["ci_low"] <= truth <= row["ci_high"]
        assert least <= covered / trials <= 0.99

    # failed simulations give two bounds; a value that is not finite is a violation
    few = pd.DataFrame({"gain": [0.0, 0.0, 5.0, np.nan, np.nan, 0.0]})
    simulated = np.array([True, True, True, True, False, False])
    row = yield_report(few, limits[:1], simulated).loc["all"]
    assert (row["n"], row["n_failed"], row["passed"]) == (4, 2, 2) and row["yield"] == 0.5
    assert row["yield_min"] == pytest.approx(2 / 6) and row["yield_max"] == pytest.approx(4 / 6)
    with pytest.raises(ValueError, match="at least one specification"):
        yield_report(few, [])


def test_sample_size_helpers_match_their_closed_forms():
    assert samples_for_half_width(0.01) == math.ceil(1.959964**2 * 0.25 / 1e-4) == 9604
    assert samples_for_half_width(0.01, p=0.99) == 381
    # no failure in n: (1 - p)^n = 0.05, the "rule of three" for large n
    assert zero_failure_bound(200) == pytest.approx(1 - 0.05 ** (1 / 200))
    assert zero_failure_bound(3000) == pytest.approx(3 / 3000, rel=0.01)
    assert (1 - zero_failure_bound(50, 0.99)) ** 50 == pytest.approx(0.01)
    with pytest.raises(ValueError):
        samples_for_half_width(0.0)


def third_of_supply(result):
    return float(result.plot("Operating Point")["v(out)"][0].real) / 2.0


@pytest.fixture(scope="module")
def monte_carlo(tmp_path_factory):
    conditions = [
        OperatingCondition("service", measurements=(Measurement.value("v(out)", name="vout"),)),
        OperatingCondition("bench", settings={("V1", "dc"): 2.0},
                           measurements=(Measurement.custom_result("half", third_of_supply),)),
    ]  # fmt: skip
    campaign = Campaign(
        CIRCUIT, out_dir=tmp_path_factory.mktemp("mc") / "data", samples=400,
        variations=tolerances(CIRCUIT, {"R": 0.003}), conditions=conditions,
        simulator=Simulator(Divider()), seed=11,
    )  # fmt: skip
    campaign.run(progress=False)
    return campaign.dataset()


def test_dataset_statistics_and_yield_of_a_fault_free_campaign(monte_carlo):
    dataset = monte_carlo
    assert dataset.ok.all() and len(dataset) == 800
    table = dataset.statistics()
    assert list(table.index) == [("healthy", "service", "vout"), ("healthy", "bench", "half")]
    assert (table["n"] == 400).all() and (table["n_failed"] == 0).all()
    # the ratio of two resistors within 0.3 %: centred on 0.5, never beyond 0.5 +- 0.0015
    vout = table.loc[("healthy", "service", "vout")]
    assert vout["mean_low"] < 0.5 < vout["mean_high"] and 0.4985 <= vout["minimum"]
    assert vout["maximum"] <= 0.5015 and vout["q0.5_low"] < 0.5 < vout["q0.5_high"]
    by_case = dataset.statistics(by_case=True, quantiles=(0.5,))
    assert list(by_case.index) == [("healthy", "vout"), ("healthy", "half")]
    assert by_case.loc[("healthy", "half"), "mean"] == pytest.approx(vout["mean"])

    limits = [Specification("vout", 0.4995, 0.5005), Specification("half", maximum=0.5008)]
    report = dataset.yield_report(limits)
    cases, _ = dataset.cases()
    passed = cases["vout"].between(0.4995, 0.5005) & (cases["half"] <= 0.5008)
    assert report.loc["all", "passed"] == passed.sum() and report.loc["all", "n"] == 400
    assert 0.3 < report.loc["all", "yield"] < 0.9
    assert report.loc["all", "ci_low"] < report.loc["all", "yield"] < report.loc["all", "ci_high"]
    assert report.loc["vout", "passed"] == cases["vout"].between(0.4995, 0.5005).sum()
    # other limits need no simulation and write nothing; the stored ones are the default
    wide = dataset.yield_report([Specification("vout", 0.4, 0.6)])
    assert wide.loc["all", "yield"] == 1.0 and wide.loc["all", "ci_low"] > 0.99
    assert wide.loc["all", "ci_low"] == pytest.approx(1 - zero_failure_bound(400, 0.95), abs=0.005)
    assert dataset.labels == []
    with pytest.raises(ValueError, match="label the dataset first"):
        dataset.yield_report()
    with pytest.raises(KeyError, match="no column of the drawn circuits"):
        dataset.yield_report([Specification("gain", 1.0)])
    with pytest.raises(KeyError, match="no circuit with fault 'R1:open'"):
        dataset.yield_report(limits, fault_id="R1:open")
