"""The statistics of a population against circuits whose distribution is known exactly."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repository root

from spicefault import Dataset  # noqa: E402
from validation import statistical  # noqa: E402


def test_the_exact_distributions_are_right():
    """The closed forms against a direct simulation of the populations, without SPICE."""
    rng = np.random.default_rng(0)
    r1, r2 = rng.uniform(0.95, 1.05, (2, 400_000))
    ratio = r2 / (r1 + r2)
    for x in (0.48, 0.49, 0.5, 0.515):
        assert statistical.divider_cdf(x) == pytest.approx((ratio <= x).mean(), abs=3e-3)
    assert statistical.divider_cdf(0.5) == pytest.approx(0.5, abs=1e-12)
    assert statistical.divider_cdf(0.4) == 0.0 and statistical.divider_cdf(0.6) == 1.0
    for q in statistical.QUANTILES:
        exact = statistical.divider_quantile(q)
        assert statistical.divider_cdf(exact) == pytest.approx(q, abs=1e-9)
        assert exact == pytest.approx(np.quantile(ratio, q), abs=2e-4)

    rc = statistical.R0 * statistical.C0 * np.exp(
        rng.normal(0, statistical.SIGMA_R, 400_000) + rng.normal(0, statistical.SIGMA_C, 400_000)
    )
    gain = 1 / np.sqrt(1 + (2 * np.pi * statistical.F0 * rc) ** 2)
    assert statistical.rc_gain(statistical.R0 * statistical.C0) == pytest.approx(2**-0.5)
    for q in statistical.QUANTILES:
        assert statistical.rc_quantile(q) == pytest.approx(np.quantile(gain, q), abs=3e-4)


@pytest.mark.ngspice
@pytest.mark.parametrize("name", ["divider", "rc"])
def test_estimates_agree_with_the_exact_values(name, tmp_path):
    """One campaign, at a size that CI can run: every estimate is close to the exact value,
    and closer than the width of its interval says.
    """
    case = statistical.cases()[name]
    case.campaign(tmp_path / "data", samples=600, seed=1).run(workers=2, progress=False)
    dataset = Dataset(tmp_path / "data")
    assert dataset.verify() == [] and dataset.ok.all()
    table = statistical.compare(case, dataset).set_index("quantity")
    assert list(table.index)[:3] == ["quantile 0.05", "quantile 0.5", "quantile 0.95"]
    assert "yield" in table.index and np.isfinite(table[["low", "high"]].to_numpy()).all()
    width = table["high"] - table["low"]
    assert (np.abs(table["estimate"] - table["exact"]) <= width).all()
    assert table.loc["yield", "estimate"] == pytest.approx(statistical.YIELD, abs=0.04)
    assert table["covered"].sum() >= len(table) - 1  # 95 % intervals: at most one miss


@pytest.mark.ngspice
def test_intervals_cover_the_exact_value_over_repeated_seeds(tmp_path):
    """Twelve independent campaigns: an interval that claims 95 % and covers the truth in
    fewer than nine of twelve has less than one chance in two thousand of being right.
    """
    case = statistical.cases()["divider"]
    summary = statistical.coverage(case, tmp_path, samples=150, seeds=12, workers=2)
    assert set(summary["quantity"]) == {
        "quantile 0.05", "quantile 0.5", "quantile 0.95", "mean", "yield",
    }  # fmt: skip
    assert (summary["campaigns"] == 12).all() and (summary["covered"] >= 9).all()
