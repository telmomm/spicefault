"""Validation of the statistics of a population against circuits with a known answer.

Two circuits whose output distribution has a closed form are drawn and simulated as
campaigns without faults, and what `Dataset.statistics` and `Dataset.yield_report`
estimate is compared with the exact values:

- `divider`: two equal resistors with a uniform tolerance. The output ratio is
  R2 / (R1 + R2), and its distribution function is the area of a region of a square.
- `rc`: an RC low-pass with log-normal R and C. The product RC is exactly log-normal,
  and the gain at a fixed frequency is a decreasing function of it, so its quantiles
  are those of RC.

The specification of each is placed at exact quantiles, so its yield is known: 0.90.

    python -m validation.statistical --samples 4000 --seeds 20 --workers 8

compares one campaign of each circuit, and measures over repeated seeds how often the
intervals cover the exact value. See docs/validation.md.
"""

from __future__ import annotations

import argparse
import math
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist

import pandas as pd

from spicefault import Campaign, Circuit, Dataset, Measurement, SimulationConfig, Specification
from spicefault.variation import LogNormalVariation, VariationSet, tolerances

QUANTILES = (0.05, 0.5, 0.95)
YIELD = 0.90  # of the specification of each circuit, by construction
DATA = Path("data") / "validation" / "statistical"

# --- a divider with uniform tolerances ------------------------------------------------

TOLERANCE = 0.05


def divider_cdf(x: float, tolerance: float = TOLERANCE) -> float:
    """P(R2 / (R1 + R2) <= x) for R1 and R2 uniform within nominal (1 +- tolerance).

    The ratio is below x when R1 >= c R2, with c = (1 - x) / x. With u and v uniform on
    [lo, hi], that is the area of {u >= c v} over the area of the square.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lo, hi = 1.0 - tolerance, 1.0 + tolerance
    width, c = hi - lo, (1.0 - x) / x
    # for each v, u runs over a length hi - c v, limited to [0, width]
    a = min(max(lo / c, lo), hi)  # below a the whole width counts
    b = min(max(hi / c, lo), hi)  # above b nothing does
    area = width * (a - lo) + hi * (b - a) - c * (b * b - a * a) / 2.0
    return area / (width * width)


def _inverse(cdf: Callable[[float], float], q: float, low: float, high: float) -> float:
    for _ in range(200):
        middle = 0.5 * (low + high)
        low, high = (middle, high) if cdf(middle) < q else (low, middle)
    return 0.5 * (low + high)


def divider_quantile(q: float, tolerance: float = TOLERANCE) -> float:
    return _inverse(lambda x: divider_cdf(x, tolerance), q, 0.0, 1.0)


# --- an RC low-pass with log-normal values ---------------------------------------------

R0, C0, F0 = 1e3, 159.15494309189535e-9, 1e3  # the corner is at 1 kHz
SIGMA_R, SIGMA_C = 0.03, 0.08
SIGMA = math.hypot(SIGMA_R, SIGMA_C)  # of ln(RC): the sum of two independent normals


def rc_gain(rc: float) -> float:
    """|H(F0)| of a first-order low-pass of time constant `rc`."""
    return 1.0 / math.sqrt(1.0 + (2.0 * math.pi * F0 * rc) ** 2)


def rc_quantile(q: float) -> float:
    """Quantile of the gain: the gain falls as RC grows, so it is the gain at the
    (1 - q) quantile of RC.
    """
    return rc_gain(R0 * C0 * math.exp(SIGMA * NormalDist().inv_cdf(1.0 - q)))


# --- the two studies -------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    """A circuit, its population, one measurement and what is known about it exactly."""

    name: str
    circuit: Circuit
    variations: VariationSet
    config: SimulationConfig
    measurement: Measurement
    quantile: Callable[[float], float]
    mean: float | None  # None if it has no closed form

    @property
    def specification(self) -> Specification:
        """Limits at exact quantiles, for a yield of exactly `YIELD`."""
        tail = (1.0 - YIELD) / 2.0
        return Specification(self.measurement.name, self.quantile(tail), self.quantile(1 - tail))

    def campaign(self, out_dir: str | Path, samples: int, seed: int) -> Campaign:
        return Campaign(
            self.circuit, out_dir=out_dir, samples=samples, variations=self.variations,
            config=self.config, measurements=[self.measurement], seed=seed,
        )  # fmt: skip


def cases() -> dict[str, Case]:
    divider = Circuit("divider\nV1 in 0 dc 1\nR1 in out 10k\nR2 out 0 10k\n.end\n", "divider")
    rc = Circuit(f"rc\nV1 in 0 dc 0 ac 1\nR1 in out {R0!r}\nC1 out 0 {C0!r}\n.end\n", "rc")
    return {
        "divider": Case(
            "divider", divider, tolerances(divider, {"R": TOLERANCE}),
            SimulationConfig(("op",), outputs=("v(out)",)),
            Measurement.value("v(out)", name="ratio"), divider_quantile,
            mean=0.5,  # R1 and R2 are exchangeable, so the ratio and 1 - ratio are too
        ),
        "rc": Case(
            "rc", rc,
            VariationSet([LogNormalVariation("R1", None, SIGMA_R),
                          LogNormalVariation("C1", None, SIGMA_C)]),
            # ten points per decade from 100 Hz: 1 kHz is a point of the sweep
            SimulationConfig(("ac dec 10 100 10k",), outputs=("v(out)",)),
            Measurement.magnitude("v(out)", F0, name="gain"), rc_quantile, mean=None,
        ),
    }  # fmt: skip


def compare(case: Case, dataset: Dataset, confidence: float = 0.95) -> pd.DataFrame:
    """Each estimate with its interval next to the exact value, and whether it covers it."""
    name = case.measurement.name
    row = dataset.statistics(quantiles=QUANTILES, confidence=confidence).iloc[0]
    found = [
        (f"quantile {q:g}", row[f"q{q:g}"], row[f"q{q:g}_low"], row[f"q{q:g}_high"],
         case.quantile(q))
        for q in QUANTILES
    ]  # fmt: skip
    if case.mean is not None:
        found.append(("mean", row["mean"], row["mean_low"], row["mean_high"], case.mean))
    report = dataset.yield_report([case.specification], confidence=confidence).loc[name]
    found.append(("yield", report["yield"], report["ci_low"], report["ci_high"], YIELD))
    table = pd.DataFrame(found, columns=["quantity", "estimate", "low", "high", "exact"])
    table.insert(0, "circuit", case.name)
    table["covered"] = (table["low"] <= table["exact"]) & (table["exact"] <= table["high"])
    return table


def coverage(
    case: Case, folder: str | Path, samples: int, seeds: int, workers: int = 1
) -> pd.DataFrame:
    """Over `seeds` independent campaigns, how often each interval covers the exact value."""
    tables = []
    for seed in range(seeds):
        out_dir = Path(folder) / f"{case.name}_{samples}_{seed}"
        case.campaign(out_dir, samples, seed).run(workers=workers, progress=False)
        tables.append(compare(case, Dataset(out_dir)))
    joined = pd.concat(tables)
    summary = joined.groupby(["circuit", "quantity"], sort=False)["covered"].agg(["sum", "count"])
    summary.columns = ["covered", "campaigns"]
    summary["fraction"] = summary["covered"] / summary["campaigns"]
    return summary.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--samples", type=int, default=4000)
    parser.add_argument("--seeds", type=int, default=20, help="campaigns for the coverage")
    parser.add_argument("--coverage-samples", type=int, default=500)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--keep", action="store_true", help="keep the datasets")
    arguments = parser.parse_args()
    pd.set_option("display.width", 160)
    compared, covered = [], []
    for case in cases().values():
        out_dir = DATA / f"{case.name}_{arguments.samples}"
        case.campaign(out_dir, arguments.samples, seed=1).run(arguments.workers, progress=False)
        compared.append(compare(case, Dataset(out_dir)))
        covered.append(
            coverage(case, DATA, arguments.coverage_samples, arguments.seeds, arguments.workers)
        )
    print(f"one campaign of {arguments.samples} circuits, 95 % intervals:")
    print(pd.concat(compared).to_string(index=False, float_format="{:.5f}".format))
    print(f"\ncoverage over {arguments.seeds} campaigns of {arguments.coverage_samples}:")
    print(pd.concat(covered).to_string(index=False, float_format="{:.3f}".format))
    if not arguments.keep:
        shutil.rmtree(DATA, ignore_errors=True)


if __name__ == "__main__":
    main()
