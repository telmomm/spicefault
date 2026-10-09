"""Estimation error of the sampling methods at equal numbers of simulations.

    python -m benchmarks.sampling.run --sizes 64 256 --repetitions 16 --workers 8

The RC low-pass of `validation/statistical.py`, whose gain distribution is known, is
sampled at random, by Latin hypercube and by Sobol sequence. For each method and
number of simulations, the experiment is repeated with different seeds and the
root-mean-square error of two estimates is reported: the mean gain, against its value
by numerical integration, and the yield of a specification placed for a yield of 0.90.

A stratified design is worth its restrictions (no independent interval, every
variation with a quantile function) only where its error is clearly below that of
random sampling at the same cost. It is for a mean; for a proportion, such as a yield,
the gain is smaller.
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from benchmarks.common import environment, save
from spicefault import Experiment
from validation import statistical

METHODS = ("random", "lhs", "sobol")


def exact_mean_gain() -> float:
    """E[gain] over the log-normal RC, by integrating over the standard normal."""
    z = np.linspace(-8.0, 8.0, 20001)
    rc = statistical.R0 * statistical.C0 * np.exp(statistical.SIGMA * z)
    gain = 1.0 / np.sqrt(1.0 + (2.0 * math.pi * statistical.F0 * rc) ** 2)
    density = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    return float(np.sum(gain * density) * (z[1] - z[0]))


def run(sizes=(64, 256), repetitions: int = 16, workers: int = 1) -> dict:
    case = statistical.cases()["rc"]
    specification, name = case.specification, case.measurement.name
    truth = {"mean": exact_mean_gain(), "yield": statistical.YIELD}
    rows = []
    for method in METHODS:
        for size in sizes:
            errors = {"mean": [], "yield": []}
            for seed in range(repetitions):
                experiment = Experiment(
                    case.circuit, variations=case.variations, config=case.config,
                    measurements=[case.measurement], samples=size, seed=seed, sampling=method,
                )  # fmt: skip
                values = experiment.run(workers=workers).to_frame()[name].to_numpy(dtype=float)
                errors["mean"].append(values.mean() - truth["mean"])
                errors["yield"].append(specification.met(values).mean() - truth["yield"])
            rows.append({
                "method": method, "simulations": size, "repetitions": repetitions,
                **{f"rmse_{k}": float(np.sqrt(np.mean(np.square(v)))) for k, v in errors.items()},
            })  # fmt: skip
    reference = {(r["simulations"]): r for r in rows if r["method"] == "random"}
    for row in rows:
        base = reference[row["simulations"]]
        row["mean_error_vs_random"] = row["rmse_mean"] / base["rmse_mean"]
        row["yield_error_vs_random"] = row["rmse_yield"] / base["rmse_yield"]
    return {"benchmark": "sampling", "environment": environment(), "exact": truth, "rows": rows}


def report(result: dict) -> str:
    lines = [f"exact mean gain {result['exact']['mean']:.6f}, exact yield "
             f"{result['exact']['yield']:.2f}",
             "method  simulations  rmse(mean)  vs random  rmse(yield)  vs random"]  # fmt: skip
    for r in result["rows"]:
        lines.append(
            f"{r['method']:<7} {r['simulations']:>11}  {r['rmse_mean']:.2e}  "
            f"{r['mean_error_vs_random']:>9.2f}  {r['rmse_yield']:>11.4f}  "
            f"{r['yield_error_vs_random']:>9.2f}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=[64, 256])
    parser.add_argument("--repetitions", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--label", default="")
    arguments = parser.parse_args()
    result = run(tuple(arguments.sizes), arguments.repetitions, arguments.workers)
    print(report(result))
    print("written to", save("sampling", result, arguments.label))


if __name__ == "__main__":
    main()
