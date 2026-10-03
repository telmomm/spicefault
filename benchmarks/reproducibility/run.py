"""Reproducibility of a campaign (docs/EXPERIMENT_PLAN.md, experiment B).

    python -m benchmarks.reproducibility.run --workers 8

The same campaign, with the same seed, is run with one worker, with several, twice
with several, and interrupted and resumed. Each run is compared with the first:

- L1, sample definitions: fault, condition, seed key and every drawn value must be
  exactly equal;
- L2, outputs: status, measurements and waveforms, with the largest difference found.

Samples of the first dataset are then simulated again from its folder.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np

from benchmarks import workloads
from benchmarks.common import environment, save
from spicefault import Dataset, FaultCampaign
from spicefault.experiments import run_chunks, simulate_sample


def compare(reference: Dataset, other: Dataset) -> dict:
    """Level 1 and level 2 comparison of two datasets of the same campaign."""
    a, b = reference.samples, other.samples
    definition = ["sample_id", "fault_index", "fault_id", "replica", "condition", "seed_key"]
    parameters = list(reference.parameters)
    features = reference.features
    x, y = a[features].to_numpy(float), b[features].to_numpy(float)
    both_missing = np.isnan(x) & np.isnan(y)
    diff = np.where(both_missing, 0.0, np.abs(x - y))
    relative = diff / np.maximum(np.abs(x), 1e-300)
    result = {
        "L1_definitions_identical": bool(a[definition].equals(b[definition])),
        "L1_drawn_values_identical": bool(
            np.array_equal(
                a[parameters].to_numpy(float), b[parameters].to_numpy(float), equal_nan=True
            )
        ),
        "labels_identical": bool(a[reference.labels].equals(b[reference.labels])),
        "status_identical": bool(a["status"].equals(b["status"])),
        "L2_max_abs_diff": float(np.nanmax(diff)) if diff.size else 0.0,
        "L2_max_rel_diff": float(np.nanmax(np.where(both_missing, 0.0, relative)))
        if diff.size
        else 0.0,
        "L2_measurements_identical": bool(np.array_equal(x, y, equal_nan=True)),
    }
    if reference.waveforms is not None:
        result["L2_waveforms_identical"] = bool(
            np.array_equal(reference.waveforms, other.waveforms, equal_nan=True)
        )
    return result


def run(workload: str = "rc", workers: int = 8, n_samples: int = 5000, chunk: int = 500) -> dict:
    experiment = workloads.experiment(workload, n_samples)
    n_points = None if experiment.waveform is None else experiment.waveform.n_points
    result = {
        "benchmark": "reproducibility",
        "environment": environment(),
        "protocol": {
            "workload": workload,
            "n_samples": len(experiment.plan()),
            "seed": experiment.seed,
            "seeding": experiment.seeding,
            "workers": workers,
            "chunk": chunk,
        },
    }
    with tempfile.TemporaryDirectory(prefix="spicefault_bench_") as tmp:
        tmp = Path(tmp)

        def campaign(name: str) -> FaultCampaign:
            return FaultCampaign.from_experiment(experiment, tmp / name)

        reference = campaign("one_worker")
        reference.run(workers=1, chunk=chunk, progress=False)
        runs = {"several_workers": (workers, chunk), "several_workers_again": (workers, chunk),
                "other_chunk_size": (workers, max(chunk // 3, 1))}  # fmt: skip
        for name, (n_workers, size) in runs.items():
            campaign(name).run(workers=n_workers, chunk=size, progress=False, validate=False)

        # interrupted after the first third of the chunks, then launched again
        resumed = campaign("resumed")
        plan = experiment.plan()
        done = (len(plan) // chunk // 3 or 1) * chunk
        run_chunks(plan[:done], simulate_sample, experiment, resumed.out_dir / "parts", n_points,
                   workers, chunk, progress=False)  # fmt: skip
        resumed.run(workers=workers, chunk=chunk, progress=False, validate=False)

        first = reference.dataset()
        result["status_counts"] = first.manifest.summary["status_counts"]
        result["comparisons_with_one_worker"] = {
            name: compare(first, campaign(name).dataset()) for name in [*runs, "resumed"]
        }
        result["resumed_run"] = {
            "samples_before_interruption": done,
            "manifest_says_resumed": campaign("resumed").dataset().manifest.resumed,
        }
        result["integrity_problems"] = first.verify()
        again = first.reproduce(n=min(50, len(first)), experiment=experiment)
        result["simulated_again_from_the_folder"] = {
            "n_samples": len(again),
            "definitions_identical": bool(again["definition"].all()),
            "drawn_values_identical": bool(again["parameters"].all()),
            "status_identical": bool(again["status"].all()),
            "max_abs_diff": float(again["max_abs_diff"].max()),
            "max_rel_diff": float(again["max_rel_diff"].max()),
            "waveform_max_abs_diff": float(again["waveform_abs_diff"].max()),
        }
    checks = result["comparisons_with_one_worker"].values()
    result["verdict"] = {
        "L1_exact": all(
            c["L1_definitions_identical"] and c["L1_drawn_values_identical"] for c in checks
        ),
        "L2_identical": all(
            c["L2_measurements_identical"] and c.get("L2_waveforms_identical", True) for c in checks
        ),
        "L2_max_rel_diff": max(c["L2_max_rel_diff"] for c in checks),
    }
    return result


def report(result: dict) -> str:
    protocol = result["protocol"]
    lines = [
        f"workload {protocol['workload']}, {protocol['n_samples']} samples, "
        f"1 worker against {protocol['workers']}"
    ]
    for name, checks in result["comparisons_with_one_worker"].items():
        lines.append(f"  {name}: " + ", ".join(f"{k}={v}" for k, v in checks.items()))
    lines.append(f"  simulated again: {result['simulated_again_from_the_folder']}")
    lines.append(f"verdict: {result['verdict']}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--workload", default="rc", choices=workloads.WORKLOADS)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--chunk", type=int, default=500)
    parser.add_argument("--label", default="")
    args = parser.parse_args()
    result = run(args.workload, args.workers, args.samples, args.chunk)
    print(report(result))
    print("\nsaved to", save("reproducibility", result, args.label or args.workload))


if __name__ == "__main__":
    main()
