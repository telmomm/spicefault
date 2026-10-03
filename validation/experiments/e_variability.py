"""Experiment E: how tolerance degrades detectability and diagnostic coverage.

    python -m validation.experiments.e_variability run --circuit sallen_key --workers 8
    python -m validation.experiments.e_variability analyse --circuit sallen_key

`run` simulates the campaign of a study at several tolerance scales (0, 0.2, 1 and 2
times the declared tolerances). The circuits are the same at every scale, with
deviations in proportion. `analyse` reads those datasets and writes, per scale, the
false-alarm rate, the yield, the diagnostic coverage and the detection probability of
every fault, and per fault the largest scale at which it is still detected.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import validation
from spicefault.reliability import robustness

from .common import ALPHA, BETA, DATA, analysis_of, save

SCALES = (0.0, 0.2, 1.0, 2.0)


def folder(data: Path, circuit: str, scale: float) -> Path:
    return data / circuit / f"scale_{scale:g}"


def run(circuit, data, scales, samples_per_fault, healthy, workers) -> None:
    study = validation.get(circuit)
    for scale in scales:
        campaign = study.campaign(
            folder(data, circuit, scale),
            samples_per_fault=samples_per_fault,
            healthy_samples=healthy,
            tolerance_scale=scale,
        )
        campaign.run(workers=workers, progress=False)
        print(f"{circuit}, tolerance scale {scale:g}: {campaign.status()}")


def analyse(circuit, data, scales, alpha=ALPHA, beta=BETA) -> dict:
    study = validation.get(circuit)
    analyses = {scale: analysis_of(study, folder(data, circuit, scale)) for scale in scales}
    per_scale = {}
    for scale, analysis in analyses.items():
        detection = analysis.detectability(alpha)
        failure = analysis.failure_probability()["p_failure"]
        coverage = analysis.diagnostic_coverage(alpha, n_boot=500)
        per_scale[f"{scale:g}"] = {
            "false_alarm": detection.false_alarm,
            "false_alarm_interval": detection.false_alarm_interval,
            "n_healthy_for_false_alarm": detection.n_evaluation,
            "yield": 1.0 - float(failure["healthy"]),
            "mean_failure_probability_of_faults": float(failure.drop("healthy").mean()),
            "mean_detection_probability": float(detection.table["p_detect"].mean()),
            "faults_detected": int((detection.table["p_detect"] >= 1 - beta).sum()),
            "faults_simulated": int(detection.table["n_ok"].gt(0).sum()),
            "failed_simulations": int((detection.table["n"] - detection.table["n_ok"]).sum()),
            "diagnostic_coverage": coverage["diagnostic_coverage"],
            "diagnostic_coverage_interval": coverage["interval"],
            "escape_rate": coverage["escape_rate"],
            "false_reject_rate": coverage["false_reject_rate"],
        }
    table = robustness(analyses, alpha, beta)
    shifts = pd.DataFrame(
        {scale: analysis.standardised_shift()["d_prime"] for scale, analysis in analyses.items()}
    )
    failure = pd.DataFrame(
        {scale: analysis.failure_probability()["p_failure"] for scale, analysis in analyses.items()}
    )
    lost = table[(table[min(scales)] >= 1 - beta) & (table[max(scales)] < 1 - beta)]
    result = {
        "experiment": "E, impact of variability",
        "circuit": circuit,
        "alpha": alpha,
        "beta": beta,
        "scales": list(scales),
        "declared_tolerances": study.variations.metadata(),
        "n_faults": len(table),
        "per_scale": per_scale,
        "faults_lost_between_smallest_and_largest_scale": {
            fault: float(row["critical_tolerance"]) for fault, row in lost.iterrows()
        },
    }
    path = save(
        "e_variability",
        circuit,
        result,
        {"detection": table, "standardised_shift": shifts, "failure_probability": failure},
    )
    summary = pd.DataFrame(per_scale).T[
        ["false_alarm", "yield", "mean_detection_probability", "faults_detected",
         "diagnostic_coverage", "escape_rate", "false_reject_rate"]
    ]  # fmt: skip
    print(f"{study.description}: {len(table)} faults\n{summary.to_string()}")
    print(f"faults detected at scale {min(scales):g} and lost at {max(scales):g}: {len(lost)}")
    print("saved to", path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("action", choices=("run", "analyse"))
    parser.add_argument("--circuit", required=True, choices=sorted(validation.STUDIES))
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--scales", type=float, nargs="+", default=list(SCALES))
    parser.add_argument("--samples-per-fault", type=int, default=200)
    parser.add_argument("--healthy", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.action == "run":
        run(
            args.circuit, args.data, args.scales, args.samples_per_fault, args.healthy, args.workers
        )
    else:
        analyse(args.circuit, args.data, args.scales)


if __name__ == "__main__":
    main()
