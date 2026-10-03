"""Experiment F: does detectability depend on the operating conditions?

    python -m validation.experiments.f_conditions run --workers 8
    python -m validation.experiments.f_conditions analyse

On the voltage regulator, whose devices depend on temperature. `run` simulates each
drawn circuit, with each fault, under every operating condition of a set: `one_factor`
(input voltage, load and temperature changed one at a time) or `corners`. `analyse`
judges each condition against its own healthy reference and writes the detection
probability of every fault under every condition, its worst case, and the conditions
under which a fault detected at the nominal one stops being detected.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import validation
from spicefault.reliability import detectability_across

from .common import ALPHA, BETA, DATA, analysis_of, save

CIRCUIT = "regulator"


def folder(data: Path, circuit: str, conditions: str) -> Path:
    return data / circuit / f"conditions_{conditions}"


def run(data, conditions, samples_per_fault, healthy, workers, circuit=CIRCUIT) -> None:
    study = validation.get(circuit)
    campaign = study.campaign(
        folder(data, circuit, conditions),
        samples_per_fault=samples_per_fault,
        healthy_samples=healthy,
        conditions=conditions,
    )
    campaign.run(workers=workers, progress=False)
    print(f"{circuit}, conditions {conditions}: {campaign.status()}")


def analyse(data, conditions, circuit=CIRCUIT, alpha=ALPHA, beta=BETA) -> dict:
    study = validation.get(circuit)
    parts = analysis_of(study, folder(data, circuit, conditions)).by("condition")
    table = detectability_across(parts, alpha, beta)
    names = list(parts)
    per_condition = {}
    for name, analysis in parts.items():
        detection = analysis.detectability(alpha)
        coverage = analysis.diagnostic_coverage(alpha, n_boot=500)
        failure = analysis.failure_probability()["p_failure"]
        per_condition[name] = {
            "false_alarm": detection.false_alarm,
            "yield": 1.0 - float(failure["healthy"]),
            "faults_detected": int((detection.table["p_detect"] >= 1 - beta).sum()),
            "failed_simulations": int((detection.table["n"] - detection.table["n_ok"]).sum()),
            "diagnostic_coverage": coverage["diagnostic_coverage"],
            "escape_rate": coverage["escape_rate"],
        }
    lost = {fault: conditions_ for fault, conditions_ in table["lost"].items() if conditions_}
    result = {
        "experiment": "F, impact of operating conditions",
        "circuit": circuit,
        "alpha": alpha,
        "beta": beta,
        "conditions": [c.metadata() for c in study.conditions[conditions]],
        "reference_condition": names[0],
        "n_faults": len(table),
        "per_condition": per_condition,
        "faults_detected_at_reference_and_lost_elsewhere": lost,
        "largest_range_of_detection_probability": float(table["range"].max()),
    }
    path = save("f_conditions", f"{circuit}_{conditions}", result,
                {"detection": table.drop(columns="lost")})  # fmt: skip
    print(f"{study.description}: {len(table)} faults, {len(names)} conditions")
    print(pd.DataFrame(per_condition).T.to_string())
    print(f"faults detected at '{names[0]}' and lost under another condition: {len(lost)}")
    print("saved to", path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("action", choices=("run", "analyse"))
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--conditions", default="one_factor", choices=("one_factor", "corners"))
    parser.add_argument("--samples-per-fault", type=int, default=200)
    parser.add_argument("--healthy", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.action == "run":
        run(args.data, args.conditions, args.samples_per_fault, args.healthy, args.workers)
    else:
        analyse(args.data, args.conditions)


if __name__ == "__main__":
    main()
