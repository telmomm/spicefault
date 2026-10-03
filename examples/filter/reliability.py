"""Reliability analysis of an RC low-pass filter: one example of every metric.

    python examples/filter/reliability.py [output folder]

It simulates the same fault campaign at three tolerance scales (about 15,000 short
simulations, a minute or two on eight cores) and prints the metrics of
docs/RELIABILITY_METRICS.md. Rerunning reuses the datasets already written.
"""

import sys
from pathlib import Path

import pandas as pd

from spicefault import Circuit, FaultCampaign, Measurement, SimulationConfig
from spicefault.faults import FaultUniverse, open_rule, parametric_rule, short_rule
from spicefault.reliability import (
    ReliabilityAnalysis,
    collinear_groups,
    local_sensitivity,
    normalised_sensitivity,
    robustness,
    testability_rank,
)
from spicefault.variation import tolerances

HERE = Path(__file__).parent
SCALES = (0.5, 1.0, 2.0)  # of the declared tolerances: 1 % resistors, 5 % capacitor
ALPHA, BETA = 0.01, 0.1

circuit = Circuit.from_netlist(HERE / "rc_lowpass.cir")
config = SimulationConfig(("op", "ac dec 20 1 1e5", "tran 10u 10m"), outputs=("v(out)",))
measurements = [
    Measurement.value("v(out)", name="dc"),
    Measurement.magnitude("v(out)", 100.0, name="gain_100"),
    Measurement.magnitude("v(out)", 1e3, name="gain_1k"),
    Measurement.phase("v(out)", 1e3, name="phase_1k"),
    Measurement.peak("v(out)", name="peak"),
]
universe = FaultUniverse(
    circuit,
    [open_rule(), short_rule(), parametric_rule([-0.5, -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, 0.5])],
)
population = tolerances(circuit, {"R": 0.01, "C": 0.05})


def campaign(scale: float, out: Path) -> FaultCampaign:
    return FaultCampaign(
        circuit,
        universe.selected(),
        out_dir=out / f"scale_{scale:g}",
        samples_per_fault=100,
        healthy_samples=2000,
        variations=population.scaled(scale),
        config=config,
        measurements=measurements,
        seed=42,
        seeding="content",
    )


def main(out: Path) -> None:
    pd.set_option("display.width", 200, "display.max_rows", 100)
    pd.set_option("display.float_format", "{:.3g}".format)
    analyses = {}
    for scale in SCALES:
        run = campaign(scale, out)
        run.run(workers=8, progress=False)
        analyses[scale] = ReliabilityAnalysis.from_dataset(run.out_dir)
        print(f"tolerance scale {scale:g}: {run.status()}")
    analysis = analyses[1.0]

    print("\n--- Fault coverage of the campaign (M10)")
    print(universe.coverage_matrix().to_string(), f"\ncoverage: {universe.coverage():.0%}")
    overlap = universe.selected().tolerance_overlap(population, circuit)
    print("faults partly inside the tolerance band:")
    inside = overlap.loc[overlap["inside_fraction"] > 0, ["fault_id", "inside_fraction"]]
    print(inside.to_string(index=False))

    print("\n--- Detection probability at the declared tolerances (M1), shift (M2), AUC (M3)")
    detection = analysis.detectability(ALPHA)
    low, high = detection.false_alarm_interval
    print(
        f"false-alarm rate on {detection.n_evaluation} held-out healthy samples: "
        f"{detection.false_alarm:.3f} [{low:.3f}, {high:.3f}], target {ALPHA}"
    )
    table = detection.table[["n_ok", "p_detect", "ci_low", "ci_high"]].join(
        [analysis.standardised_shift(), analysis.auc()["auc"]]
    )
    print(table.to_string())

    print("\n--- Minimum detectable deviation (M7b)")
    print(analysis.minimum_detectable(ALPHA, BETA).drop(columns="grid").to_string(index=False))

    print("\n--- Local sensitivity, per 1 % change (M7a)")
    sensitivity = local_sensitivity(circuit, measurements, config)
    print(sensitivity.to_string())
    z = normalised_sensitivity(sensitivity, analysis.response("healthy").spread())
    groups, insensitive = collinear_groups(z)
    print(f"testability rank: {testability_rank(z)}")
    print(f"collinear groups: {groups}; insensitive: {insensitive}")

    print("\n--- Detectability against the tolerance scale (M8)")
    robust = robustness(analyses, ALPHA, BETA)
    print(robust[robust["loss"] != 0].to_string())

    print("\n--- Ambiguity (M9)")
    ambiguity = analysis.ambiguity(threshold=3.0)
    print("not separated from healthy:", ambiguity.undetectable)
    print("confusable components:", ambiguity.confusable)
    for group in ambiguity.groups:
        if len(group) > 1:
            print("  ambiguity group:", group)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "output")
