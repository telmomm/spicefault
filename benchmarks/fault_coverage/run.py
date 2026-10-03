"""Fault coverage of the campaigns (docs/EXPERIMENT_PLAN.md, experiment D).

    python -m benchmarks.fault_coverage.run

For each validation circuit, the fault universe is generated from its applicability
rules. The result holds the number of components, fault types, magnitudes and fault
conditions, the coverage matrix, what was excluded and why, the components outside
the fault model, and the parametric faults that lie partly inside the tolerance band.
"""

from __future__ import annotations

import argparse

from benchmarks import workloads
from benchmarks.common import environment, save
from spicefault import Circuit
from spicefault.faults import FaultUniverse
from spicefault.variation import VariationSet


def describe(
    circuit: Circuit, universe: FaultUniverse, variations: VariationSet, n_conditions: int = 1
) -> dict:
    matrix = universe.coverage_matrix("universe")
    selected = universe.coverage_matrix("selected")
    overlap = universe.selected().tolerance_overlap(variations, circuit)
    inside = overlap[overlap["inside_fraction"] > 0]
    record = universe.metadata()
    magnitudes = sorted({f.magnitude for f in universe.faults if f.magnitude is not None})
    return {
        "circuit": circuit.name,
        "netlist_sha256": circuit.metadata()["netlist_sha256"],
        "n_components": len(circuit.components()),
        "n_components_with_faults": len(matrix),
        "components_without_faults": record["components_without_faults"],
        "fault_types": record["rules"],
        "n_distinct_magnitudes": len(magnitudes),
        "n_fault_conditions": record["n_universe"],
        "n_selected": record["n_selected"],
        "structural_coverage": record["coverage"],
        "n_operating_conditions": n_conditions,
        "n_simulated_conditions": record["n_selected"] * n_conditions,
        "conditions_per_fault_type": {k: int(v) for k, v in selected.sum().items()},
        "coverage_matrix_universe": {
            c: {k: int(v) for k, v in row.dropna().items()} for c, row in matrix.iterrows()
        },
        "coverage_matrix_selected": {
            c: {k: int(v) for k, v in row.dropna().items()} for c, row in selected.iterrows()
        },
        "exclusions": record["exclusions"],
        "inside_tolerance": {
            "n_fault_conditions": len(inside),
            "faults": dict(
                zip(inside["fault_id"], inside["inside_fraction"].round(4), strict=True)
            ),
        },
    }


def run() -> dict:
    circuits = {}
    rc = workloads.rc_circuit()
    circuits["rc"] = describe(rc, workloads.rc_universe(rc), workloads.rc_tolerances(rc))
    return {"benchmark": "fault_coverage", "environment": environment(), "circuits": circuits}


def report(result: dict) -> str:
    lines = []
    for name, c in result["circuits"].items():
        lines.append(
            f"{name}: {c['n_components_with_faults']} of {c['n_components']} components, "
            f"{len(c['fault_types'])} fault types, {c['n_fault_conditions']} fault conditions, "
            f"coverage {c['structural_coverage']:.0%}, "
            f"{c['inside_tolerance']['n_fault_conditions']} partly inside tolerance"
        )
        lines.append(f"   per fault type: {c['conditions_per_fault_type']}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--label", default="")
    args = parser.parse_args()
    result = run()
    print(report(result))
    print("\nsaved to", save("fault_coverage", result, args.label))


if __name__ == "__main__":
    main()
