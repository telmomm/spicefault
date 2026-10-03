"""Faults as objects, a fault universe from rules, and what a campaign covers.

    python examples/02_faults_and_coverage.py

Nothing is simulated here.
"""

import json

from rc_study import circuit, population

from spicefault.faults import (
    CompositeFault,
    Fault,
    FaultSet,
    FaultUniverse,
    LeakageFault,
    OpenCircuit,
    ParametricFault,
    ShortCircuit,
    leakage_rule,
    open_rule,
    parametric_rule,
    short_rule,
)

# --- every fault is a list of primitive changes, with a record ------------------------
faults = FaultSet(
    [
        OpenCircuit("R1"),
        ShortCircuit("C1"),
        LeakageFault("C1", 1e6, r_min=1e4, r_max=1e8),
        ParametricFault("R2", deviation=-0.2),
        # an ageing capacitor: less capacitance and more series resistance, one cause
        CompositeFault(
            "capacitor_ageing",
            [ParametricFault("C1", factor=0.7), OpenCircuit("C1", r_open=50.0)],
            magnitude=0.3,
            unit="capacitance loss",
        ),
    ]
)
for fault in faults:
    severity = "-" if fault.severity is None else f"{fault.severity.value:.2f}"
    print(f"{fault.fault_id:28s} type {fault.fault_type:18s} severity {severity}")

print("\nthe record of one fault:")
record = faults["R2:parametric:-0.2"].metadata()
print(json.dumps(record, indent=2))
print("rebuilt from its record, it is the same fault:", Fault.from_metadata(record) == faults[3])

# --- a universe generated from rules ---------------------------------------------------
universe = FaultUniverse(
    circuit,
    [
        open_rule(),
        short_rule(),
        parametric_rule([-0.2, -0.05, 0.05, 0.2]),
        leakage_rule([1e5, 1e6, 1e7], kinds="C"),
    ],
)
universe.exclude("R2:short", reason="ties the output to ground: same effect as C1:short")
print(f"\n{len(universe)} fault conditions in the universe, {len(universe.selected())} selected")
print("\nconditions to simulate, by component and fault type:")
print(universe.coverage_matrix())
print("\nleft out, with the reason:")
print(universe.exclusions().to_string(index=False))
print(f"\nstructural coverage: {universe.coverage():.0%}")
print("components outside the fault model:", universe.metadata()["components_without_faults"])

# --- faults that are not faults: inside the tolerance band ----------------------------
overlap = universe.selected().tolerance_overlap(population, circuit)
inside = overlap[overlap["inside_fraction"] > 0]
print("\nparametric faults partly inside the tolerance band of their component:")
print(inside[["fault_id", "tolerance", "inside_fraction"]].to_string(index=False))
