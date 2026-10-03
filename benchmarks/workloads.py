"""The workloads of the benchmarks.

For now there is one, an RC low-pass filter with 30 faults. It only checks that the
benchmarks work: a simulation takes about 15 ms, so starting processes weighs as much
as simulating. The validation circuits of docs/SCIENTIFIC_SCOPE.md, section 8, are to
be added here as they are written.
"""

from __future__ import annotations

from pathlib import Path

from spicefault import Circuit, Experiment, Measurement, SimulationConfig, VariationSet, Waveform
from spicefault.faults import FaultUniverse, open_rule, parametric_rule, short_rule
from spicefault.variation import tolerances

REPO = Path(__file__).resolve().parents[1]
WORKLOADS = ("rc",)

RC_DEVIATIONS = [-0.5, -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, 0.5]


def rc_circuit() -> Circuit:
    return Circuit.from_netlist(REPO / "examples" / "filter" / "rc_lowpass.cir")


def rc_universe(circuit: Circuit | None = None) -> FaultUniverse:
    circuit = circuit or rc_circuit()
    return FaultUniverse(circuit, [open_rule(), short_rule(), parametric_rule(RC_DEVIATIONS)])


def rc_tolerances(circuit: Circuit | None = None) -> VariationSet:
    return tolerances(circuit or rc_circuit(), {"R": 0.01, "C": 0.05})


def rc_experiment(n_samples: int, seed: int = 42) -> Experiment:
    """About `n_samples` simulations: half healthy, half spread over the 30 faults."""
    circuit = rc_circuit()
    faults = rc_universe(circuit).selected()
    per_fault = max(n_samples // (2 * len(faults)), 1)
    return Experiment(
        circuit,
        faults=faults,
        samples=per_fault,
        healthy_samples=max(n_samples - per_fault * len(faults), 1),
        variations=rc_tolerances(circuit),
        config=SimulationConfig(("op", "ac dec 20 1 1e5", "tran 10u 10m"), outputs=("v(out)",)),
        measurements=[
            Measurement.value("v(out)", name="dc"),
            Measurement.magnitude("v(out)", 100.0, name="gain_100"),
            Measurement.magnitude("v(out)", 1e3, name="gain_1k"),
            Measurement.phase("v(out)", 1e3, name="phase_1k"),
            Measurement.peak("v(out)", name="peak"),
        ],
        waveform=Waveform("v(out)", fs=10e3, duration=10e-3),
        seed=seed,
        seeding="content",
    )


def experiment(workload: str, n_samples: int) -> Experiment:
    if workload == "rc":
        return rc_experiment(n_samples)
    raise ValueError(f"unknown workload {workload!r}; available: {WORKLOADS}")
