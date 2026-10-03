"""The workloads of the benchmarks.

- `rc`: an RC low-pass filter with 30 faults. It needs nothing but ngspice.
- `ecg`: the self-test measurements of the ECG front-end study, the realistic case. It
  needs the `ecgfd` package, and comes with the same work done by the code of that
  study, as the baseline to compare the framework with.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

from spicefault import Circuit, Experiment, Measurement, SimulationConfig, Waveform
from spicefault.faults import FaultUniverse, open_rule, parametric_rule, short_rule
from spicefault.variation import tolerances

REPO = Path(__file__).resolve().parents[1]
WORKLOADS = ("rc", "ecg", "ecg-reference")
HAS_ECGFD = importlib.util.find_spec("ecgfd") is not None
# the ECG study written with spicefault lives with the equivalence tests for now
sys.path.insert(0, str(REPO / "tests" / "regression"))

RC_DEVIATIONS = [-0.5, -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, 0.5]


def rc_circuit() -> Circuit:
    return Circuit.from_netlist(REPO / "examples" / "filter" / "rc_lowpass.cir")


def rc_universe(circuit: Circuit | None = None) -> FaultUniverse:
    circuit = circuit or rc_circuit()
    return FaultUniverse(circuit, [open_rule(), short_rule(), parametric_rule(RC_DEVIATIONS)])


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
        variations=tolerances(circuit, {"R": 0.01, "C": 0.05}),
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
    )


def ecg_config(circuit: str, n_samples: int) -> dict:
    """Configuration of the ECG study with one sample per fault and the rest healthy."""
    from ecgfd.config import DEFAULT_CONFIG, load_config
    from ecgfd.faults import fault_catalogue
    from ecgfd.specs import with_nominal_gain

    cfg = with_nominal_gain(load_config(DEFAULT_CONFIG, circuit))
    n_faults = len(fault_catalogue(cfg))
    cfg["dataset"] = {"n_healthy": max(n_samples - n_faults, 1), "n_per_fault": 1}
    return cfg


def ecg_experiment(circuit: str, n_samples: int) -> Experiment:
    from ecg_adapter import service_experiment

    return service_experiment(ecg_config(circuit, n_samples))


def experiment(workload: str, n_samples: int) -> Experiment:
    if workload == "rc":
        return rc_experiment(n_samples)
    if workload in ("ecg", "ecg-reference"):
        if not HAS_ECGFD:
            raise RuntimeError("the ECG workload needs the ecgfd package")
        return ecg_experiment("integrated" if workload == "ecg" else "reference", n_samples)
    raise ValueError(f"unknown workload {workload!r}; available: {WORKLOADS}")


# --- the same ECG work, done by the code of the ECG study -------------------------------


def ecg_baseline_tasks(cfg: dict) -> list:
    from ecgfd.dataset import build_tasks

    return build_tasks(cfg)


def ecg_baseline_worker(task, cfg: dict) -> tuple[dict, np.ndarray | None]:
    """One case with `ecgfd`: draw, inject, simulate the self-test and extract its features.

    The first of the two ngspice runs of `ecgfd.dataset.simulate_task`, which is the
    work the spicefault experiment does.
    """
    from ecgfd.dataset import sample_rng
    from ecgfd.sampling import instance_parameters, sample_instance
    from ecgfd.simulate import measure, scalar_features
    from ecgfd.spice import SimulationError

    index, fault, replica = task
    inst = fault.apply(sample_instance(cfg, sample_rng(cfg, index, replica)), cfg)
    row = {
        "condition": fault.id,
        "condition_index": index,
        "kind": fault.kind,
        "target": fault.target,
        "level": fault.level,
        "replica": replica,
        "sim_ok": True,
        **instance_parameters(inst),
    }
    try:
        m = measure(inst, cfg)
    except SimulationError:
        row["sim_ok"] = False
        return row, None
    row.update(scalar_features(m, cfg))
    return row, m.pulse.astype(np.float32)
