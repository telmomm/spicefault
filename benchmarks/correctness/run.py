"""Correctness against ngspice run directly (docs/EXPERIMENT_PLAN.md, experiment A).

    python -m benchmarks.correctness.run

For each validation circuit, the fault-free circuit and every fault of its universe
are realised once, at the declared tolerances. The deck of each sample is simulated
twice: through `spicefault` (`Simulator.run`), and with `ngspice -b` launched here,
whose raw file is read by the reader of this module, which shares no code with the
library. The vectors of the two are compared bit by bit.

This compares the execution and the reading of the output, on the deck the library
built. That the library builds the right netlist for a fault is checked elsewhere,
against the direct script (tests/validation/test_direct_script.py).
"""

from __future__ import annotations

import argparse
import struct
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

import validation
from benchmarks import workloads
from benchmarks.common import environment, save
from spicefault import Experiment
from spicefault.simulation.backend import build_deck
from spicefault.simulation.ngspice import ngspice_path

WORKLOADS = tuple(sorted(validation.STUDIES))
SEED = 42
MARK = b"Binary:\n"


def read_raw(path: Path) -> list[tuple[str, dict[str, np.ndarray]]]:
    """Independent reader of an ngspice binary raw file: [(plot name, {vector: values})]."""
    data = path.read_bytes()
    plots, position = [], 0
    while (cut := data.find(MARK, position)) >= 0:
        fields, names = {}, []
        lines = data[position:cut].decode("latin-1").split("\n")
        for number, line in enumerate(lines):
            if line.startswith("Variables:"):
                names = [row.split("\t")[2].lower() for row in lines[number + 1 :] if row.strip()]
                break
            key, _, value = line.partition(":")
            fields[key.strip()] = value.strip()
        n_points, n_vars = int(fields["No. Points"]), int(fields["No. Variables"])
        is_complex = "complex" in fields["Flags"]
        width = 16 if is_complex else 8
        start = cut + len(MARK)
        vectors: dict[str, list] = {name: [] for name in names}
        for point in range(n_points):
            for column, name in enumerate(names):
                offset = start + (point * n_vars + column) * width
                if is_complex:
                    vectors[name].append(complex(*struct.unpack_from("<dd", data, offset)))
                else:
                    vectors[name].append(struct.unpack_from("<d", data, offset)[0])
        kind = complex if is_complex else float
        plots.append((fields["Plotname"], {k: np.array(v, dtype=kind) for k, v in vectors.items()}))
        position = start + n_points * n_vars * width
    return plots


def run_directly(deck: str) -> list[tuple[str, dict[str, np.ndarray]]] | None:
    """The plots of a deck run with `ngspice -b`, or None if it wrote no raw file."""
    with tempfile.TemporaryDirectory(prefix="direct_") as folder:
        (Path(folder) / "deck.cir").write_text(deck)
        subprocess.run(
            [ngspice_path(), "-b", "-n", "deck.cir"], cwd=folder, capture_output=True, timeout=120
        )
        raw = Path(folder) / "out.raw"
        return read_raw(raw) if raw.exists() else None


def identical_values(a: np.ndarray, b: np.ndarray) -> int:
    """The number of values of `a` with the same bits as in `b` (both parts, if complex)."""
    a, b = (np.ascontiguousarray(x).view(np.uint64).reshape(x.size, -1) for x in (a, b))
    return int((a == b).all(axis=1).sum())


def _experiment(workload: str) -> Experiment:
    """One healthy sample and one sample of every fault, with the seeding of the campaigns."""
    if workload == "rc":
        return workloads.rc_experiment(len(workloads.rc_universe()) + 1, seed=SEED)
    return validation.get(workload).experiment(samples_per_fault=1, healthy_samples=1, seed=SEED)


def compare(workload: str) -> dict:
    experiment = _experiment(workload)
    groups: dict[str, dict] = defaultdict(
        lambda: {"decks": 0, "status": defaultdict(int), "vectors": 0, "values": 0,
                 "values_bit_identical": 0, "max_abs_diff": 0.0, "plot_mismatch": 0,
                 "direct_without_output": 0}
    )  # fmt: skip
    for sample in experiment.plan():
        fault = experiment.fault(sample)
        group = groups["healthy" if fault is None else fault.fault_type]
        realised = experiment.realise(sample)
        config = experiment.config_for(sample.condition_index)
        ours = experiment.simulator.run(realised.netlist, config)
        theirs = run_directly(build_deck(realised.netlist, config))
        group["decks"] += 1
        group["status"][ours.status.value] += 1
        if theirs is None:
            group["direct_without_output"] += 1
            continue
        if not ours.plots:
            continue  # a failed simulation keeps no vectors to compare
        if [p.name for p in ours.plots] != [name for name, _ in theirs]:
            group["plot_mismatch"] += 1
            continue
        for plot, (_, vectors) in zip(ours.plots, theirs, strict=True):
            if list(plot.vectors) != list(vectors):
                group["plot_mismatch"] += 1
                continue
            for name, b in vectors.items():
                a = np.asarray(plot.vectors[name])
                if a.shape != b.shape or a.dtype != b.dtype:
                    group["plot_mismatch"] += 1
                    continue
                group["vectors"] += 1
                group["values"] += a.size
                group["values_bit_identical"] += identical_values(a, b)
                if a.size:
                    group["max_abs_diff"] = max(group["max_abs_diff"], float(np.abs(a - b).max()))
    by_type = {name: {**group, "status": dict(group["status"])} for name, group in groups.items()}
    total = {
        key: sum(group[key] for group in by_type.values())
        for key in ("decks", "vectors", "values", "values_bit_identical", "plot_mismatch",
                    "direct_without_output")
    }  # fmt: skip
    total["max_abs_diff"] = max(group["max_abs_diff"] for group in by_type.values())
    return {
        "n_components": len(experiment.circuit.components()),
        "n_faults": len(experiment.faults),
        "total": total,
        "by_fault_type": by_type,
    }


def run(circuits: tuple[str, ...] = WORKLOADS) -> dict:
    compared = {name: compare(name) for name in circuits}
    totals = [circuit["total"] for circuit in compared.values()]
    return {
        "benchmark": "correctness",
        "environment": environment(),
        "protocol": {"seed": SEED, "seeding": "content", "tolerance_scale": 1.0,
                     "samples_per_fault": 1, "healthy_samples": 1},  # fmt: skip
        "circuits": compared,
        "verdict": {
            "decks": sum(t["decks"] for t in totals),
            "values": sum(t["values"] for t in totals),
            "bit_identical": all(
                t["values"] == t["values_bit_identical"] > 0
                and not t["plot_mismatch"]
                and not t["direct_without_output"]
                for t in totals
            ),
            "max_abs_diff": max(t["max_abs_diff"] for t in totals),
        },
    }


def report(result: dict) -> str:
    lines = []
    for name, circuit in result["circuits"].items():
        for fault_type, group in circuit["by_fault_type"].items():
            lines.append(
                f"{name:11s} {fault_type:16s} decks={group['decks']:3d} "
                f"status={group['status']} values={group['values']} "
                f"identical={group['values_bit_identical']} max|d|={group['max_abs_diff']:g} "
                f"mismatch={group['plot_mismatch']} no_output={group['direct_without_output']}"
            )
    verdict = result["verdict"]
    lines.append(
        f"{verdict['decks']} decks, {verdict['values']} values: "
        + ("all bit-identical" if verdict["bit_identical"] else "NOT all bit-identical")
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--circuits", nargs="+", default=WORKLOADS, choices=workloads.WORKLOADS)
    parser.add_argument("--label", default="")
    args = parser.parse_args()
    result = run(tuple(args.circuits))
    print(report(result))
    print("\nsaved to", save("correctness", result, args.label))


if __name__ == "__main__":
    main()
