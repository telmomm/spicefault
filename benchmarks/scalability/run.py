"""Runtime and parallel scalability (docs/EXPERIMENT_PLAN.md, experiment C).

    python -m benchmarks.scalability.run --workers 1 2 4 8

A fixed workload is simulated to disk with each number of workers, several times. The
result holds every timing, and per number of workers the median throughput, the
speed-up S(N) = T1 / TN and the efficiency E(N) = S(N) / N. It also holds the time of
one sample split by phase, the peak memory, the bytes written and the cost of resuming
a campaign.

With the `sallen_key` workload, the same campaign is also run by a script written
directly against ngspice (`validation/direct/sallen_key_direct.py`), to measure what
the framework costs.

The protocol asks for an idle machine on mains power, at least 5 repetitions and 2000
samples; those are the defaults. Smaller runs are for checking that the benchmark works.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks import workloads
from benchmarks.common import (
    Timer,
    environment,
    folder_bytes,
    load_average,
    peak_memory_mb,
    save,
    spread,
    wait_until_quiet,
)
from spicefault.experiments import run_campaign, run_chunks, simulate_sample
from spicefault.simulation import RAW_NAME, build_deck, ngspice_path, parse_raw


def timed_campaign(tasks, worker, context, n_points, workers: int, chunk: int) -> dict:
    """Simulate the tasks to a dataset in a scratch folder; wall time, throughput, bytes."""
    with tempfile.TemporaryDirectory(prefix="spicefault_bench_") as tmp:
        with Timer() as timer:
            out = run_campaign(
                tasks, worker, context, Path(tmp) / "data", config={}, n_points=n_points,
                workers=workers, chunk=chunk, progress=False,
            )  # fmt: skip
        return {
            "wall_s": timer.seconds,
            "sims_per_s": len(tasks) / timer.seconds,
            "bytes_written": folder_bytes(out),
            "samples": pd.read_parquet(out / "samples.parquet"),
        }


def phase_breakdown(experiment, n: int) -> dict:
    """Mean time of one sample by phase, in this process, over `n` samples spread over the plan."""
    plan = experiment.plan()
    picked = plan[:: max(len(plan) // n, 1)][:n]
    times = {"netlist_generation": 0.0, "simulator_process": 0.0, "output_parsing": 0.0,
             "measurement": 0.0}  # fmt: skip
    rows, waveforms = [], []
    exe = ngspice_path()
    for sample in picked:
        t0 = time.perf_counter()
        deck = build_deck(experiment.realise(sample).netlist, experiment.config)
        t1 = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="spicefault_bench_") as tmp:
            (Path(tmp) / "deck.cir").write_text(deck)
            subprocess.run([exe, "-b", "-n", "deck.cir"], cwd=tmp, capture_output=True)
            t2 = time.perf_counter()
            plots = parse_raw(Path(tmp) / RAW_NAME)
        t3 = time.perf_counter()
        result = type("Result", (), {"plots": tuple(plots), "plot": _plot})()
        rows.append({m.name: m(result) for m in experiment.measurements})
        if experiment.waveform is not None:
            waveforms.append(experiment.waveform(result))
        t4 = time.perf_counter()
        for name, dt in zip(times, (t1 - t0, t2 - t1, t3 - t2, t4 - t3), strict=True):
            times[name] += dt
    with tempfile.TemporaryDirectory(prefix="spicefault_bench_") as tmp, Timer() as timer:
        pd.DataFrame(rows).to_parquet(Path(tmp) / "part.parquet", index=False)
        if waveforms:
            np.save(Path(tmp) / "part.npy", np.array(waveforms))
    times["storage"] = timer.seconds
    per_sample = {name: 1e3 * seconds / len(picked) for name, seconds in times.items()}
    total = sum(per_sample.values())
    return {
        "n_samples": len(picked),
        "ms_per_sample": per_sample,
        "fraction": {name: value / total for name, value in per_sample.items()},
    }


def _plot(self, which):
    """`SimulationResult.plot`, for the bare result used in the breakdown."""
    from spicefault.simulation import SimulationResult, SimulationStatus

    return SimulationResult(SimulationStatus.SUCCESS, self.plots).plot(which)


def resume_overhead(experiment, n_points, workers: int, chunk: int) -> dict:
    """Cost of launching again a campaign whose chunks are all complete: nothing is
    simulated, so this is the time to find the chunks, start the workers and write the
    dataset. An interruption loses at most the chunk that was being simulated.
    """
    plan = experiment.plan()
    with tempfile.TemporaryDirectory(prefix="spicefault_bench_") as tmp:
        out = Path(tmp) / "data"
        run_chunks(plan, simulate_sample, experiment, out / "parts", n_points, workers, chunk,
                   progress=False)  # fmt: skip
        shutil.copytree(out / "parts", Path(tmp) / "parts_copy")
        with Timer() as timer:
            # the key of the configuration is written by the first, interrupted, launch
            run_campaign(plan, simulate_sample, experiment, out, config={}, n_points=n_points,
                         workers=workers, chunk=chunk, progress=False)  # fmt: skip
    return {"seconds": timer.seconds, "simulations_repeated": 0, "max_lost_on_interruption": chunk}


def timed_direct(workload: str, n_samples: int, n_workers: int) -> dict:
    """The same campaign with the direct script: wall time and throughput.

    The script does the same simulations and measurements, and writes a CSV and the
    waveforms. It keeps no status, no manifest and cannot resume.
    """
    from validation.direct import sallen_key_direct

    per_fault, healthy = workloads.study_sizes(workload, n_samples)
    n = healthy + per_fault * len(sallen_key_direct.fault_list())
    with tempfile.TemporaryDirectory(prefix="spicefault_bench_") as tmp:
        wall = sallen_key_direct.run(Path(tmp) / "data", healthy, per_fault, n_workers)
        return {"wall_s": wall, "sims_per_s": n / wall, "bytes_written": folder_bytes(tmp)}


def run(
    workload: str = "rc",
    workers: tuple[int, ...] = (1, 2, 4, 8),
    repetitions: int = 5,
    n_samples: int = 2000,
    chunk: int = 500,
    settle: float = 240.0,
) -> dict:
    experiment = workloads.experiment(workload, n_samples)
    experiment.check_picklable()
    plan = experiment.plan()
    n_points = None if experiment.waveform is None else experiment.waveform.n_points
    implementations = {"spicefault": (plan, simulate_sample, experiment)}

    # a job that has just ended is still in the load average: let it fade before timing
    quiet = wait_until_quiet(settle)
    result = {
        "benchmark": "scalability",
        "environment": environment(),
        "machine_before_start": quiet,
        "protocol": {
            "workload": workload,
            "n_samples": len(plan),
            "workers": list(workers),
            "repetitions": repetitions,
            "chunk": chunk,
            "warm_up": "one discarded run of each implementation",
        },
        "runs": [],
    }
    compare = workload == "sallen_key"  # the circuit the direct script was written for
    for tasks, worker, context in implementations.values():  # warm-up, discarded
        timed_campaign(tasks[: min(len(tasks), 40)], worker, context, n_points, workers[-1], chunk)
    if compare:
        timed_direct(workload, 120, workers[-1])
    names = [*implementations, *(["direct_script"] if compare else [])]
    for repetition in range(repetitions):
        for n_workers in workers:
            # both implementations one after the other, so that whatever else the machine
            # is doing affects them alike
            for name in names:
                load_before = load_average()
                if name == "direct_script":
                    measured = timed_direct(workload, len(plan), n_workers)
                else:
                    tasks, worker, context = implementations[name]
                    measured = timed_campaign(tasks, worker, context, n_points, n_workers, chunk)
                    measured.pop("samples")
                result["runs"].append(
                    {
                        "implementation": name,
                        "workers": n_workers,
                        "repetition": repetition,
                        "load_average_before": load_before,
                        **measured,
                    }
                )

    runs = pd.DataFrame(result["runs"])
    summary = {}
    for name in names:
        table = {}
        mine = runs[runs["implementation"] == name]
        t1 = float(mine.loc[mine["workers"] == workers[0], "wall_s"].median()) * workers[0]
        for n_workers in workers:
            walls = mine.loc[mine["workers"] == n_workers, "wall_s"].tolist()
            speedup = t1 / float(np.median(walls))
            table[str(n_workers)] = {
                "wall_s": spread(walls),
                # how much the repetitions differ: the evidence of an undisturbed machine
                "relative_range": relative_range(spread(walls)),
                "sims_per_s": spread([len(plan) / w for w in walls]),
                "speedup": speedup,
                "efficiency": speedup / n_workers,
            }
        summary[name] = table
    result["summary"] = summary
    result["bytes_written"] = int(
        runs.loc[runs["implementation"] == "spicefault", "bytes_written"].iloc[-1]
    )
    result["bytes_per_sample"] = result["bytes_written"] / len(plan)

    if compare:
        # medians, and the fastest run of each: the least disturbed by anything else
        result["relative_throughput"] = {
            str(n): summary["spicefault"][str(n)]["sims_per_s"]["median"]
            / summary["direct_script"][str(n)]["sims_per_s"]["median"]
            for n in workers
        }
        result["relative_throughput_of_fastest_runs"] = {
            str(n): summary["direct_script"][str(n)]["wall_s"]["min"]
            / summary["spicefault"][str(n)]["wall_s"]["min"]
            for n in workers
        }

    result["phase_breakdown"] = phase_breakdown(experiment, min(100, len(plan)))
    result["resume_overhead"] = resume_overhead(experiment, n_points, workers[-1], chunk)
    result["peak_memory"] = peak_memory_mb()
    return result


def relative_range(wall: dict) -> float:
    """Slowest minus fastest repetition, over the median."""
    return (wall["max"] - wall["min"]) / wall["median"]


def report(result: dict) -> str:
    protocol = result["protocol"]
    lines = [
        f"workload {protocol['workload']}, {protocol['n_samples']} samples, "
        f"{protocol['repetitions']} repetitions"
    ]
    for name, table in result["summary"].items():
        lines.append(f"\n{name}\n workers   wall [s]   range   sims/s   speed-up   efficiency")
        for n, row in table.items():
            lines.append(
                f" {n:>7} {row['wall_s']['median']:>10.2f} {relative_range(row['wall_s']):>6.0%} "
                f"{row['sims_per_s']['median']:>8.1f} "
                f"{row['speedup']:>10.2f} {row['efficiency']:>12.2f}"
            )
    if "relative_throughput" in result:
        ratio = ", ".join(f"{n}: {r:.3f}" for n, r in result["relative_throughput"].items())
        lines.append(
            f"\nthroughput of spicefault relative to the direct script, by workers: {ratio}"
        )
        fastest = result["relative_throughput_of_fastest_runs"]
        ratio = ", ".join(f"{n}: {r:.3f}" for n, r in fastest.items())
        lines.append(f"the same from the fastest run of each: {ratio}")
    before = result.get("machine_before_start")
    if before is not None and not before["quiet"]:
        lines.append(
            f"\nNOTE: after waiting {before['waited_s']:.0f} s the load average was still "
            f"{before['load_average_1min']:.1f} (quiet is below {before['quiet_below']:.1f}). "
            "Something else was using the machine, or its load average rests high. Judge the "
            "timings by the range between repetitions, and do not report them if it is wide."
        )
    phases = result["phase_breakdown"]["ms_per_sample"]
    lines.append(
        "\ntime of one sample [ms]: " + ", ".join(f"{k} {v:.2f}" for k, v in phases.items())
    )
    lines.append(f"resume overhead: {result['resume_overhead']['seconds']:.2f} s")
    lines.append(f"peak memory [MB]: {result['peak_memory']}")
    lines.append(f"bytes per sample: {result['bytes_per_sample']:.0f}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--workload", default="rc", choices=workloads.WORKLOADS)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--chunk", type=int, default=500)
    parser.add_argument(
        "--settle", type=float, default=240.0, help="seconds to wait for a quiet machine"
    )
    parser.add_argument("--label", default="", help="added to the name of the result file")
    args = parser.parse_args()
    result = run(
        args.workload, tuple(args.workers), args.repetitions, args.samples, args.chunk, args.settle
    )
    print(report(result))
    print("\nsaved to", save("scalability", result, args.label or args.workload))


if __name__ == "__main__":
    main()
