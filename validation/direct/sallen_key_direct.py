"""The Sallen-Key fault campaign written directly against ngspice, without spicefault.

    python -m validation.direct.sallen_key_direct --out data/sallen_key_direct

This is the reference for what the framework costs and what it adds: the same
circuit, tolerances, faults, analyses and measurements as `validation/sallen_key.py`,
in the form such a study usually takes, a script of its own. It uses only the
standard library and NumPy. It is kept as it was written, as the frozen baseline of
the comparison (docs/EXPERIMENT_PLAN.md).

What it does not do, by construction: resume an interrupted run, keep a status per
simulation beyond "ok or not", record the definition of the experiment, or give a
sample the same random numbers when the fault list changes.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import tempfile
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

OPAMP = (Path(__file__).resolve().parents[1] / "netlists" / "opamp.lib").read_text()
# name: (node 1, node 2, nominal value, tolerance)
PASSIVES = {
    "R1": ("in", "a", 5.18e3, 0.05),
    "R2": ("a", "out", 1e3, 0.05),
    "R3": ("b", "0", 2e3, 0.05),
    "R4": ("fb", "0", 4e3, 0.05),
    "R5": ("out", "fb", 4e3, 0.05),
    "C1": ("a", "0", 5e-9, 0.10),
    "C2": ("a", "b", 5e-9, 0.10),
}
AOL, GBW, GBW_TOLERANCE = 2e5, 1e7, 0.2
DEVIATIONS = [-0.5, -0.25, -0.1, 0.1, 0.25, 0.5]
R_OPEN, R_SHORT = 1e9, 1.0
FS, DURATION = 500e3, 300e-6
FEATURES = ["centre_frequency", "peak_gain", "bandwidth", "gain_10k", "gain_25k", "gain_60k",
            "phase_25k", "pulse_peak", "pulse_minimum", "pulse_rms"]  # fmt: skip


def fault_list() -> list[tuple[str, str, float]]:
    """(component, kind, level) of every fault, in a fixed order."""
    faults = []
    for name in PASSIVES:
        faults += [(name, "open", 0.0), (name, "short", 0.0)]
        faults += [(name, "parametric", level) for level in DEVIATIONS]
    return faults + [("XU1", "gain_loss", 1e-3), ("XU1", "bandwidth_loss", 1e-2)]


def netlist(values: dict[str, float], aol: float, gbw: float, fault) -> str:
    lines = ["Sallen-Key band-pass filter", OPAMP.rstrip("\n"),
             "Vin in 0 dc 0 ac 1 pulse(0 5 10u 1u 1u 10u 1)"]  # fmt: skip
    for name, (n1, n2, _, _) in PASSIVES.items():
        value = values[name]
        if fault and fault[0] == name:
            if fault[1] == "parametric":
                value *= 1.0 + fault[2]
            elif fault[1] == "open":
                lines.append(f"Rser_{name} {name}_x {n2} {R_OPEN!r}")
                n2 = f"{name}_x"
        lines.append(f"{name} {n1} {n2} {value!r}")
        if fault and fault[0] == name and fault[1] == "short":
            lines.append(f"Rpar_{name} {n1} {n2} {R_SHORT!r}")
    if fault and fault[1] == "gain_loss":
        aol *= fault[2]
    if fault and fault[1] == "bandwidth_loss":
        gbw *= fault[2]
    lines += [
        f"XU1 b fb out opamp aol={aol!r} gbw={gbw!r}",
        ".control", "set noaskquit", "set appendwrite",
        "ac dec 100 1k 1e6", "write out.raw v(out)",
        "tran 0.5u 400u", "write out.raw v(out)",
        ".endc", ".end", "",
    ]  # fmt: skip
    return "\n".join(lines)


def read_raw(path: Path) -> list[dict[str, np.ndarray]]:
    """The plots of an ngspice binary raw file, each as {vector: values}."""
    data, plots, pos = path.read_bytes(), [], 0
    while (idx := data.find(b"Binary:\n", pos)) >= 0:
        header = data[pos:idx].decode("latin-1").splitlines()
        n_vars = n_points = 0
        is_complex, names = False, []
        for i, line in enumerate(header):
            key, _, value = line.partition(":")
            if key == "Flags":
                is_complex = "complex" in value
            elif key == "No. Variables":
                n_vars = int(value)
            elif key == "No. Points":
                n_points = int(value)
            elif key == "Variables":
                names = [h.split()[1].lower() for h in header[i + 1 : i + 1 + n_vars]]
        dtype = np.complex128 if is_complex else np.float64
        start = idx + 8
        block = np.frombuffer(data, dtype, n_points * n_vars, start).reshape(n_points, n_vars)
        plots.append({name: block[:, k] for k, name in enumerate(names)})
        pos = start + n_points * n_vars * np.dtype(dtype).itemsize
    return plots


def at(freq: np.ndarray, h: np.ndarray, f: float) -> complex:
    """Response at f, interpolating log-magnitude and phase over log-frequency."""
    x = np.log10(freq)
    magnitude = np.interp(np.log10(f), x, np.log(np.maximum(np.abs(h), 1e-300)))
    phase = np.interp(np.log10(f), x, np.unwrap(np.angle(h)))
    return complex(np.exp(magnitude) * np.exp(1j * phase))


def measure(ac: dict, tran: dict) -> tuple[dict[str, float], np.ndarray]:
    freq, h = ac["frequency"].real, ac["v(out)"]
    x, y = np.log(freq), np.log(np.maximum(np.abs(h), 1e-300))
    k = int(np.argmax(y))
    x0, y0 = x[k], y[k]
    if 0 < k < len(y) - 1:  # parabola through the highest point and its neighbours
        a, b, c = np.polyfit(x[k - 1 : k + 2] - x[k], y[k - 1 : k + 2], 2)
        if a < 0:
            x0, y0 = x[k] - b / (2 * a), c - b * b / (4 * a)
    level = y0 - 0.5 * np.log(2.0)
    below = y < level
    j = np.flatnonzero(below[:-1] != below[1:])
    crossings = np.exp(x[j] + (level - y[j]) * (x[j + 1] - x[j]) / (y[j + 1] - y[j]))
    lower, upper = crossings[crossings < np.exp(x0)], crossings[crossings > np.exp(x0)]
    low = lower[-1] if len(lower) else freq[0]  # a band edge outside the sweep: its end
    high = upper[0] if len(upper) else freq[-1]
    t, v = tran["time"].real, tran["v(out)"].real
    duration = t[-1] - t[0]
    integrate = getattr(np, "trapezoid", None) or np.trapz
    values = {
        "centre_frequency": float(np.exp(x0)),
        "peak_gain": float(np.exp(y0)),
        "bandwidth": float(high - low),
        "gain_10k": abs(at(freq, h, 10e3)),
        "gain_25k": abs(at(freq, h, 25e3)),
        "gain_60k": abs(at(freq, h, 60e3)),
        "phase_25k": float(np.degrees(np.angle(at(freq, h, 25e3)))),
        "pulse_peak": float(v.max()),
        "pulse_minimum": float(v.min()),
        "pulse_rms": float(np.sqrt(integrate(v * v, t) / duration)),
    }
    samples = np.arange(round(DURATION * FS)) / FS
    return values, np.interp(samples, t, v).astype(np.float32)


def simulate(task: tuple[int, int, int, tuple | None, float]):
    """One simulation: (row, waveform). The row says `ok` False if anything went wrong."""
    sample_id, seed, replica, fault, scale = task
    rng = np.random.default_rng(seed + sample_id)
    values = {
        name: nominal * (1.0 + scale * tolerance * rng.uniform(-1.0, 1.0))
        for name, (_, _, nominal, tolerance) in PASSIVES.items()
    }
    gbw = GBW * (1.0 + scale * GBW_TOLERANCE * rng.uniform(-1.0, 1.0))
    row = {
        "sample_id": sample_id,
        "fault": "healthy" if fault is None else f"{fault[0]}:{fault[1]}:{fault[2]:+g}",
        "replica": replica,
        "ok": False,
        **{f"p_{name}": value for name, value in values.items()},
        "p_gbw": gbw,
        **dict.fromkeys(FEATURES, float("nan")),
    }
    try:
        with tempfile.TemporaryDirectory(prefix="direct_") as tmp:
            (Path(tmp) / "deck.cir").write_text(netlist(values, AOL, gbw, fault))
            subprocess.run(["ngspice", "-b", "-n", "deck.cir"], cwd=tmp, capture_output=True,
                           timeout=120)  # fmt: skip
            ac, tran = read_raw(Path(tmp) / "out.raw")
        measured, waveform = measure(ac, tran)
        if not all(np.isfinite(list(measured.values()))):
            raise ValueError("not finite")
    except Exception:
        return row, None
    return {**row, **measured, "ok": True}, waveform


def tasks(healthy: int, per_fault: int, seed: int, scale: float) -> list[tuple]:
    todo = [(None, r) for r in range(healthy)]
    todo += [(fault, r) for fault in fault_list() for r in range(per_fault)]
    return [(i, seed, replica, fault, scale) for i, (fault, replica) in enumerate(todo)]


def run(out: Path, healthy: int, per_fault: int, workers: int, seed: int = 42,
        scale: float = 1.0) -> float:  # fmt: skip
    """Run the campaign and write `samples.csv` and `waveforms.npy`; returns the seconds taken."""
    out.mkdir(parents=True, exist_ok=True)
    todo = tasks(healthy, per_fault, seed, scale)
    start = time.perf_counter()
    with Pool(workers) as pool:
        results = pool.map(simulate, todo, chunksize=16)
    waveforms = np.full((len(todo), round(DURATION * FS)), np.nan, dtype=np.float32)
    with (out / "samples.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0][0]))
        writer.writeheader()
        for i, (row, waveform) in enumerate(results):
            writer.writerow(row)
            if waveform is not None:
                waveforms[i] = waveform
    np.save(out / "waveforms.npy", waveforms)
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--healthy", type=int, default=5000)
    parser.add_argument("--samples-per-fault", type=int, default=200)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seconds = run(Path(args.out), args.healthy, args.samples_per_fault, args.workers, args.seed)
    n = args.healthy + args.samples_per_fault * len(fault_list())
    print(f"{n} simulations in {seconds:.1f} s ({n / seconds:.1f} per second)")


if __name__ == "__main__":
    main()
