"""What the benchmarks share: the record of the environment and the result files."""

from __future__ import annotations

import json
import os
import platform
import resource
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import spicefault
from spicefault.simulation import ngspice_version

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "benchmarks" / "results"


def _run(*command: str, cwd: Path | None = None) -> str:
    try:
        return subprocess.run(command, capture_output=True, text=True, cwd=cwd).stdout.strip()
    except OSError:
        return ""


def _cpu() -> dict:
    """Processor model and core counts, as far as the platform tells."""
    info = {"model": platform.processor() or platform.machine(), "logical_cores": os.cpu_count()}
    if sys.platform == "darwin":
        info["model"] = _run("sysctl", "-n", "machdep.cpu.brand_string") or info["model"]
        for key, name in (
            ("physical_cores", "hw.physicalcpu"),
            ("performance_cores", "hw.perflevel0.physicalcpu"),
            ("efficiency_cores", "hw.perflevel1.physicalcpu"),
        ):
            value = _run("sysctl", "-n", name)
            if value.isdigit():
                info[key] = int(value)
    elif Path("/proc/cpuinfo").exists():
        text = Path("/proc/cpuinfo").read_text()
        lines = [line for line in text.splitlines() if "model name" in line]
        models = [line.split(":", 1)[1].strip() for line in lines]
        if models:
            info["model"] = models[0]
    return info


def environment() -> dict:
    """Everything docs/EXPERIMENT_PLAN.md, section 4, asks to record with a run."""
    commit = _run("git", "rev-parse", "HEAD", cwd=REPO)
    dirty = bool(_run("git", "status", "--porcelain", "--untracked-files=no", cwd=REPO))
    load = os.getloadavg() if hasattr(os, "getloadavg") else (None, None, None)
    return {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "cpu": _cpu(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "spicefault_version": spicefault.__version__,
        "spicefault_commit": commit,
        "uncommitted_changes": dirty,
        "simulator": ngspice_version(),
        # the protocol asks for an idle machine; this is the evidence available
        "load_average_1min_at_start": load[0],
    }


def load_average() -> float | None:
    return os.getloadavg()[0] if hasattr(os, "getloadavg") else None


def wait_until_quiet(timeout: float = 240.0, poll: float = 5.0) -> dict:
    """Wait for the 1-minute load average to fall below 0.4 per core, at most `timeout` s.

    The load average remembers the last minute. A benchmark launched right after
    another job would start with that job's load in it, and be told that the machine
    is busy when it no longer is. Returns the seconds waited and the load reached.
    """
    quiet = 0.4 * (os.cpu_count() or 1)
    start = time.perf_counter()
    while (load := load_average()) is not None and load > quiet:
        if time.perf_counter() - start >= timeout:
            break
        time.sleep(poll)
    return {
        "waited_s": round(time.perf_counter() - start, 1),
        "load_average_1min": load,
        "quiet_below": quiet,
        "quiet": load is None or load <= quiet,
    }


def peak_memory_mb() -> dict:
    """Peak resident memory of this process and of its largest finished child process."""
    scale = 1 / 2**20 if sys.platform == "darwin" else 1 / 2**10  # bytes on macOS, kB on Linux
    return {
        "parent_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale, 1),
        "largest_child_mb": round(
            resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * scale, 1
        ),
    }


def folder_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.start


def spread(values: list[float]) -> dict:
    """Median and range, as the protocol reports repeated timings."""
    return {
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "n": len(values),
    }


def save(benchmark: str, result: dict, label: str = "") -> Path:
    """Write a result as JSON under benchmarks/results/<benchmark>/ and return its path."""
    folder = RESULTS / benchmark
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = folder / f"{stamp}{'_' + label if label else ''}.json"
    path.write_text(json.dumps(result, indent=2, default=_plain))
    return path


def _plain(value):
    """JSON for the NumPy and pandas values that end up in results."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")
