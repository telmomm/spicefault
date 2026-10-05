"""Thin ngspice wrapper: run a deck in batch mode and read its binary raw file."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RAW_NAME = "out.raw"


class SimulationError(RuntimeError):
    pass


@dataclass
class Plot:
    """One analysis result. Vector names are lower case, e.g. `v(out)`, `frequency`."""

    name: str
    vectors: dict[str, np.ndarray]

    def __getitem__(self, key: str) -> np.ndarray:
        return self.vectors[key]


# GUI-launched kernels (VS Code, Jupyter) often lack the Homebrew directories in PATH
_FALLBACK_DIRS = os.pathsep.join(["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin"])


def ngspice_path() -> str | None:
    return shutil.which("ngspice") or shutil.which("ngspice", path=_FALLBACK_DIRS)


def ngspice_version() -> str:
    exe = ngspice_path()
    if exe is None:
        return "not found"
    out = subprocess.run([exe, "--version"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if "ngspice-" in line:
            return line.strip("* ").split(":")[0].strip()
    return "unknown"


def parse_raw(path: str | Path) -> list[Plot]:
    """Parse an ngspice binary raw file, which may hold several appended plots."""
    data = Path(path).read_bytes()
    plots: list[Plot] = []
    pos = 0
    marker = b"Binary:\n"
    while pos < len(data):
        idx = data.find(marker, pos)
        if idx < 0:
            break
        header = data[pos:idx].decode("latin-1").splitlines()
        name, is_complex, n_vars, n_points, var_names = "", False, 0, 0, []
        for i, line in enumerate(header):
            key, _, value = line.partition(":")
            if key == "Plotname":
                name = value.strip()
            elif key == "Flags":
                is_complex = "complex" in value
            elif key == "No. Variables":
                n_vars = int(value)
            elif key == "No. Points":
                n_points = int(value)
            elif key == "Variables":
                var_names = [h.split()[1].lower() for h in header[i + 1 : i + 1 + n_vars]]
        dtype = np.complex128 if is_complex else np.float64
        start = idx + len(marker)
        n_bytes = n_points * n_vars * np.dtype(dtype).itemsize
        block = np.frombuffer(data, dtype=dtype, count=n_points * n_vars, offset=start)
        block = block.reshape(n_points, n_vars)
        plots.append(Plot(name, {v: block[:, k] for k, v in enumerate(var_names)}))
        pos = start + n_bytes
    return plots


def execute(
    netlist: str,
    timeout: float = 120.0,
    spiceinit: str | None = None,
    raw_name: str = RAW_NAME,
) -> tuple[list[Plot] | None, str]:
    """Run a deck; return (plots, or None if no raw file was written; ngspice's messages).

    Raises `subprocess.TimeoutExpired` after `timeout` seconds and `SimulationError`
    if ngspice is not installed.
    """
    exe = ngspice_path()
    if exe is None:
        raise SimulationError("ngspice executable not found on PATH")
    with tempfile.TemporaryDirectory(prefix="spicefault_") as tmp:
        deck = Path(tmp) / "deck.cir"
        deck.write_text(netlist)
        if spiceinit is None:
            command = [exe, "-b", "-n", deck.name]  # -n: ignore the user's .spiceinit
        else:
            (Path(tmp) / ".spiceinit").write_text(spiceinit + "\n")
            command = [exe, "-b", deck.name]
        proc = subprocess.run(command, cwd=tmp, capture_output=True, text=True, timeout=timeout)
        raw = Path(tmp) / raw_name
        return (parse_raw(raw) if raw.exists() else None), proc.stdout + proc.stderr


def run_deck(
    netlist: str,
    timeout: float = 120.0,
    spiceinit: str | None = None,
    raw_name: str = RAW_NAME,
) -> list[Plot]:
    """Run a deck whose control block writes to `raw_name`; return the plots in order.

    `spiceinit` holds start-up commands (e.g. a compatibility mode for vendor models),
    which ngspice only accepts from an initialisation file. Failures raise
    `SimulationError`; `NgspiceBackend` reports them as a status instead.
    """
    try:
        plots, log = execute(netlist, timeout, spiceinit, raw_name)
    except subprocess.TimeoutExpired as exc:
        raise SimulationError(f"ngspice timed out after {timeout} s") from exc
    if plots is None:
        raise SimulationError(f"ngspice produced no output:\n{log[-2000:]}")
    return plots
