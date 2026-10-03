"""Simulator abstraction: configuration, result with an explicit status, and backends.

A backend runs one netlist and always returns a `SimulationResult`. A simulation that
fails is a result with a status, not an exception, so that failures stay in the
experimental record.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
from typing import Protocol

import numpy as np

from .ngspice import RAW_NAME, Plot, SimulationError, execute, ngspice_version

# commands of ngspice that run an analysis and leave a plot to save
_ANALYSIS_COMMANDS = ("op", "ac", "tran", "dc", "noise", "tf", "sens", "pz", "disto")


class SimulationStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    CONVERGENCE_ERROR = "CONVERGENCE_ERROR"
    INVALID_OUTPUT = "INVALID_OUTPUT"


@dataclass(frozen=True)
class SimulationConfig:
    """What to simulate and how.

    `analyses` are ngspice analysis commands (`"op"`, `"ac dec 20 1 1e5"`,
    `"tran 1u 10m"`), run in order; the vectors in `outputs` are saved after each one,
    giving one plot per analysis. Any other command in the list, such as
    `"alter @vin[acmag]=1"`, is run in its place and saves nothing. Leave `analyses`
    empty for a netlist that brings its own `.control` block writing to `out.raw`.
    """

    analyses: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ("all",)
    timeout: float = 120.0
    spiceinit: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "analyses", tuple(self.analyses))
        object.__setattr__(self, "outputs", tuple(self.outputs))

    def plots(self) -> list[str]:
        """The analysis command behind each expected plot, in order."""
        return [a for a in self.analyses if a.split()[0].lower() in _ANALYSIS_COMMANDS]

    def metadata(self) -> dict:
        return {
            "analyses": list(self.analyses),
            "outputs": list(self.outputs),
            "timeout": self.timeout,
            "spiceinit": self.spiceinit,
        }

    @classmethod
    def from_metadata(cls, record: dict) -> SimulationConfig:
        return cls(**record)


@dataclass(frozen=True)
class SimulationResult:
    """Outcome of one simulation: a status, and one plot per analysis if it succeeded."""

    status: SimulationStatus
    plots: tuple[Plot, ...] = ()
    message: str = ""
    elapsed: float = 0.0
    log: str = field(default="", repr=False)  # tail of the simulator's messages on failure

    @property
    def ok(self) -> bool:
        return self.status is SimulationStatus.SUCCESS

    def plot(self, which: int | str = 0) -> Plot:
        """Plot by position, or the first whose name starts with `which` (case-insensitive)."""
        if isinstance(which, int):
            return self.plots[which]
        for plot in self.plots:
            if plot.name.lower().startswith(which.lower()):
                return plot
        raise KeyError(f"no plot named {which!r}; available: {[p.name for p in self.plots]}")


class SimulatorBackend(Protocol):
    name: str

    def version(self) -> str: ...

    def run(self, netlist: str, config: SimulationConfig) -> SimulationResult: ...


# analysis command -> start of the name ngspice gives to its plot
_PLOT_NAMES = {
    "op": "Operating Point",
    "ac": "AC Analysis",
    "tran": "Transient Analysis",
    "dc": "DC transfer characteristic",
}
# messages of ngspice when the solver gives up, lower case
_CONVERGENCE = (
    "singular matrix",
    "timestep too small",
    "no convergence",
    "iteration limit",
    "gmin stepping failed",
    "source stepping failed",
)


def build_deck(netlist: str, config: SimulationConfig) -> str:
    """The netlist with a control block that runs the analyses of `config`."""
    if not config.analyses:
        return netlist
    lines = netlist.rstrip("\n").split("\n")
    if any(line.strip().lower().startswith(".control") for line in lines):
        raise ValueError("the netlist already has a .control block; leave `analyses` empty")
    while lines and lines[-1].strip().lower() in ("", ".end"):
        lines.pop()
    lines += [".control", "set noaskquit", "set appendwrite"]
    write = f"write {RAW_NAME} " + " ".join(config.outputs)
    saved = config.plots()
    for command in config.analyses:
        lines += [command, write] if command in saved else [command]
    return "\n".join([*lines, ".endc", ".end", ""])


class NgspiceBackend:
    """ngspice as an external process, one per simulation."""

    name = "ngspice"

    @cached_property
    def _version(self) -> str:
        return ngspice_version()

    def version(self) -> str:
        return self._version

    def run(self, netlist: str, config: SimulationConfig) -> SimulationResult:
        t0 = time.perf_counter()

        def result(status: SimulationStatus, message: str = "", plots=(), log: str = ""):
            return SimulationResult(
                status, tuple(plots), message, time.perf_counter() - t0, log[-2000:]
            )

        try:
            plots, log = execute(build_deck(netlist, config), config.timeout, config.spiceinit)
        except subprocess.TimeoutExpired:
            return result(SimulationStatus.TIMEOUT, f"ngspice timed out after {config.timeout} s")
        except SimulationError as exc:  # ngspice missing
            return result(SimulationStatus.FAILED, str(exc))
        except (ValueError, IndexError) as exc:  # raw file cut short or malformed
            return result(SimulationStatus.INVALID_OUTPUT, f"unreadable raw file: {exc}")

        problem = self._check(plots, config)
        if problem is None:
            return result(SimulationStatus.SUCCESS, plots=plots)
        lowered = log.lower()
        cause = next((text for text in _CONVERGENCE if text in lowered), None)
        if cause is not None:
            return result(SimulationStatus.CONVERGENCE_ERROR, f"ngspice: {cause}", log=log)
        status = SimulationStatus.FAILED if plots is None else SimulationStatus.INVALID_OUTPUT
        return result(status, problem, log=log)

    @staticmethod
    def _check(plots: list[Plot] | None, config: SimulationConfig) -> str | None:
        """Reason why the output cannot be trusted, or None."""
        if plots is None:
            return "ngspice produced no output"
        if config.analyses:
            # a failed analysis leaves the previous plot current, and `write` saves it again
            expected = [_PLOT_NAMES.get(a.split()[0].lower(), "") for a in config.plots()]
            names = [p.name for p in plots]
            if len(names) != len(expected) or not all(map(str.startswith, names, expected)):
                return f"expected plots {expected}, got {names}"
        for plot in plots:
            for name, vector in plot.vectors.items():
                if not vector.size:
                    return f"{plot.name}: {name} is empty"
                if not np.isfinite(vector).all():
                    return f"{plot.name}: {name} is not finite"
        return None


BACKENDS = {"ngspice": NgspiceBackend}


class Simulator:
    """The numerical engine of an experiment, chosen by name or given as a backend."""

    def __init__(self, backend: str | SimulatorBackend = "ngspice"):
        if isinstance(backend, str):
            if backend not in BACKENDS:
                raise ValueError(f"unknown backend {backend!r}; available: {sorted(BACKENDS)}")
            backend = BACKENDS[backend]()
        self.backend = backend

    def run(self, netlist: str, config: SimulationConfig | None = None) -> SimulationResult:
        return self.backend.run(str(netlist), config or SimulationConfig())

    def metadata(self) -> dict:
        return {"simulator": self.backend.name, "simulator_version": self.backend.version()}
