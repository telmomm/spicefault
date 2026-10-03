"""Simulator backends. The simulator is an external process; ngspice is the first one."""

from .ngspice import (
    RAW_NAME,
    Plot,
    SimulationError,
    ngspice_path,
    ngspice_version,
    parse_raw,
    run_deck,
)

__all__ = [
    "RAW_NAME",
    "Plot",
    "SimulationError",
    "ngspice_path",
    "ngspice_version",
    "parse_raw",
    "run_deck",
]
