"""Local sensitivity of the measurements to the components, and what it bounds."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from ..circuit import Circuit
from ..measurements import Measurement
from ..simulation import SimulationConfig, Simulator
from .structure import connected_groups


def local_sensitivity(
    circuit: Circuit,
    measurements: Sequence[Measurement],
    config: SimulationConfig,
    parameters: Sequence[tuple[str, str]] | None = None,
    simulator: Simulator | None = None,
    step: float = 0.01,
) -> pd.DataFrame:
    """Change of each measurement for a +1 % change of each parameter, at the nominal circuit.

    Central differences with relative step `step`:
    S = (m(x (1 + h)) - m(x (1 - h))) / (2 h) * 0.01. Rows are measurements; columns
    are `component` for a value and `component.parameter` otherwise. `parameters`
    defaults to the values of the resistors, capacitors and inductors.

    It is valid near the nominal point only, and it is noise-free.
    """
    simulator = simulator or Simulator()
    if parameters is None:
        parameters = [(c.name, "value") for c in circuit.components() if c.kind in "RCL"]

    def measured(component: str, parameter: str, factor: float) -> pd.Series:
        netlist = circuit.netlist()
        netlist.set_parameter(component, parameter, "scale", factor)
        result = simulator.run(netlist, config)
        if not result.ok:
            raise RuntimeError(
                f"{component}.{parameter} x {factor}: {result.status.value}: {result.message}"
            )
        return pd.Series({m.name: m(result) for m in measurements})

    columns = {}
    for component, parameter in parameters:
        plus = measured(component, parameter, 1.0 + step)
        minus = measured(component, parameter, 1.0 - step)
        name = component if parameter == "value" else f"{component}.{parameter}"
        columns[name] = (plus - minus) / (2.0 * step) * 0.01
    return pd.DataFrame(columns)


def normalised_sensitivity(sensitivity: pd.DataFrame, spread: pd.Series) -> pd.DataFrame:
    """Sensitivity in units of the healthy spread of each measurement (z per 1 % change).

    A value of 1 means that a 1 % deviation of the component moves the measurement by
    one standard deviation of the healthy population. Measurements without spread are
    dropped.
    """
    spread = spread.reindex(sensitivity.index)
    keep = spread.notna() & (spread > 0)
    return sensitivity[keep].div(spread[keep], axis=0)


def testability_rank(z: pd.DataFrame, deviation_pct: float = 10.0, threshold: float = 3.0) -> int:
    """Number of independent directions along which a small deviation is visible.

    Counts the singular values s of the normalised sensitivity matrix with
    s * `deviation_pct` >= `threshold`: directions along which a deviation of that size
    moves the measurements by at least `threshold` healthy standard deviations. It
    bounds how many components can be told apart by deviations of that size.
    """
    if z.empty:
        return 0
    singular = np.linalg.svd(z.to_numpy(), compute_uv=False)
    return int(np.sum(singular * deviation_pct >= threshold))


def collinear_groups(
    z: pd.DataFrame,
    cos_threshold: float = 0.99,
    deviation_pct: float = 10.0,
    threshold: float = 3.0,
) -> tuple[list[list[str]], list[str]]:
    """(groups of components with parallel sensitivity vectors, insensitive components).

    A component is insensitive when a deviation of `deviation_pct` moves the
    measurements by less than `threshold` healthy standard deviations (Euclidean
    norm). The rest are linked when the absolute cosine of their sensitivity vectors
    reaches `cos_threshold`, since then a small deviation of one can be mimicked by
    the other.
    """
    min_norm = threshold / deviation_pct
    norms = np.linalg.norm(z.to_numpy(), axis=0)
    sensitive = [c for c, n in zip(z.columns, norms, strict=True) if n >= min_norm]
    insensitive = [c for c, n in zip(z.columns, norms, strict=True) if n < min_norm]
    v = z[sensitive].to_numpy()
    v = v / np.maximum(np.linalg.norm(v, axis=0), 1e-300)
    cos = np.abs(v.T @ v)
    close = np.triu(cos >= cos_threshold, k=1)
    return connected_groups(sensitive, zip(*np.nonzero(close), strict=True)), insensitive
