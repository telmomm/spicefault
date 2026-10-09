"""Yield: the fraction of the drawn circuits that meets its specifications."""

from __future__ import annotations

import math
from collections.abc import Sequence
from statistics import NormalDist

import numpy as np
import pandas as pd

from ..reliability.statistics import INTERVALS

ALL = "all"


def samples_for_half_width(half_width: float, p: float = 0.5, confidence: float = 0.95) -> int:
    """Samples for a proportion near `p` to be estimated within +- `half_width`:
    n = z^2 p (1 - p) / half_width^2, the normal approximation. `p = 0.5` is the worst
    case; for a yield near 1 use the expected yield.
    """
    if not 0.0 < half_width < 1.0 or not 0.0 <= p <= 1.0:
        raise ValueError("half_width must be in (0, 1) and p in [0, 1]")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    return max(1, math.ceil(z * z * p * (1.0 - p) / half_width**2))


def zero_failure_bound(n: int, confidence: float = 0.95) -> float:
    """Upper bound of the failure probability when none of `n` independent samples
    failed: 1 - (1 - confidence)^(1 / n), about 3 / n at 95 %. No failure in a sample
    does not mean that the probability of one is zero.
    """
    if n < 1:
        raise ValueError("the bound needs at least one sample")
    return 1.0 - (1.0 - confidence) ** (1.0 / n)


def yield_report(
    values: pd.DataFrame,
    specifications: Sequence,
    simulated: np.ndarray | None = None,
    confidence: float = 0.95,
    interval: str = "wilson",
) -> pd.DataFrame:
    """Yield per specification and overall, from one row per drawn circuit.

    `values` holds the column of each specification; `simulated` is False for the
    circuits whose simulation failed (all True by default). One row per specification
    and a last row `all`:

    - `n`: circuits simulated successfully; `n_failed`: those that were not;
    - `passed`, `yield`, `ci_low`, `ci_high`: over the `n` simulated circuits, with a
      binomial interval (`wilson` or `clopper-pearson`) that assumes independent,
      randomly drawn circuits;
    - `yield_min`, `yield_max`: the yield over all the circuits if every failed
      simulation is counted as not compliant, and as compliant. A value that is not
      finite in a simulated circuit is a violation;
    - `only_this`: circuits that fail this specification and no other;
    - `mean`, `std`, `minimum`, `maximum`: of the specified quantity;
    - `margin_sigma`: distance from the mean to the nearest limit, in standard
      deviations. It describes the sample; it is not a guarantee, and it says little
      about a distribution that is far from normal.

    A finite sample estimates and bounds the yield; it does not prove it.
    """
    specifications = list(specifications)
    if not specifications:
        raise ValueError("a yield needs at least one specification")
    estimate = INTERVALS[interval]
    simulated = (
        np.ones(len(values), dtype=bool) if simulated is None else np.asarray(simulated, bool)
    )
    n_all, n = len(values), int(simulated.sum())
    met = np.column_stack(
        [s.met(values[s.name].to_numpy(dtype=float)[simulated]) for s in specifications]
    )
    n_violated = (~met).sum(axis=1)

    def row(name: str, passed: np.ndarray) -> dict:
        k = int(passed.sum())
        low, high = estimate(k, n, confidence)
        return {
            "specification": name, "n": n, "n_failed": n_all - n, "passed": k,
            "yield": k / n if n else math.nan, "ci_low": low, "ci_high": high,
            "yield_min": k / n_all if n_all else math.nan,
            "yield_max": (k + n_all - n) / n_all if n_all else math.nan,
        }  # fmt: skip

    rows = []
    for j, specification in enumerate(specifications):
        x = values[specification.name].to_numpy(dtype=float)[simulated]
        x = x[np.isfinite(x)]
        mean = float(x.mean()) if len(x) else math.nan
        std = float(x.std(ddof=1)) if len(x) > 1 else math.nan
        distances = [
            d
            for d in (
                None if specification.minimum is None else mean - specification.minimum,
                None if specification.maximum is None else specification.maximum - mean,
            )
            if d is not None
        ]
        rows.append({
            **row(specification.name, met[:, j]),
            "only_this": int((~met[:, j] & (n_violated == 1)).sum()),
            "minimum_limit": specification.minimum, "maximum_limit": specification.maximum,
            "mean": mean, "std": std,
            "minimum": float(x.min()) if len(x) else math.nan,
            "maximum": float(x.max()) if len(x) else math.nan,
            "margin_sigma": min(distances) / std if std and std > 0 else math.nan,
        })  # fmt: skip
    rows.append(row(ALL, n_violated == 0))
    return pd.DataFrame(rows).set_index("specification")
