"""The distribution of each measurement over the drawn circuits."""

from __future__ import annotations

import math
from collections.abc import Sequence
from statistics import NormalDist

import numpy as np
import pandas as pd


def ecdf(values) -> tuple[np.ndarray, np.ndarray]:
    """Empirical distribution of the finite values: (x sorted, fraction of values <= x)."""
    x = np.sort(np.asarray(values, dtype=float))
    x = x[np.isfinite(x)]
    return x, np.arange(1, len(x) + 1) / max(len(x), 1)


def _binomial_cdf(n: int, p: float) -> np.ndarray:
    """P(X <= k) for X ~ Binomial(n, p), for every k from 0 to n."""
    log_factorial = np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, n + 1)))])
    k = np.arange(n + 1)
    log_pmf = (
        log_factorial[n] - log_factorial[k] - log_factorial[n - k]
        + k * math.log(p) + (n - k) * math.log1p(-p)
    )  # fmt: skip
    return np.minimum(np.cumsum(np.exp(log_pmf)), 1.0)


def quantile_interval(n: int, q: float, confidence: float = 0.95) -> tuple[int, int] | None:
    """Ranks (l, u), counted from 1, of the order statistics that bracket the quantile `q`
    of any continuous distribution with at least the given confidence:
    P(x_(l) <= quantile <= x_(u)) >= confidence.

    The number of samples below the quantile is Binomial(n, q), so the ranks come from
    its tails, with half of the error probability on each side. None if the sample is
    too small for such ranks to exist: the quantile cannot be estimated from it.
    """
    if not 0.0 < q < 1.0:
        raise ValueError(f"a quantile is between 0 and 1, not {q}")
    if n < 1:
        return None
    tail = (1.0 - confidence) / 2.0
    cdf = _binomial_cdf(n, q)
    # P(x_(l) > quantile) = P(X <= l - 1) <= tail;  P(x_(u) < quantile) = P(X >= u) <= tail
    below = np.flatnonzero(cdf <= tail)
    above = np.flatnonzero(cdf >= 1.0 - tail)
    if len(below) == 0 or len(above) == 0 or above[0] + 1 > n:
        return None
    return int(below[-1]) + 1, int(above[0]) + 1


def describe(
    samples: pd.DataFrame,
    features: Sequence[str],
    by: Sequence[str] = ("fault_id", "condition"),
    ok: str | None = "sim_ok",
    quantiles: Sequence[float] = (0.01, 0.5, 0.99),
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Statistics of each feature over the samples of each group, one row per group and
    feature.

    - `n`: samples used, those whose simulation succeeded; `n_failed`: those that did
      not. They are counted, never dropped silently: a statistic over the successful
      samples describes the circuits that could be simulated.
    - `mean`, `std` (with n - 1), `minimum`, `maximum`, and `mean_low`, `mean_high`:
      the interval of the mean, mean +- z std / sqrt(n), which assumes that n is large
      enough for the mean to be normal.
    - `q<p>`, `q<p>_low`, `q<p>_high` for each quantile p: the estimate (linear
      interpolation between order statistics) and a distribution-free interval from
      the order statistics (`quantile_interval`). When the sample is too small for the
      interval to exist, the three are NaN: the smallest of 100 values is not an
      estimate of the 0.1 % quantile.

    A group in which a feature has no value at all, as for a measurement that its
    operating condition does not make, gives no row.
    """
    by, features = list(by), list(features)
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    valid = np.ones(len(samples), dtype=bool)
    if ok and ok in samples:
        valid = samples[ok].to_numpy(dtype=bool)
    groups = samples.groupby(by, sort=False).indices if by else {(): np.arange(len(samples))}
    rows = []
    for key, index in groups.items():
        key = key if isinstance(key, tuple) else (key,)
        used = index[valid[index]]
        for feature in features:
            x = samples[feature].to_numpy(dtype=float)[used]
            x = np.sort(x[np.isfinite(x)])
            n = len(x)
            if n == 0 and len(used):
                continue  # nothing measured here: not a failure
            row = dict(zip(by, key, strict=True))
            mean = float(x.mean()) if n else math.nan
            std = float(x.std(ddof=1)) if n > 1 else math.nan
            half = z * std / math.sqrt(n) if n > 1 else math.nan
            row.update(
                measurement=feature, n=n, n_failed=len(index) - len(used), mean=mean,
                mean_low=mean - half, mean_high=mean + half, std=std,
                minimum=float(x[0]) if n else math.nan, maximum=float(x[-1]) if n else math.nan,
            )  # fmt: skip
            for q in quantiles:
                ranks = quantile_interval(n, q, confidence)
                name = f"q{q:g}"
                estimate = low = high = math.nan
                if ranks is not None:
                    estimate = float(np.quantile(x, q))
                    low, high = float(x[ranks[0] - 1]), float(x[ranks[1] - 1])
                row.update({name: estimate, f"{name}_low": low, f"{name}_high": high})
            rows.append(row)
    return pd.DataFrame(rows).set_index([*by, "measurement"]) if rows else pd.DataFrame()
