"""Estimators used by the reliability metrics, with their uncertainty."""

from __future__ import annotations

import math
from collections.abc import Callable
from statistics import NormalDist

import numpy as np


def wilson_interval(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval of a proportion k / n. (nan, nan) if n is 0."""
    if n == 0:
        return math.nan, math.nan
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    # at k = 0 and k = n the exact limits are 0 and 1; rounding must not move them
    return (0.0 if k == 0 else max(centre - half, 0.0), 1.0 if k == n else min(centre + half, 1.0))


def _binomial_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p)."""
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return float(k >= n)
    i = np.arange(0, k + 1)
    log_comb = math.lgamma(n + 1) - np.array(
        [math.lgamma(j + 1) + math.lgamma(n - j + 1) for j in i]
    )
    return float(min(np.exp(log_comb + i * math.log(p) + (n - i) * math.log1p(-p)).sum(), 1.0))


def clopper_pearson_interval(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Exact (conservative) interval of a proportion k / n, by inverting the binomial tails."""
    if n == 0:
        return math.nan, math.nan
    tail = (1.0 - confidence) / 2.0

    def solve(target: Callable[[float], float]) -> float:
        lo, hi = 0.0, 1.0
        for _ in range(60):  # target is decreasing in p
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if target(mid) > 0 else (lo, mid)
        return 0.5 * (lo + hi)

    # lower limit: P(X >= k | p) = tail; upper limit: P(X <= k | p) = tail
    low = 0.0 if k == 0 else solve(lambda p: tail - (1.0 - _binomial_cdf(k - 1, n, p)))
    high = 1.0 if k == n else solve(lambda p: _binomial_cdf(k, n, p) - tail)
    return low, high


INTERVALS = {"wilson": wilson_interval, "clopper-pearson": clopper_pearson_interval}


def robust_spread(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """IQR / 1.349: equal to the standard deviation for a normal cloud, and not inflated
    by the few extreme values of a fault that saturates an output.
    """
    q75, q25 = np.quantile(x, [0.75, 0.25], axis=axis)
    return (q75 - q25) / 1.349


def auc(fault: np.ndarray, healthy: np.ndarray) -> float:
    """P(fault > healthy) + P(fault = healthy) / 2, the Mann-Whitney statistic."""
    both = np.concatenate([fault, healthy])
    _, inverse, counts = np.unique(both, return_inverse=True, return_counts=True)
    last = np.cumsum(counts)  # rank of the last element of each group of ties
    ranks = (last - (counts - 1) / 2.0)[inverse]
    n_f, n_h = len(fault), len(healthy)
    return float((ranks[:n_f].sum() - n_f * (n_f + 1) / 2.0) / (n_f * n_h))


def bootstrap_interval(
    statistic: Callable[[np.random.Generator], float],
    n_boot: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile interval of `statistic(rng)`, which resamples with the generator it gets."""
    rng = np.random.default_rng(seed)
    values = [statistic(rng) for _ in range(n_boot)]
    tail = (1.0 - confidence) / 2.0
    low, high = np.nanquantile(values, [tail, 1.0 - tail])
    return float(low), float(high)
