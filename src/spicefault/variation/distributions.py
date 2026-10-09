"""Random deviations used to draw a healthy circuit."""

from __future__ import annotations

from statistics import NormalDist

import numpy as np

DISTRIBUTIONS = ("uniform", "truncnorm")


def unit_deviation(rng: np.random.Generator, distribution: str) -> float:
    """Draw a deviation in [-1, 1], in units of the tolerance.

    `uniform`, or `truncnorm`: normal with sigma = 1/3, redrawn until it falls in range.
    """
    if distribution == "uniform":
        return float(rng.uniform(-1.0, 1.0))
    if distribution == "truncnorm":
        while True:
            x = rng.normal(0.0, 1.0 / 3.0)
            if abs(x) <= 1.0:
                return float(x)
    raise ValueError(f"unknown tolerance distribution: {distribution}")


def check_probability(u: float) -> float:
    if not 0.0 < u < 1.0:
        raise ValueError(f"a quantile is asked at a probability in (0, 1), not {u}")
    return float(u)


def truncated_normal_quantile(u: float, limit: float | None) -> float:
    """Quantile of a standard normal truncated at +-`limit` (not truncated if None)."""
    u, normal = check_probability(u), NormalDist()
    if limit is None:
        return normal.inv_cdf(u)
    low = normal.cdf(-limit)
    return normal.inv_cdf(low + u * (1.0 - 2.0 * low))


def unit_deviation_quantile(u: float, distribution: str) -> float:
    """The deviation in [-1, 1] that `unit_deviation` draws, at probability `u`."""
    if distribution == "uniform":
        return 2.0 * check_probability(u) - 1.0
    if distribution == "truncnorm":  # sigma = 1/3, truncated at 3 sigma
        return truncated_normal_quantile(u, 3.0) / 3.0
    raise ValueError(f"unknown tolerance distribution: {distribution}")


def toleranced(
    nominal: float, tolerance: float, rng: np.random.Generator, distribution: str = "uniform"
) -> float:
    """Value drawn within nominal * (1 +- tolerance)."""
    return nominal * (1.0 + tolerance * unit_deviation(rng, distribution))


def log_uniform_factor(rng: np.random.Generator, spread: float) -> float:
    """Factor drawn log-uniformly between 1 / spread and spread."""
    log_spread = np.log(float(spread))
    return float(np.exp(rng.uniform(-log_spread, log_spread)))
