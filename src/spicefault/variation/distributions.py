"""Random deviations used to draw a healthy circuit."""

from __future__ import annotations

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


def toleranced(
    nominal: float, tolerance: float, rng: np.random.Generator, distribution: str = "uniform"
) -> float:
    """Value drawn within nominal * (1 +- tolerance)."""
    return nominal * (1.0 + tolerance * unit_deviation(rng, distribution))


def log_uniform_factor(rng: np.random.Generator, spread: float) -> float:
    """Factor drawn log-uniformly between 1 / spread and spread."""
    log_spread = np.log(float(spread))
    return float(np.exp(rng.uniform(-log_spread, log_spread)))
