"""The uncertainty of an estimate from independent replications of an experiment."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def student_t_quantile(p: float, df: int) -> float:
    """Quantile of the Student t distribution with `df` degrees of freedom, by
    integrating its density and bisecting.
    """
    if not 0.0 < p < 1.0 or df < 1:
        raise ValueError("a t quantile needs a probability in (0, 1) and at least one degree")
    if p < 0.5:
        return -student_t_quantile(1.0 - p, df)
    log_constant = (
        math.lgamma((df + 1) / 2) - math.lgamma(df / 2) - 0.5 * math.log(df * math.pi)
    )

    def cdf(x: float) -> float:
        # the substitution t = tan(a) keeps the heavy tail of few degrees inside a finite range
        angle = np.linspace(0.0, math.atan(x), 4001)
        t = np.tan(angle)
        density = np.exp(log_constant - (df + 1) / 2 * np.log1p(t * t / df)) / np.cos(angle) ** 2
        step = angle[1] - angle[0]
        simpson = step / 3 * (density[0] + density[-1] + 4 * density[1:-1:2].sum()
                              + 2 * density[2:-1:2].sum())  # fmt: skip
        return 0.5 + float(simpson)

    low, high = 0.0, 1.0
    while cdf(high) < p:
        high *= 2.0
    for _ in range(80):
        middle = 0.5 * (low + high)
        low, high = (middle, high) if cdf(middle) < p else (low, middle)
    return 0.5 * (low + high)


def replicated_interval(
    estimates: Sequence[float], confidence: float = 0.95
) -> tuple[float, float, float]:
    """(mean, low, high) of an estimate obtained in several independent replications.

    This is the interval for an experiment whose samples are not independent, such as a
    Latin hypercube or a Sobol design: the whole experiment is repeated with different
    seeds, each repetition gives one estimate, and the estimates are independent even
    though the samples inside a repetition are not. mean +- t s / sqrt(k), with the
    Student t of k - 1 degrees; it assumes that the estimates are roughly normal.
    """
    values = np.asarray(list(estimates), dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("an interval from replications needs at least two finite estimates")
    mean = float(values.mean())
    half = student_t_quantile(0.5 + confidence / 2.0, len(values) - 1) * float(
        values.std(ddof=1)
    ) / math.sqrt(len(values))
    return mean, mean - half, mean + half
