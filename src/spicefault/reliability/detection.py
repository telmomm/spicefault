"""Decision rules: what turns a vector of measurements into "not healthy"."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


class Detector(Protocol):
    def fit(self, healthy: np.ndarray) -> Detector:
        """Set the thresholds from healthy samples, [n, features]."""

    def flag(self, x: np.ndarray) -> np.ndarray:
        """True for the rows of `x` judged not healthy."""


@dataclass
class LimitTest:
    """Limit test: a sample is flagged when any feature is outside its limits.

    Each feature gets the central interval of the healthy samples at tail probability
    alpha / (2 d) per side, so that the d features together keep a false-alarm rate of
    about `alpha` (Bonferroni; exact if the features were independent and the limits
    known). The limits are estimated, and with small alpha / (2 d) they sit close to
    the extreme healthy samples: the realised false-alarm rate must be measured on
    healthy samples that were not used here.
    """

    alpha: float = 0.01
    low: np.ndarray = field(default=None, repr=False)
    high: np.ndarray = field(default=None, repr=False)

    def fit(self, healthy: np.ndarray) -> LimitTest:
        tail = self.alpha / (2 * healthy.shape[1])
        self.low = np.quantile(healthy, tail, axis=0)
        self.high = np.quantile(healthy, 1.0 - tail, axis=0)
        return self

    def flag(self, x: np.ndarray) -> np.ndarray:
        return ((x < self.low) | (x > self.high)).any(axis=1)
