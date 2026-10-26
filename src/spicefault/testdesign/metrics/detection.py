"""Detection and false alarms of a set of tests, with Clopper–Pearson intervals.

`detect` reads the test matrix of `candidates.matrix` (one row per drawn circuit, one
column per test). A circuit raises an alarm when any test of the set leaves its limits;
the limits are set on one part of the fault-free circuits and the false alarm rate is
measured on the other.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from statistics import NormalDist

import numpy as np
import pandas as pd

from ...experiments.experiment import HEALTHY_ID
from ...reliability import LimitTest, clopper_pearson_interval, robust_spread

HEALTHY = HEALTHY_ID


@dataclass
class GaussianLimits:
    """Limit test with limits at median ± z · robust spread of the fault-free values.

    The tail of each side is alpha / (2 d) for d tests, as in `LimitTest`, but the limits
    are extrapolated from a normal model instead of read from the sample, so a small
    alpha can be targeted with few fault-free circuits. With `cover_sample` a limit is
    never inside the range of the calibration values, which protects against tests with
    heavier tails than the normal. The realised false alarm rate still has to be
    measured on circuits not used to fit the limits.
    """

    alpha: float = 0.01
    cover_sample: bool = True
    low: np.ndarray | None = field(default=None, repr=False)
    high: np.ndarray | None = field(default=None, repr=False)

    def fit(self, healthy: np.ndarray) -> GaussianLimits:
        """Set the limits from fault-free values, [n, tests]."""
        z = NormalDist().inv_cdf(1.0 - self.alpha / (2 * healthy.shape[1]))
        centre, spread = np.median(healthy, axis=0), robust_spread(healthy, axis=0)
        self.low, self.high = centre - z * spread, centre + z * spread
        if self.cover_sample:
            self.low = np.minimum(self.low, healthy.min(axis=0))
            self.high = np.maximum(self.high, healthy.max(axis=0))
        return self

    def flag(self, x: np.ndarray) -> np.ndarray:
        """True for the rows of `x` with any test outside its limits."""
        return ((x < self.low) | (x > self.high)).any(axis=1)


def split_healthy(matrix: pd.DataFrame, fraction: float = 0.5) -> tuple[pd.Index, pd.Index]:
    """Fault-free rows used to set the limits and rows kept to measure false alarms.

    The split is by replica (the first `fraction` of them calibrate), so it does not
    depend on the order of the rows and is the same for every set of tests.
    """
    healthy = matrix[matrix.fault_id == HEALTHY]
    cut = fraction * (healthy.replica.max() + 1)
    return healthy.index[healthy.replica < cut], healthy.index[healthy.replica >= cut]


@dataclass(frozen=True)
class Detection:
    """Result of `detect`.

    `per_fault` is indexed by fault, with the circuits `n`, the alarms `detected`, the
    proportion `p` and its interval; `false_alarm` gives the same on the held-out
    fault-free circuits; `alarms` is indexed by row of the matrix; `angles` gives, for
    each phase test, the fault-free direction it is read from.
    """

    tests: tuple[str, ...]
    per_fault: pd.DataFrame
    false_alarm: dict
    alarms: pd.Series
    detector: object
    confidence: float
    angles: dict = field(default_factory=dict)


def is_circular(test_id: str) -> bool:
    """Tests whose value is an angle in degrees: the phase features of `candidates`."""
    return "|phase@" in test_id


def circular_mean(degrees: np.ndarray) -> float:
    """The mean direction of angles in degrees, in (-180, 180]."""
    r = np.radians(degrees)
    return float(np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean())))


def values(matrix: pd.DataFrame, tests: Sequence[str], angles: dict) -> np.ndarray:
    """The values of the tests, [rows, tests], with each test in `angles` read as its
    deviation from that direction, in (-180, 180]: a phase that crosses ±180 degrees is
    not a large change.
    """
    x = matrix[list(tests)].to_numpy(dtype=float).copy()
    for j, test in enumerate(tests):
        if test in angles:
            x[:, j] = -((angles[test] - x[:, j] + 180.0) % 360.0 - 180.0)
    return x


def angles_of(matrix: pd.DataFrame, rows: pd.Index, tests: Sequence[str]) -> dict[str, float]:
    """The fault-free direction of each phase test, from the circular mean of `rows`."""
    return {
        t: circular_mean(matrix.loc[rows, t].to_numpy(float)) for t in tests if is_circular(t)
    }


def detect(
    matrix: pd.DataFrame,
    tests: Sequence[str],
    detector=None,
    fraction: float = 0.5,
    confidence: float = 0.95,
) -> Detection:
    """Alarms of the tests `tests`: a circuit raises an alarm when any test leaves its limits.

    The detector (by default `LimitTest(0.01)`) is fitted on the calibration fault-free
    rows (see `split_healthy`); detection is counted on every faulty row and false
    alarms on the held-out fault-free rows. Intervals are two-sided Clopper–Pearson at
    `confidence`. Phase tests are read as their deviation from the circular mean of the
    calibration rows.
    """
    tests = tuple(tests)
    if not tests:
        raise ValueError("a set of tests needs at least one test")
    detector = detector if detector is not None else LimitTest(0.01)
    calibration, evaluation = split_healthy(matrix, fraction)
    angles = angles_of(matrix, calibration, tests)
    x = values(matrix, tests, angles)
    if not np.isfinite(x).all():
        raise ValueError("the matrix has values that are not finite")
    detector.fit(x[matrix.index.get_indexer(calibration)])
    alarms = pd.Series(np.asarray(detector.flag(x), dtype=bool), index=matrix.index)
    faulty = matrix.fault_id != HEALTHY
    rows = []
    for fault_id, flags in alarms[faulty].groupby(matrix.fault_id[faulty], sort=False):
        k, n = int(flags.sum()), len(flags)
        low, high = clopper_pearson_interval(k, n, confidence)
        row = {"fault_id": fault_id, "n": n, "detected": k, "p": k / n}
        rows.append({**row, "ci_low": low, "ci_high": high})
    k, n = int(alarms[evaluation].sum()), len(evaluation)
    low, high = clopper_pearson_interval(k, n, confidence)
    false_alarm = {
        "n": n,
        "alarms": k,
        "p": k / n if n else math.nan,
        "ci_low": low,
        "ci_high": high,
        "n_calibration": len(calibration),
    }
    per_fault = pd.DataFrame(rows).set_index("fault_id")
    return Detection(tests, per_fault, false_alarm, alarms, detector, confidence, angles)
