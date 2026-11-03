"""Risk-weighted coverage of a plan, scored and certified on circuits that did not
choose it.

The metrics themselves are in `metrics` (DC_w, DC, SFF, SPFM and LFM from a fault table,
H(F | Y) with a prior); the fault table can come from `testdesign.faultprior`. What this
module adds is the separation of the data:

- the limits of every test are set once, on fault-free circuits used for nothing else;
- a plan is chosen on one half of the replicas of each fault (`part(..., "selection")`)
  and scored on the other (`"evaluation"`), so the score of the winner is not inflated
  by having chosen it (`score`);
- a claim "metric ≥ target" is made by weighted fault sampling
  (`metrics.adaptive.certify_sampled`) on circuits drawn with another seed, through
  `ConditionSampler`, which simulates replicas on demand under any operating conditions
  and can read them through an instrument model (`costmodel.observe`) before the
  detector.

It also turns operating conditions into candidate tests, one per measurement, named
`condition:measurement` so that a measurement read under two conditions gives two tests
(`prefixed`).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from ...circuit import Circuit
from ...conditions import OperatingCondition
from ...experiments import Experiment
from ...faults import Fault
from ...variation import VariationSet
from .. import candidates
from .coverage import coverage, metric_values
from .detection import (
    HEALTHY,
    Detection,
    GaussianLimits,
    angles_of,
    detect,
    outside,
    values,
)
from .localisation import localisation

__all__ = ["ConditionSampler", "Score", "part", "prefixed", "score"]


def prefixed(
    conditions: Sequence[OperatingCondition],
) -> tuple[tuple[OperatingCondition, ...], list[str]]:
    """The conditions with each measurement renamed `condition:measurement`, and the
    names: one candidate test per condition and measurement."""
    renamed, ids = [], []
    for c in conditions:
        measurements = tuple(replace(m, name=f"{c.name}:{m.name}") for m in c.measurements)
        renamed.append(replace(c, measurements=measurements))
        ids += [m.name for m in measurements]
    return tuple(renamed), ids


def part(matrix: pd.DataFrame, which: str) -> pd.DataFrame:
    """Every fault-free row, and the faulty rows of the first (`selection`) or second
    (`evaluation`) half of the replicas of each fault."""
    if which not in ("selection", "evaluation"):
        raise ValueError(f"part is selection or evaluation, not {which!r}")
    healthy = matrix.fault_id == HEALTHY
    half = matrix.replica < (matrix[~healthy].replica.max() + 1) / 2
    keep = healthy | (half if which == "selection" else ~half)
    return matrix[keep]


@dataclass(frozen=True)
class Score:
    """A plan scored on one part of the data: its detection, each metric with its bounds
    (rows of `metrics.coverage`, or only the estimates), and H(F | Y) with a uniform
    prior and with the rates as prior, if asked for. `score["dc_w"]` is the estimate of
    a metric and `score.lower("dc_w")` its lower bound."""

    tests: tuple[str, ...]
    detection: Detection
    metrics: pd.DataFrame
    h_uniform: float | None = None
    h_prior: float | None = None

    def __getitem__(self, metric: str) -> float:
        return float(self.metrics.loc[metric, "estimate"])

    def lower(self, metric: str) -> float:
        return float(self.metrics.loc[metric, "lower"])


def score(
    matrix: pd.DataFrame,
    tests: Sequence[str],
    faults: pd.DataFrame,
    detector: Callable[[], object] = lambda: GaussianLimits(0.01),
    confidence: float = 0.95,
    method: str = "jeffreys",
    localise: bool = False,
    bounds: bool = True,
) -> Score:
    """Detection and metrics of `tests` on `matrix` (a part of the data), with the limits
    set on the calibration half of its fault-free rows (`metrics.split_healthy`).

    `faults` is a `metrics.fault_table`; `detector()` makes a fresh detector. With
    `localise`, H(F | Y) is computed with a uniform prior and with `faults.rate` as prior.
    `bounds=False` gives only the estimates, which is enough to compare plans inside a
    search.
    """
    d = detect(matrix, list(tests), detector())
    if bounds:
        table = coverage(d, faults.loc[d.per_fault.index], confidence, method)
    else:
        estimate = metric_values(d.per_fault.p, faults.loc[d.per_fault.index])
        table = pd.DataFrame({"estimate": estimate}).rename_axis("metric")
    h_uniform = h_prior = None
    if localise:
        h_uniform = localisation(matrix, d)["h_given_tests"]
        h_prior = localisation(matrix, d, prior=faults.rate)["h_given_tests"]
    return Score(tuple(tests), d, table, h_uniform, h_prior)


class ConditionSampler:
    """Alarms of the replicas of a fault under `conditions`, simulated on demand.

    As `metrics.adaptive.SpiceSampler`, but for any operating conditions and test names
    (the measurement names of the conditions, see `prefixed`), and with an optional
    `transform` of the test matrix (for example `costmodel.observe`) applied before the
    detector. The detector is fitted once on fault-free replicas 0..n_calibration-1, and
    fault-free replicas asked for afterwards start after them. Each replica is simulated
    once and kept; `simulated` counts the replicas simulated. A failed simulation counts
    as undetected for a fault and as an alarm for a fault-free circuit. With `record`,
    `alarming` keeps, per fault, the tests that left their limits on any of its replicas,
    as for `SpiceSampler`.
    """

    def __init__(
        self,
        circuit: Circuit,
        faults: Iterable[Fault],
        conditions: Sequence[OperatingCondition],
        tests: Sequence[str],
        detector,
        variations: VariationSet | Sequence = (),
        n_calibration: int = 200,
        n_max: int = 1600,
        seed: int = 0,
        transform: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
        workers: int = 1,
        record: bool = False,
        **experiment,
    ):
        self.ids = list(tests)
        self.n_calibration = n_calibration
        self.transform = transform
        self.alarming: dict[str, set[str]] | None = {} if record else None
        self.experiment = Experiment(
            circuit,
            faults=list(faults),
            variations=variations,
            conditions=tuple(conditions),
            samples=n_max,
            healthy_samples=n_calibration + n_max,
            seed=seed,
            **experiment,
        )
        self.index: dict[tuple[str, int], list] = {}
        for sample in self.experiment.plan():
            key = (self.experiment.fault_id(sample), sample.replica)
            self.index.setdefault(key, []).append(sample)
        self.pool = self.experiment.pool(workers) if workers > 1 else None
        self.simulated = 0
        self.cache: dict[tuple[str, int], bool] = {}
        calibration = self._matrix(HEALTHY, range(n_calibration))
        self.angles = angles_of(calibration, calibration.index, self.ids)
        self.detector = detector.fit(values(calibration, self.ids, self.angles))
        if record:  # fail now, not on the first fault, if the detector has no limits
            outside(self.detector, np.empty((0, len(self.ids))))

    def _matrix(self, fault_id: str, replicas) -> pd.DataFrame:
        replicas = list(replicas)
        samples = [s for r in replicas for s in self.index[fault_id, r]]
        result = self.experiment.run_samples(samples, pool=self.pool)
        self.simulated += len(set(replicas))
        m = candidates.matrix(result.to_frame(), self.ids)
        return self.transform(m) if self.transform is not None else m

    def __call__(self, fault_id: str, start: int, stop: int) -> np.ndarray:
        """Alarms of replicas start..stop-1; replicas already simulated are not simulated
        again."""
        offset = self.n_calibration if fault_id == HEALTHY else 0
        replicas = list(range(start + offset, stop + offset))
        new = [r for r in replicas if (fault_id, r) not in self.cache]
        if new:
            m = self._matrix(fault_id, new).set_index("replica")
            x = values(m, self.ids, self.angles)
            alarms = pd.Series(np.asarray(self.detector.flag(x), dtype=bool), index=m.index)
            failed_as = fault_id == HEALTHY
            for r, alarm in alarms.reindex(new, fill_value=failed_as).items():
                self.cache[fault_id, r] = bool(alarm)
            if self.alarming is not None and fault_id != HEALTHY:
                alarming = self.alarming.setdefault(fault_id, set())
                left = outside(self.detector, x)
                alarming.update(t for t, o in zip(self.ids, left, strict=True) if o)
        return np.array([self.cache[fault_id, r] for r in replicas], dtype=bool)

    def close(self) -> None:
        """Stop the worker processes, if any."""
        if self.pool is not None:
            self.pool.shutdown()
            self.pool = None

    def __enter__(self) -> ConditionSampler:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
