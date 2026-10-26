"""Stopping rules: simulate until the claim "coverage ≥ target" is decided.

The rules decide how many replicas of each fault to read, never which circuits exist:
replica r of fault k is the circuit that the experiment draws for it, from its own
stream, whatever the order in which replicas are read, the number of workers or where
a run stopped. Each fault is read in stages of increasing size (`schedule`), and after
every stage the bounds of the metric are recomputed:

- `certify` stops when the lower bound of the metric reaches the target (certified)
  or its upper bound falls below it (not reachable), and otherwise grows only the
  faults that cost most in the lower bound. The per-fault bounds are one-sided at
  alpha / (K L), for K faults and L stages, so they hold for every fault at every stage
  at once: looking at the data after each stage does not inflate the chance of a wrong
  claim.
- `certify_sampled` draws the fault of each circuit with the weights of the metric, so
  that every circuit is one Bernoulli trial of the metric itself and one interval on
  the total is exact: far fewer circuits than bounding every fault, but no claim about
  any one fault.
- `estimate` grows each fault until its interval is narrower than a width. It estimates
  and makes no claim.
- `certify_false_alarm` decides whether the false alarm rate on fault-free circuits is
  below a target.

A sampler gives the alarms of replicas `start..stop-1` of a fault; `SpiceSampler`
simulates them, and any function with the same signature (for example a Bernoulli draw
with a known probability) can be used to check a rule.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ...circuit import Circuit
from ...experiments import Experiment
from ...faults import Fault
from ...reliability import clopper_pearson_interval
from ...variation import VariationSet
from .. import candidates
from .coverage import metric_values
from .detection import HEALTHY, angles_of, values

Sampler = Callable[[str, int, int], np.ndarray]

__all__ = [
    "Certification",
    "Sampler",
    "SpiceSampler",
    "certify",
    "certify_false_alarm",
    "certify_sampled",
    "estimate",
    "schedule",
]


def schedule(n_start: int = 25, n_max: int = 1600, growth: float = 2.0) -> list[int]:
    """Stage sizes n_start, n_start · growth, ... up to n_max, which is always the last."""
    sizes, n = [], n_start
    while n < n_max:
        sizes.append(int(n))
        n = math.ceil(n * growth)
    return [*sizes, n_max]


def _bounds(k: np.ndarray, n: np.ndarray, one_sided: float) -> tuple[np.ndarray, np.ndarray]:
    """One-sided Clopper–Pearson bounds at level 1 − one_sided on each side."""
    level = 1 - 2 * one_sided
    pairs = [clopper_pearson_interval(ki, ni, level) for ki, ni in zip(k, n, strict=True)]
    return np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])


@dataclass(frozen=True)
class Certification:
    """Result of `certify`, `certify_sampled` or `estimate`.

    `decision` is certified, not reachable or undecided (estimated or width not
    reached for `estimate`). `per_fault` has, per fault, the circuits simulated, the
    alarms, the estimate and the bounds used for the decision; `history` one row per
    stage with the circuits simulated so far and the estimate and bounds of the metric.
    """

    decision: str
    metric: str
    target: float | None
    confidence: float
    per_fault: pd.DataFrame
    history: pd.DataFrame

    @property
    def simulated(self) -> int:
        return int(self.per_fault.n.sum())

    @property
    def lower(self) -> float:
        return float(self.history.lower.iloc[-1])

    @property
    def upper(self) -> float:
        return float(self.history.upper.iloc[-1])

    @property
    def estimate(self) -> float:
        return float(self.history.estimate.iloc[-1])


def _metric(P: np.ndarray, faults: pd.DataFrame, metric: str) -> float:
    return metric_values(pd.Series(P, index=faults.index), faults)[metric]


def certify(
    sampler: Sampler,
    faults: pd.DataFrame,
    target: float,
    metric: str = "dc_w",
    confidence: float = 0.95,
    sizes: Sequence[int] | None = None,
    share: float = 0.5,
) -> Certification:
    """Decide whether `metric` ≥ `target` holds at `confidence`, simulating as little as
    possible.

    `faults` is a `fault_table`. Every fault starts at the first stage size. After each
    round, the faults whose contribution to the gap between the estimate and the lower
    bound is at least `share` times the largest one move to their next stage size. It
    stops when the claim is decided or no fault can grow.
    """
    sizes = list(sizes or schedule())
    ids = list(faults.index)
    K, L = len(ids), len(sizes)
    one_sided = (1 - confidence) / (K * L)
    stage = np.zeros(K, dtype=int)
    n = np.zeros(K, dtype=int)
    k = np.zeros(K, dtype=int)
    history = []

    def grow(j: int) -> None:
        stop = sizes[stage[j]]
        alarms = np.asarray(sampler(ids[j], int(n[j]), stop), dtype=bool)
        k[j] += int(alarms.sum())
        n[j] = stop

    for j in range(K):
        grow(j)
    decision = "undecided"
    while True:
        low, high = _bounds(k, n, one_sided)
        p = k / n
        estimate, lower, upper = (_metric(v, faults, metric) for v in (p, low, high))
        history.append(
            {"round": len(history), "simulated": int(n.sum()), "estimate": estimate,
             "lower": lower, "upper": upper, "max_n": int(n.max())}
        )
        if lower >= target:
            decision = "certified"
            break
        if upper < target:
            decision = "not reachable"
            break
        open_ = stage < L - 1
        if not open_.any():
            break
        # what each fault costs in the lower bound: the metric with only that fault at its bound
        alone = (np.where(np.arange(K) == j, low, p) for j in range(K))
        cost = np.array([estimate - _metric(v, faults, metric) for v in alone])
        cost[~open_] = -np.inf
        chosen = np.flatnonzero(open_ & (cost >= share * cost[open_].max()))
        for j in chosen:
            stage[j] += 1
            grow(j)
    low, high = _bounds(k, n, one_sided)
    per_fault = pd.DataFrame(
        {"fault_id": ids, "n": n, "detected": k, "p": k / n, "lower": low, "upper": high}
    ).set_index("fault_id")
    return Certification(decision, metric, target, confidence, per_fault, pd.DataFrame(history))


def _draw_weights(faults: pd.DataFrame, metric: str) -> tuple[pd.Series, pd.Series]:
    """For `certify_sampled`: the probability of drawing each fault, and whether a draw of
    it counts as a success without simulating it (a fault that cannot lower the metric).
    """
    rate = faults.rate.astype(float)
    free = pd.Series(False, index=faults.index)
    if metric == "dc_w":
        weight = rate * faults.severity.astype(float)
    elif metric == "dc":
        weight = rate.where(faults.iec == "dangerous", 0.0)
    elif metric == "sff":
        weight, free = rate, faults.iec == "safe"
    elif metric == "spfm":
        weight, free = rate, faults.iso != "single_point"
    else:
        raise ValueError(f"{metric!r} is not a mean over faults; expected dc_w, dc, sff or spfm")
    if weight.sum() <= 0:
        raise ValueError(f"no fault has weight for {metric}")
    return weight / weight.sum(), free


def certify_sampled(
    sampler: Sampler,
    faults: pd.DataFrame,
    target: float,
    metric: str = "dc_w",
    confidence: float = 0.95,
    sizes: Sequence[int] | None = None,
    seed: int = 0,
    correct_looks: bool = True,
) -> Certification:
    """Decide whether `metric` ≥ `target` by drawing the fault of each circuit at random.

    DC_w, DC, SFF and SPFM are means over the faults weighted by their rates (and, for
    DC_w, severities). If the fault of each simulated circuit is drawn with those
    weights, each circuit is a success (an alarm, or a fault that cannot lower the
    metric) with probability equal to the metric, so one Clopper–Pearson interval on the
    total is exact, with no bound per fault. The sizes are numbers of circuits in total,
    and the bounds are corrected for the number of stages; `correct_looks=False` drops
    that correction, which is wrong and only serves to show why it is needed. It says
    nothing about any one fault. The faults are drawn from a generator seeded with
    `seed`, so the same call reads the same replicas.
    """
    sizes = list(sizes or schedule(50, 6400))
    probability, free = _draw_weights(faults, metric)
    rng = np.random.default_rng(seed)
    draws = rng.choice(len(faults), size=sizes[-1], p=probability.to_numpy())
    one_sided = (1 - confidence) / (len(sizes) if correct_looks else 1)
    ids = list(faults.index)
    used = dict.fromkeys(ids, 0)
    detected = dict.fromkeys(ids, 0)
    drawn = dict.fromkeys(ids, 0)
    successes = n = 0
    history, decision = [], "undecided"
    for size in sizes:
        for j, count in sorted(pd.Series(draws[n:size]).value_counts().items()):
            fault_id, count = ids[j], int(count)
            drawn[fault_id] += count
            if free[fault_id]:
                successes += count
                continue
            stop = used[fault_id] + count
            alarms = np.asarray(sampler(fault_id, used[fault_id], stop), dtype=bool)
            used[fault_id] = stop
            detected[fault_id] += int(alarms.sum())
            successes += int(alarms.sum())
        n = size
        low, high = clopper_pearson_interval(successes, n, 1 - 2 * one_sided)
        history.append(
            {"round": len(history), "simulated": int(sum(used.values())),
             "estimate": successes / n, "lower": low, "upper": high,
             "max_n": max(used.values())}
        )
        if low >= target:
            decision = "certified"
            break
        if high < target:
            decision = "not reachable"
            break
    per_fault = pd.DataFrame(
        {"fault_id": ids, "drawn": [drawn[f] for f in ids], "n": [used[f] for f in ids],
         "detected": [detected[f] for f in ids]}
    ).set_index("fault_id")
    per_fault["p"] = per_fault.detected / per_fault.n.where(per_fault.n > 0)
    return Certification(decision, metric, target, confidence, per_fault, pd.DataFrame(history))


def estimate(
    sampler: Sampler,
    faults: pd.DataFrame,
    width: float,
    metric: str = "dc_w",
    confidence: float = 0.95,
    sizes: Sequence[int] | None = None,
) -> Certification:
    """Grow each fault until its two-sided Clopper–Pearson interval at `confidence` is at
    most `width` wide, or it reaches the last stage. The intervals are not corrected for
    the looks: this rule estimates, it does not certify.
    """
    sizes = list(sizes or schedule())
    ids = list(faults.index)
    n = np.zeros(len(ids), dtype=int)
    k = np.zeros(len(ids), dtype=int)
    tail = (1 - confidence) / 2
    for j, fault_id in enumerate(ids):
        for size in sizes:
            k[j] += int(np.asarray(sampler(fault_id, int(n[j]), size), dtype=bool).sum())
            n[j] = size
            low, high = clopper_pearson_interval(k[j], n[j], confidence)
            if high - low <= width:
                break
    low, high = _bounds(k, n, tail)
    p = k / n
    history = pd.DataFrame(
        [{"round": 0, "simulated": int(n.sum()), "estimate": _metric(p, faults, metric),
          "lower": _metric(low, faults, metric), "upper": _metric(high, faults, metric),
          "max_n": int(n.max())}]
    )
    per_fault = pd.DataFrame(
        {"fault_id": ids, "n": n, "detected": k, "p": p, "lower": low, "upper": high}
    ).set_index("fault_id")
    reached = ((per_fault.upper - per_fault.lower) <= width).all()
    decision = "estimated" if reached else "width not reached"
    return Certification(decision, metric, None, confidence, per_fault, history)


def certify_false_alarm(
    sampler: Sampler,
    target: float,
    confidence: float = 0.95,
    sizes: Sequence[int] | None = None,
) -> dict:
    """Decide whether the false alarm rate is at most `target`, growing the fault-free
    circuits stage by stage, with the bounds corrected for the number of stages.

    The decision is met, exceeded or undecided.
    """
    sizes = list(sizes or schedule())
    one_sided = (1 - confidence) / len(sizes)
    n = k = 0
    decision = "undecided"
    for size in sizes:
        k += int(np.asarray(sampler(HEALTHY, n, size), dtype=bool).sum())
        n = size
        low, high = clopper_pearson_interval(k, n, 1 - 2 * one_sided)
        if high <= target:
            decision = "met"
            break
        if low > target:
            decision = "exceeded"
            break
    return {"decision": decision, "n": n, "alarms": k, "p": k / n, "lower": low, "upper": high,
            "target": target}


class SpiceSampler:
    """Alarms of the replicas of a fault, simulated on demand.

    It builds one `Experiment` of `circuit` with `faults`, `variations` and one operating
    condition per stimulus of `tests` (see `candidates.conditions`), with `n_max`
    replicas per fault and `n_calibration + n_max` fault-free ones. The detector is
    fitted once, on fault-free replicas 0..n_calibration-1, and then kept fixed.
    Fault-free replicas asked for by `certify_false_alarm` start after the calibration
    ones.

    A replica is the same circuit whatever the order in which it is asked for, since its
    random stream depends only on the seed, the fault and the replica, and each replica
    is simulated once and kept. A failed simulation counts as undetected for a fault and
    as an alarm for a fault-free circuit, which is the conservative side for both claims.
    With `workers` > 1 the worker processes are kept until `close`, or the end of a
    `with` block.
    """

    def __init__(
        self,
        circuit: Circuit,
        faults: Iterable[Fault],
        tests: Sequence[candidates.Test],
        detector,
        variations: VariationSet | Sequence = (),
        n_calibration: int = 200,
        n_max: int = 1600,
        seed: int = 0,
        workers: int = 1,
        **experiment,
    ):
        self.tests = list(tests)
        self.ids = [t.test_id for t in self.tests]
        self.n_calibration = n_calibration
        self.experiment = Experiment(
            circuit,
            faults=list(faults),
            variations=variations,
            conditions=candidates.conditions(self.tests),
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

    def _matrix(self, fault_id: str, replicas) -> pd.DataFrame:
        replicas = list(replicas)
        samples = [s for r in replicas for s in self.index[fault_id, r]]
        result = self.experiment.run_samples(samples, pool=self.pool)
        self.simulated += len(set(replicas))
        return candidates.matrix(result.to_frame(), self.tests)

    def __call__(self, fault_id: str, start: int, stop: int) -> np.ndarray:
        """Alarms of replicas start..stop-1; replicas already simulated are not simulated
        again.
        """
        offset = self.n_calibration if fault_id == HEALTHY else 0
        replicas = list(range(start + offset, stop + offset))
        new = [r for r in replicas if (fault_id, r) not in self.cache]
        if new:
            m = self._matrix(fault_id, new).set_index("replica")
            flags = self.detector.flag(values(m, self.ids, self.angles))
            alarms = pd.Series(np.asarray(flags, dtype=bool), index=m.index)
            failed_as = fault_id == HEALTHY
            for r, alarm in alarms.reindex(new, fill_value=failed_as).items():
                self.cache[fault_id, r] = bool(alarm)
        return np.array([self.cache[fault_id, r] for r in replicas], dtype=bool)

    def close(self) -> None:
        """Stop the worker processes, if any."""
        if self.pool is not None:
            self.pool.shutdown()
            self.pool = None

    def __enter__(self) -> SpiceSampler:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
