"""Exact minimum-cost self-test plan by integer linear programming.

To make coverage linear, the detection of each test is fixed once: its limits are set
alone, on the calibration fault-free circuits, at a false alarm rate of
`alpha_per_test`, and a plan raises an alarm when any of its tests does. With at most
`max_tests` tests and `alpha_per_test = alpha / max_tests`, the union bound keeps the
false alarm rate of any plan within `alpha` in theory; since the limits of a test with
heavy tails can leak more than that, the false alarm rate of the plan on the held-out
fault-free circuits can also be bounded exactly. On that fixed detection matrix:

    minimise    euros(plan), then test time(plan), then |plan|, then -coverage(plan)
    subject to  coverage(plan) >= target,  |plan| <= max_tests,
                false alarms(plan) <= max_false_alarm (optional),
                euros(plan) <= max_euros (optional)

where coverage is the metric (DC_w, DC, SFF or SPFM) as a mean over the faulty circuits
weighted by the fault weights (`metrics.adaptive.draw_weights`), and euros and time are
those of `costmodel.cost`, written linearly: a node read adds a channel and, if it needs
one, a buffer (internal op-amps first); channels beyond the free ones need one 8:1
multiplexer per seven; each stimulus source needs a switch; AC tests need one channel for
the stimulus; time is the sum of the acquisitions the plan uses. The objectives are
optimised one after another, each kept at its optimum while the next is, so that ties
are not broken by the solver.

The same `Instance` gives the coverage, false alarm rate, euros and time of any plan,
and `Instance.subset` restricts it to some tests, so a heuristic (`select.pareto`) is
scored on exactly the same problem as the optimum. `brute_force` solves it by
enumeration, for checks.

`solve` needs PuLP and the HiGHS solver: ``pip install "spicefault[ilp]"``. The rest
of the module needs neither.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import combinations

import numpy as np
import pandas as pd

from ..candidates import Test
from ..costmodel import Access, Microcontroller
from ..metrics.adaptive import draw_weights
from ..metrics.detection import HEALTHY, GaussianLimits, detect, split_healthy

__all__ = ["Instance", "Solution", "brute_force", "detection", "instance", "solve"]


@dataclass(frozen=True)
class Instance:
    """A selection problem on a fixed detection matrix.

    `detected` is [faulty circuits, tests] and `held_out` [held-out fault-free circuits,
    tests]; `weight` the share of the metric each faulty circuit carries and `free` the
    share of faults that count as covered without a test (safe faults for SFF, faults
    that are not single-point for SPFM). The rest describes the tests for the cost: node,
    stimulus sources, AC or not, acquisition and its time, which nodes need a buffer,
    the free channels and op-amps and the prices. `false_alarm` is the false alarm rate
    of each test alone.
    """

    tests: tuple[str, ...]
    detected: np.ndarray
    weight: np.ndarray
    free: float
    node: tuple[str, ...]
    sources: tuple[tuple[str, ...], ...]
    ac: tuple[bool, ...]
    acquisition: tuple[str, ...]
    duration: Mapping[str, float]
    buffer: Mapping[str, bool]
    channels: int
    opamps: int
    price: Mapping[str, float]
    false_alarm: np.ndarray
    held_out: np.ndarray

    def _columns(self, tests: Sequence[str]) -> list[int]:
        return [self.tests.index(t) for t in tests]

    def coverage(self, tests: Sequence[str]) -> float:
        """The metric of a plan on the faulty circuits of the instance."""
        cols = self._columns(tests)
        if not cols:
            return self.free
        return self.free + float(self.weight @ self.detected[:, cols].any(axis=1))

    def false_alarm_rate(self, tests: Sequence[str]) -> float:
        """Share of the held-out fault-free circuits on which any test of the plan alarms."""
        cols = self._columns(tests)
        return float(self.held_out[:, cols].any(axis=1).mean()) if cols else 0.0

    def euros(self, tests: Sequence[str]) -> float:
        """Hardware added to the board, as `costmodel.cost`."""
        idx = self._columns(tests)
        nodes = {self.node[i] for i in idx}
        buffers = max(sum(self.buffer[n] for n in nodes) - self.opamps, 0)
        excess = len(nodes) + any(self.ac[i] for i in idx) - self.channels
        muxes = math.ceil(excess / 7) if excess > 0 else 0
        sources = {s for i in idx for s in self.sources[i]}
        return round(
            buffers * self.price["external_opamp_channel"] + muxes * self.price["mux_8"]
            + len(sources) * self.price["injection_switch"],
            6,
        )  # fmt: skip

    def time(self, tests: Sequence[str]) -> float:
        """Test time: the sum of the acquisitions the plan uses."""
        keys = {self.acquisition[i] for i in self._columns(tests)}
        return sum(self.duration[k] for k in keys)

    def subset(self, tests: Sequence[str]) -> Instance:
        """The instance restricted to `tests`, with the same detection of each test."""
        keep = self._columns(tests)
        return replace(
            self,
            tests=tuple(self.tests[i] for i in keep),
            detected=self.detected[:, keep],
            node=tuple(self.node[i] for i in keep),
            sources=tuple(self.sources[i] for i in keep),
            ac=tuple(self.ac[i] for i in keep),
            acquisition=tuple(self.acquisition[i] for i in keep),
            false_alarm=self.false_alarm[keep],
            held_out=self.held_out[:, keep],
        )


def detection(
    part: pd.DataFrame, tests: Sequence[str], alpha_per_test: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The fault of each faulty row of `part`, and the alarms of each test alone on the
    faulty rows and on the held-out fault-free rows, with its limits
    (`GaussianLimits(alpha_per_test)`) set on the calibration fault-free rows."""
    faulty = part.fault_id != HEALTHY
    _, held_out = split_healthy(part)
    columns, healthy = [], []
    for t in tests:
        d = detect(part, [t], GaussianLimits(alpha_per_test))
        columns.append(d.alarms[faulty].to_numpy())
        healthy.append(d.alarms[held_out].to_numpy())
    return part[faulty].fault_id.to_numpy(), np.column_stack(columns), np.column_stack(healthy)


def _acquisition(test: Test) -> str:
    """Tests that share an acquisition: gain and phase of one tone at one node; every
    reading of one step at one node; one DC level at one node."""
    feature = test.feature.name
    if test.stimulus.kind == "ac":
        return f"tone|{test.node}|{feature.split('@')[1]}"
    return f"{test.stimulus.name}|{test.node}"


def instance(
    part: pd.DataFrame,
    tests: Sequence[Test],
    faults: pd.DataFrame,
    metric: str,
    access: Mapping[str, Access],
    mcu: Microcontroller,
    price: Mapping[str, float],
    alpha_per_test: float,
    duration_of: Callable[[Test], float],
) -> Instance:
    """Build an instance from a part of the test matrix (`metrics.formulation.part`),
    the candidate tests, the fault table and the cost model: the access of each node,
    the board and the prices of `costmodel`. `duration_of(test)` gives the time of the
    acquisition a test belongs to, for example `costmodel.cost([test], ...).time`."""
    ids = [t.test_id for t in tests]
    rows, detected, held_out = detection(part, ids, alpha_per_test)
    probability, free = draw_weights(faults, metric)
    counts = pd.Series(rows).value_counts()
    # each faulty circuit carries the weight of its fault shared among its replicas
    weight = np.array([0.0 if free[f] else probability[f] / counts[f] for f in rows])
    keys = [_acquisition(t) for t in tests]
    return Instance(
        tests=tuple(ids), detected=detected, weight=weight, free=float(probability[free].sum()),
        node=tuple(t.node for t in tests),
        sources=tuple(tuple(sorted({s for s, _ in t.stimulus.settings})) for t in tests),
        ac=tuple(t.stimulus.kind == "ac" for t in tests), acquisition=tuple(keys),
        duration={k: duration_of(t) for k, t in zip(keys, tests, strict=True)},
        buffer={n: access[n].buffer for n in {t.node for t in tests}},
        channels=mcu.channels, opamps=mcu.opamps, price=dict(price),
        false_alarm=held_out.mean(axis=0), held_out=held_out,
    )  # fmt: skip


@dataclass(frozen=True)
class Solution:
    """A plan and its euros, time, coverage and false alarm rate on the instance.
    `status` is optimal, infeasible or the status the solver gave."""

    status: str
    tests: tuple[str, ...]
    euros: float
    time: float
    coverage: float
    false_alarm: float = math.nan


def _pulp():
    try:
        import pulp
    except ImportError as error:
        raise ImportError(
            'select.ilp.solve needs PuLP and HiGHS: pip install "spicefault[ilp]"'
        ) from error
    return pulp


def solve(
    inst: Instance,
    target: float | None = None,
    max_tests: int = 12,
    time_limit: float = 120.0,
    maximise: bool = False,
    max_false_alarm: float | None = None,
    max_euros: float | None = None,
) -> Solution:
    """The plan of least euros, then least time, then fewest tests, then most coverage,
    whose coverage reaches `target`; or, with `maximise`, the plan of greatest coverage
    with at most `max_tests` tests (then least euros, time and tests). With
    `max_false_alarm`, the plan alarms on at most that share of the held-out fault-free
    circuits; with `max_euros`, it adds at most that much hardware. `time_limit` is the
    limit of the solver in seconds, for each stage."""
    if target is None and not maximise:
        raise ValueError("give a target, or maximise")
    pulp = _pulp()
    T, C = len(inst.tests), len(inst.weight)
    nodes = sorted(set(inst.node))
    sources = sorted({s for ss in inst.sources for s in ss})
    keys = sorted(set(inst.acquisition))
    p = pulp.LpProblem("plan", pulp.LpMinimize)
    x = [p.add_variable(f"x{i}", cat="Binary") for i in range(T)]
    y = {n: p.add_variable(f"y{j}", cat="Binary") for j, n in enumerate(nodes)}
    s = {k: p.add_variable(f"s{j}", cat="Binary") for j, k in enumerate(sources)}
    a = {k: p.add_variable(f"a{j}", cat="Binary") for j, k in enumerate(keys)}
    r = p.add_variable("ref", cat="Binary")
    buf = p.add_variable("buffers", lowBound=0, cat="Integer")
    mux = p.add_variable("muxes", lowBound=0, cat="Integer")
    # only circuits that some test detects and that carry weight need a variable
    useful = [c for c in range(C) if inst.weight[c] > 0 and inst.detected[c].any()]
    z = {c: p.add_variable(f"z{c}", lowBound=0, upBound=1) for c in useful}
    for c in useful:
        p += z[c] <= pulp.lpSum(x[i] for i in np.flatnonzero(inst.detected[c]))
    for i in range(T):
        p += x[i] <= y[inst.node[i]]
        p += x[i] <= a[inst.acquisition[i]]
        for src in inst.sources[i]:
            p += x[i] <= s[src]
        if inst.ac[i]:
            p += x[i] <= r
    p += buf >= pulp.lpSum(y[n] for n in nodes if inst.buffer[n]) - inst.opamps
    p += 7 * mux >= pulp.lpSum(y.values()) + r - inst.channels
    p += pulp.lpSum(x) <= max_tests
    if max_false_alarm is not None:
        # u_h = 1 if any chosen test alarms on held-out fault-free circuit h
        alarming = [h for h in range(len(inst.held_out)) if inst.held_out[h].any()]
        u = {h: p.add_variable(f"u{h}", lowBound=0, upBound=1) for h in alarming}
        for h in alarming:
            for i in np.flatnonzero(inst.held_out[h]):
                p += u[h] >= x[i]
        p += pulp.lpSum(u.values()) <= max_false_alarm * len(inst.held_out)
    euros = (
        inst.price["external_opamp_channel"] * buf + inst.price["mux_8"] * mux
        + inst.price["injection_switch"] * pulp.lpSum(s.values())
    )  # fmt: skip
    time = pulp.lpSum(inst.duration[k] * a[k] for k in keys)
    cover = inst.free + pulp.lpSum(float(inst.weight[c]) * z[c] for c in useful)
    if max_euros is not None:
        p += euros <= max_euros + 1e-9
    solver = pulp.HiGHS(msg=False, timeLimit=time_limit)

    stages = []
    if maximise:
        stages.append(-cover)
    else:
        p += cover >= target
    stages += [euros, time, pulp.lpSum(x)]
    if not maximise:
        stages.append(-cover)
    for k, objective in enumerate(stages):
        p.setObjective(objective)
        status = p.solve(solver).status.name.lower()
        if status != "optimal":
            return Solution(status, (), math.nan, math.nan, math.nan)
        value = pulp.value(objective)
        # keep this stage at its optimum while the next one is optimised
        p += objective <= value + 1e-6 * max(1.0, abs(value)), f"stage{k}"
    chosen = tuple(t for t, v in zip(inst.tests, x, strict=True) if v.value() > 0.5)
    return Solution(
        "optimal", chosen, inst.euros(chosen), inst.time(chosen), inst.coverage(chosen),
        inst.false_alarm_rate(chosen),
    )  # fmt: skip


def brute_force(
    inst: Instance, target: float, max_tests: int, max_false_alarm: float | None = None
) -> Solution:
    """The same problem by enumeration, for small pools: least euros, then least time,
    then fewest tests, then most coverage."""
    best = None
    for k in range(1, max_tests + 1):
        for plan in combinations(inst.tests, k):
            if inst.coverage(plan) + 1e-9 < target:
                continue
            fa = inst.false_alarm_rate(plan)
            if max_false_alarm is not None and fa > max_false_alarm + 1e-12:
                continue
            key = (inst.euros(plan), round(inst.time(plan), 9), k, -round(inst.coverage(plan), 9))
            if best is None or key < best[0]:
                best = (key, plan)
    if best is None:
        return Solution("infeasible", (), math.nan, math.nan, math.nan)
    plan = best[1]
    return Solution(
        "optimal", plan, inst.euros(plan), inst.time(plan), inst.coverage(plan),
        inst.false_alarm_rate(plan),
    )  # fmt: skip
