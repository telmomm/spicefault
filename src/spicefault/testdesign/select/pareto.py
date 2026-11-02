"""Minimum-cost self-test plan for a coverage target by a greedy rule, and the
cost–coverage Pareto front.

The greedy adds, at each step, the candidate test with the largest gain in the metric
per euro added, where both are measured on the selection circuits
(`metrics.formulation.part`) and the cost is what the plan adds to the board (for
example `costmodel.cost`): a test at a node already read, under a stimulus already
applied, adds nothing. Tests that would break the test time limit are skipped. It stops
when the target is reached on the selection circuits or no test raises the metric; the
claim itself is made afterwards on other circuits (`metrics.formulation.score` on the
evaluation half, or weighted fault sampling).

The scorer and the coster are functions of a plan, so the greedy depends neither on the
cost model nor on the metric. It is a heuristic: `select.ilp` gives the optimum of the
same problem, against which its gap can be reported.

Every prefix of the greedy path is a plan; the Pareto front keeps those that no other
plan beats on both cost and coverage.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import pandas as pd

__all__ = ["Coster", "Scorer", "Step", "cheapest", "front", "greedy"]

Scorer = Callable[[Sequence[str]], float]
Coster = Callable[[Sequence[str]], tuple[float, float]]


@dataclass(frozen=True)
class Step:
    """The plan after one greedy step: its tests, euros, test time (s) and score on the
    selection circuits."""

    tests: tuple[str, ...]
    euros: float
    time: float
    score: float


def greedy(
    pool: Sequence[str],
    scorer: Scorer,
    coster: Coster,
    target: float | None = None,
    max_time: float | None = None,
    epsilon: float = 0.01,
    max_tests: int | None = None,
) -> list[Step]:
    """The greedy path from the empty plan.

    `scorer(tests)` gives the metric of a plan on the selection circuits and
    `coster(tests)` its (euros, seconds). A test is chosen by gain / (added euros +
    `epsilon`): `epsilon`, in euros, ranks the tests that add no hardware by their gain
    alone; among equal ratios the one that adds less time wins. Tests that take the plan
    over `max_time` are skipped. The path stops at `target`, at `max_tests`, or when no
    test raises the score; a scorer can return −inf to forbid a plan (for example one
    over a false alarm budget).
    """
    chosen: list[str] = []
    score, (euros, time) = 0.0, (0.0, 0.0)
    path: list[Step] = []
    remaining = list(pool)
    while remaining and (max_tests is None or len(chosen) < max_tests):
        best = None
        for t in remaining:
            e, s = coster(chosen + [t])
            if max_time is not None and s > max_time:
                continue
            gain = scorer(chosen + [t]) - score
            if gain <= 0:
                continue
            key = (gain / (e - euros + epsilon), -(s - time))
            if best is None or key > best[0]:
                best = (key, t, e, s, score + gain)
        if best is None:
            break
        _, t, euros, time, score = best
        chosen.append(t)
        remaining.remove(t)
        path.append(Step(tuple(chosen), euros, time, score))
        if target is not None and score >= target:
            break
    return path


def front(plans: pd.DataFrame, cost: str = "euros", value: str = "coverage") -> pd.DataFrame:
    """The plans that no other plan beats: none costs as little or less and covers as
    much or more, with one of the two strictly better. Sorted by cost."""
    ordered = plans.sort_values([cost, value], ascending=[True, False])
    keep, best = [], -float("inf")
    for i, row in ordered.iterrows():
        if row[value] > best:
            keep.append(i)
            best = row[value]
    return ordered.loc[keep]


def cheapest(
    plans: pd.DataFrame, target: float, value: str = "lower", cost: str = "euros"
) -> pd.Series | None:
    """The cheapest plan whose `value` (for example the lower bound on the evaluation
    circuits) reaches `target`, with fewer tests and then less time breaking ties; None
    if none does. `plans` has the columns `cost`, `value`, `tests` (a count) and `time`."""
    reached = plans[plans[value] >= target]
    if reached.empty:
        return None
    return reached.sort_values([cost, "tests", "time"]).iloc[0]
