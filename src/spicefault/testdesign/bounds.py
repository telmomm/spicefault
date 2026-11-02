"""Theoretical bound of a self-test: sensitivity matrix, testability and canonical
ambiguity groups by SVD, and the testability gap of a plan.

The sensitivity matrix S has one row per candidate test and one column per component:
S[t, c] = d y_t / d ln p_c, the change of the reading for a relative change of the
component, estimated by central differences (`perturbations`, `sensitivity`). Following
Fedi, Manetti and Piccirilli, the testability of a set of tests is the rank of its rows,
the number of component values they can identify; a canonical ambiguity group is a
minimal set of components whose columns are linearly dependent, so that no reading of
those tests can tell their deviations apart. The numerical rank and the null space come
from the SVD (Manetti and Piccirilli, 2003).

Two ranks are reported (`bound`). The algebraic rank is the classic testability: the
bound with ideal measurements. It is taken on S with each row scaled to unit norm, so
that the units of the tests (dB, degrees, volts) do not matter, and counts the singular
values above `rtol` of the largest. The default `rtol` of 1 % sits above the effects
that a numerical S cannot separate from a true dependence: the error of the central
differences and the small departures of an op-amp macromodel from an ideal op-amp; a
smaller tolerance counts that numerical noise as rank. The resolvable rank divides each
row by the fault-free spread of its test and counts the directions along which a
relative change of `deviation` moves the readings by more than `k` spreads: the bound
with the measurements the board actually has.

`gap` says what a plan leaves out of the bound of all the candidate tests: a plan chosen
for detection can reach the coverage target and still identify no component.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from ..faults import ParametricFault
from .metrics.detection import is_circular

__all__ = ["RTOL", "STEP", "Bound", "ambiguity_groups", "bound", "gap", "perturbations",
           "sensitivity"]  # fmt: skip

STEP = 0.01
RTOL = 1e-2


def perturbations(components: Sequence[str], step: float = STEP) -> list[ParametricFault]:
    """The faults that move each component by ±`step`, for `sensitivity`."""
    return [ParametricFault(c, deviation=d) for c in components for d in (step, -step)]


def sensitivity(
    matrix: pd.DataFrame, tests: Sequence[str], components: Sequence[str], step: float = STEP
) -> pd.DataFrame:
    """S[t, c] = d y_t / d ln p_c by central differences, from a test matrix with one row
    per perturbation of `perturbations` (replica 0, no tolerances). Phase differences
    are wrapped to ±180 degrees."""
    rows = matrix.set_index("fault_id")
    span = math.log(1 + step) - math.log(1 - step)
    out = {}
    for c in components:
        up = rows.loc[f"{c}:parametric:+{step:g}", list(tests)]
        down = rows.loc[f"{c}:parametric:-{step:g}", list(tests)]
        diff = (up - down).astype(float)
        for t in tests:
            if is_circular(t):
                diff[t] = (diff[t] + 180) % 360 - 180
        out[c] = diff / span
    return pd.DataFrame(out, index=list(tests))


def _normalised(a: np.ndarray) -> np.ndarray:
    """The rows of `a` scaled to unit norm, without the rows that are zero."""
    norms = np.linalg.norm(a, axis=1)
    return a[norms > 0] / norms[norms > 0, None]


def _rank(a: np.ndarray, rtol: float) -> int:
    if a.size == 0:
        return 0
    s = np.linalg.svd(a, compute_uv=False)
    return int((s > rtol * s[0]).sum()) if s[0] > 0 else 0


@dataclass(frozen=True)
class Bound:
    """Testability of a set of tests: algebraic and resolvable rank, the singular values
    of the noise-scaled matrix, and the canonical ambiguity groups."""

    components: tuple[str, ...]
    rank: int
    resolvable: int
    singular: np.ndarray
    groups: tuple[tuple[str, ...], ...]
    ambiguous: frozenset[str]

    @property
    def identifiable(self) -> tuple[str, ...]:
        """Components in no canonical ambiguity group: their deviation can be told from
        any other's."""
        return tuple(c for c in self.components if c not in self.ambiguous)


def ambiguity_groups(
    S: np.ndarray, components: Sequence[str], rtol: float = RTOL, max_size: int = 6
) -> tuple[list[tuple[str, ...]], frozenset[str]]:
    """Canonical ambiguity groups (minimal sets of columns of S that are linearly
    dependent at the tolerance `rtol`) and the components that belong to some group.

    Only components with some weight in the null space can belong to one, so the sets
    are enumerated within that support (a loose threshold, since a numerical null space
    has small entries everywhere), by size up to `max_size`: a group larger than that is
    not found, and its components count as identifiable."""
    if S.size == 0:
        return [tuple(components)], frozenset(components)
    s0 = np.linalg.svd(S, compute_uv=False)[0]
    scale = max(s0, 1e-300)
    zero = [i for i in range(S.shape[1]) if np.linalg.norm(S[:, i]) <= rtol * scale]
    groups = [(i,) for i in zero]
    _, s, vt = np.linalg.svd(S, full_matrices=True)
    rank = int((s > rtol * scale).sum())
    null = vt[rank:].T
    support = [i for i in range(S.shape[1]) if i not in zero and np.linalg.norm(null[i]) > 1e-3]
    for k in range(2, min(max_size, len(support)) + 1):
        for subset in combinations(support, k):
            if any(set(g) <= set(subset) for g in groups):
                continue
            sub = S[:, subset]
            if (np.linalg.svd(sub, compute_uv=False) > rtol * scale).sum() < k:
                groups.append(subset)
    ambiguous = frozenset(components[i] for g in groups for i in g)
    return [tuple(components[i] for i in g) for g in groups], ambiguous


def bound(
    S: pd.DataFrame,
    noise: pd.Series,
    deviation: float = 0.1,
    k: float = 3.0,
    rtol: float = RTOL,
    max_size: int = 6,
) -> Bound:
    """The bound of the tests in the rows of S. `noise` is the fault-free spread of each
    test, in the units of its row; a direction is resolvable when a relative change of
    `deviation` along it moves the readings by more than `k` spreads."""
    a = S.to_numpy(float)
    scaled = a / noise.reindex(S.index).to_numpy(float)[:, None]
    singular = np.linalg.svd(scaled, compute_uv=False) if a.size else np.zeros(0)
    resolvable = int((singular * math.log(1 + deviation) > k).sum())
    unit = _normalised(a)
    groups, ambiguous = ambiguity_groups(unit, list(S.columns), rtol, max_size)
    return Bound(
        tuple(S.columns), _rank(unit, rtol), resolvable, singular, tuple(groups), ambiguous
    )


def gap(full: Bound, plan: Bound) -> dict[str, int]:
    """What a plan leaves out of the bound of all the tests: the ranks and the number of
    identifiable components it does not reach."""
    return {
        "rank": full.rank - plan.rank,
        "resolvable": full.resolvable - plan.resolvable,
        "identifiable": len(full.identifiable) - len(plan.identifiable),
    }
