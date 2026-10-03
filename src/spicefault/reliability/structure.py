"""Separation between fault conditions and the ambiguity structure it implies."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def connected_groups(names: list[str], pairs: Iterable[tuple[int, int]]) -> list[list[str]]:
    """Connected components of the graph on `names` with edges `pairs`, largest first."""
    parent = list(range(len(names)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in pairs:
        parent[find(i)] = find(j)
    groups: dict[int, list[str]] = {}
    for i, name in enumerate(names):
        groups.setdefault(find(i), []).append(name)
    return sorted(groups.values(), key=len, reverse=True)


def pairwise_shift(
    centre: pd.DataFrame, spread: pd.DataFrame, floor: pd.Series | None = None
) -> pd.DataFrame:
    """Pairwise d' between conditions: max over features of |c_i - c_j| / pooled spread.

    `centre` and `spread` have one row per condition and one column per feature.
    `floor` is a per-feature lower bound of the spread (the measurement noise), which
    keeps numerically identical clouds from looking infinitely separable.
    """
    c, var = centre.to_numpy(), spread.to_numpy() ** 2
    if floor is not None:
        var = np.maximum(var, floor.reindex(centre.columns).to_numpy() ** 2)
    diff = np.abs(c[:, None, :] - c[None, :, :])
    pooled = np.sqrt(0.5 * (var[:, None, :] + var[None, :, :]))
    d = np.max(diff / np.maximum(pooled, 1e-300), axis=2)
    return pd.DataFrame(d, index=centre.index, columns=centre.index)


def _close_pairs(d: pd.DataFrame, threshold: float):
    close = np.triu(d.to_numpy() < threshold, k=1)
    return zip(*np.nonzero(close), strict=True)


def ambiguity_groups(d: pd.DataFrame, threshold: float = 3.0) -> list[list[str]]:
    """Connected components of the graph linking conditions with d' < threshold.

    Linking is transitive, so a chain of similar conditions merges into one group: the
    groups are an upper bound of the ambiguity.
    """
    return connected_groups(list(d.index), _close_pairs(d, threshold))


def confusable_components(
    d: pd.DataFrame, component_of: dict[str, str], threshold: float = 3.0
) -> dict[str, list[str]]:
    """For each component, the other components with a condition closer than `threshold`.

    Not transitive: it answers "a case of this component could be mistaken for which
    others?". A component with an empty list can be located without ambiguity.
    """
    conditions = list(d.index)
    partners: dict[str, set[str]] = {component_of[c]: set() for c in conditions}
    for i, j in _close_pairs(d, threshold):
        a, b = component_of[conditions[i]], component_of[conditions[j]]
        if a != b:
            partners[a].add(b)
            partners[b].add(a)
    return {name: sorted(others) for name, others in sorted(partners.items())}


def component_groups(
    d: pd.DataFrame, component_of: dict[str, str], threshold: float = 3.0
) -> list[list[str]]:
    """Transitive closure of `confusable_components`: groups of mutually linked components."""
    conditions = list(d.index)
    components = sorted({component_of[c] for c in conditions})
    index = {name: k for k, name in enumerate(components)}
    pairs = [
        (index[component_of[conditions[i]]], index[component_of[conditions[j]]])
        for i, j in _close_pairs(d, threshold)
    ]
    return connected_groups(components, pairs)
