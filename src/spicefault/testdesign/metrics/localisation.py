"""Localisation: which faults a set of tests tells apart.

The signature of a circuit is the side of its limits on which each test falls. The
residual uncertainty H(F | Y) says how much is still unknown about the fault once the
signature is read; the ambiguity sets group the faults that share a signature.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .detection import HEALTHY, Detection, values


def signatures(matrix: pd.DataFrame, detection: Detection) -> pd.Series:
    """Per row, which side of its limits each test falls on: a string of -, 0 and +."""
    x = values(matrix, detection.tests, detection.angles)
    low, high = detection.detector.low, detection.detector.high
    side = np.where(x < low, "-", np.where(x > high, "+", "0"))
    return pd.Series(["".join(row) for row in side], index=matrix.index)


def localisation(
    matrix: pd.DataFrame,
    detection: Detection,
    prior: pd.Series | None = None,
    smoothing: float = 0.5,
) -> dict:
    """Residual uncertainty about which fault is present, from the signatures of the tests.

    H(F | Y) in bits over the faulty circuits, with prior `prior` (by default uniform),
    estimated as a cross-validated cross-entropy: P(f | y) is fitted on the even
    replicas and scored on the odd ones and the other way round, with `smoothing` added
    to every count. It is an upper bound: a better model of P(f | y) can only lower it.
    Fitting and scoring on the same circuits would give H close to 0 whenever
    signatures are rare. A signature not seen in fitting takes the posterior of the
    nearest seen ones (fewest tests on a different side), averaged.

    Returns `h_prior`, H(F), the uncertainty with no test; `h_given_tests`; and the
    number of distinct `signatures`.
    """
    faulty = matrix[matrix.fault_id != HEALTHY]
    ids = list(dict.fromkeys(faulty.fault_id))
    prior = pd.Series(1.0, index=ids) if prior is None else prior.reindex(ids).astype(float)
    prior = prior / prior.sum()
    sig = signatures(faulty, detection)
    fold = (faulty.replica % 2).to_numpy()
    n_f = faulty.fault_id.value_counts()
    loss = 0.0
    for held in (0, 1):
        fit, score = fold != held, fold == held
        counts = pd.crosstab(sig[fit], faulty.fault_id[fit]).reindex(columns=ids, fill_value=0)
        likelihood = (counts + smoothing) / (counts.sum() + smoothing * (len(counts) + 1))
        posterior = likelihood * prior
        posterior = posterior.div(posterior.sum(axis=1), axis=0)
        seen = _encode(posterior.index)
        table = posterior.to_numpy()
        column = {f: j for j, f in enumerate(posterior.columns)}
        rows = {s: i for i, s in enumerate(posterior.index)}
        cache = {}
        for s, f in zip(sig[score], faulty.fault_id[score], strict=True):
            if s not in cache:
                if s in rows:
                    cache[s] = table[rows[s]]
                else:
                    distance = (seen != _encode([s])[0]).sum(axis=1)
                    cache[s] = table[distance == distance.min()].mean(axis=0)
            loss += prior[f] / n_f[f] * -math.log2(cache[s][column[f]])
    return {
        "h_prior": float(-(prior * np.log2(prior)).sum()),
        "h_given_tests": float(loss),
        "signatures": int(sig.nunique()),
    }


def _encode(signatures) -> np.ndarray:
    """Signatures as an array of bytes [signatures, tests], for counting differences."""
    signatures = list(signatures)
    encoded = np.frombuffer("".join(signatures).encode("ascii"), dtype=np.uint8)
    return encoded.reshape(len(signatures), -1)


def ambiguity_sets(matrix: pd.DataFrame, detection: Detection) -> list[list[str]]:
    """Faults grouped by their most frequent signature. Faults whose most frequent
    signature is all zeros, the fault-free one, come first: those the tests mostly miss.
    """
    faulty = matrix[matrix.fault_id != HEALTHY]
    sig = signatures(faulty, detection)
    modal = sig.groupby(faulty.fault_id, sort=False).agg(lambda s: s.value_counts().index[0])
    zero = "0" * len(detection.tests)
    groups = modal.groupby(modal, sort=False).groups
    sets = [
        sorted(map(str, members))
        for s, members in groups.items()
        if len(members) > 1 or s == zero
    ]
    return sorted(sets, key=lambda g: (modal[g[0]] != zero, -len(g), g))
