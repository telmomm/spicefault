"""Sampling designs that choose the points of a population jointly.

By default every sample of an experiment draws from its own independent stream. A
Latin hypercube or a Sobol sequence instead decides the points of a whole population
together, so that they cover the space more evenly and a mean is estimated with fewer
simulations. The design is a function of the seed, the population and its size only,
and a sample reads its row: the result still does not depend on the number of workers
or on the order of execution.

The points of such a design are not independent. Intervals that assume independence
(binomial, order statistics) are not valid for them; the uncertainty of an estimate
comes from independent replications of the whole design, with different seeds.
"""

from __future__ import annotations

import warnings

import numpy as np

from .seeding import sample_stream, text_key

METHODS = ("random", "lhs", "sobol")
_TINY = 1e-12


def latin_hypercube(rng: np.random.Generator, n: int, d: int) -> np.ndarray:
    """[n, d] points in (0, 1): each column has exactly one point in each of the n equal
    strata, at a random place inside it, and the strata are paired at random between
    columns.
    """
    points = np.empty((n, d))
    for column in range(d):
        points[:, column] = (rng.permutation(n) + rng.random(n)) / n
    return np.clip(points, _TINY, 1.0 - _TINY)


def sobol(rng: np.random.Generator, n: int, d: int) -> np.ndarray:
    """[n, d] points of a scrambled Sobol sequence, from SciPy.

    The sequence is balanced when n is a power of two. Its points depend on the version
    of SciPy, which is recorded with the experiment.
    """
    try:
        from scipy.stats import qmc
    except ImportError as exc:
        raise ImportError(
            'Sobol sampling uses SciPy, an optional dependency: pip install "spicefault[sampling]"'
        ) from exc
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # n that is not a power of two
        points = qmc.Sobol(d, scramble=True, seed=rng).random(n)
    return np.clip(points, _TINY, 1.0 - _TINY)


def uniforms(method: str, seed: int, key: tuple[int, ...], n: int, d: int) -> np.ndarray:
    """The design of the population identified by `key`: [n, d] numbers in (0, 1)."""
    rng = sample_stream(seed, *key, text_key(method))
    if method == "lhs":
        return latin_hypercube(rng, n, d)
    if method == "sobol":
        return sobol(rng, n, d)
    raise ValueError(f"unknown sampling method {method!r}; expected one of {METHODS}")


def record(method: str) -> dict:
    """What a dataset keeps about how its samples were chosen."""
    found = {"method": method}
    if method == "sobol":
        import scipy

        found["scipy"] = scipy.__version__
    return found
