import numpy as np
import pytest

from spicefault.experiments import sample_stream
from spicefault.variation import log_uniform_factor, toleranced, unit_deviation


def test_stream_depends_only_on_seed_and_key():
    a = sample_stream(42, 3, 7).uniform(size=5)
    sample_stream(42, 0, 0).uniform(size=100)  # another stream, drawn in between
    assert np.array_equal(a, sample_stream(42, 3, 7).uniform(size=5))
    assert not np.array_equal(a, sample_stream(42, 7, 3).uniform(size=5))
    assert not np.array_equal(a, sample_stream(43, 3, 7).uniform(size=5))


@pytest.mark.parametrize("distribution", ["uniform", "truncnorm"])
def test_unit_deviation_stays_in_range(distribution):
    rng = sample_stream(0)
    x = np.array([unit_deviation(rng, distribution) for _ in range(5000)])
    assert np.abs(x).max() <= 1.0
    assert abs(x.mean()) < 0.03
    # uniform on [-1, 1] has standard deviation 1/sqrt(3); the normal has sigma = 1/3
    assert x.std() == pytest.approx(3**-0.5 if distribution == "uniform" else 1 / 3, rel=0.05)


def test_unknown_distribution():
    with pytest.raises(ValueError):
        unit_deviation(sample_stream(0), "triangular")


def test_toleranced_and_log_uniform_ranges():
    rng = sample_stream(1)
    values = np.array([toleranced(10e3, 0.01, rng) for _ in range(2000)])
    assert 9900.0 <= values.min() and values.max() <= 10100.0
    factors = np.array([log_uniform_factor(rng, 2.0) for _ in range(2000)])
    assert 0.5 <= factors.min() and factors.max() <= 2.0
    assert np.log(factors).mean() == pytest.approx(0.0, abs=0.05)
