"""Random streams and tolerance draws are the ones of the ECG baseline."""

import numpy as np
import pytest
from ecgfd.dataset import sample_rng
from ecgfd.sampling import _unit, sample_instance

from spicefault.experiments import sample_stream
from spicefault.variation import log_uniform_factor, toleranced, unit_deviation

pytestmark = pytest.mark.ecgfd


@pytest.mark.parametrize("key", [(0, 0), (0, 4999), (17, 3), (293, 199)])
def test_streams_are_identical(cfg, key):
    ours = sample_stream(cfg["seed"], *key)
    theirs = sample_rng(cfg, *key)
    assert np.array_equal(ours.uniform(size=50), theirs.uniform(size=50))
    assert np.array_equal(ours.normal(size=50), theirs.normal(size=50))


@pytest.mark.parametrize("distribution", ["uniform", "truncnorm"])
def test_unit_deviations_are_identical(distribution):
    ours, theirs = sample_stream(1, 2), sample_stream(1, 2)
    for _ in range(500):
        assert unit_deviation(ours, distribution) == _unit(theirs, distribution)


def test_passives_of_a_drawn_circuit_are_reproduced(cfg):
    """The first draws of `sample_instance` are the passives, in circuit order."""
    from ecgfd.circuit import get_circuit
    from ecgfd.sampling import passive_tolerance

    inst = sample_instance(cfg, sample_rng(cfg, 5, 11))
    rng = sample_stream(cfg["seed"], 5, 11)
    for p in get_circuit(cfg).passives:
        value = toleranced(
            p.value, passive_tolerance(p.name, cfg), rng, cfg["tolerances"]["distribution"]
        )
        assert value == inst.passives[p.name]


def test_log_uniform_factor_is_the_electrode_draw():
    spread = 2.0
    ours, theirs = sample_stream(3), sample_stream(3)
    log_spread = np.log(float(spread))
    for _ in range(100):
        assert log_uniform_factor(ours, spread) == float(
            np.exp(theirs.uniform(-log_spread, log_spread))
        )
