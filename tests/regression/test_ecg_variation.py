"""The healthy population of the ECG study, drawn by a `VariationSet`."""

import numpy as np
import pytest
from ecgfd.circuit import build_netlist, nominal_instance
from ecgfd.dataset import build_tasks, sample_rng
from ecgfd.sampling import instance_parameters, sample_instance

from ecg_adapter import to_fault, variations
from spicefault import Circuit, Experiment
from spicefault.experiments import sample_stream

pytestmark = pytest.mark.ecgfd

CONTROL = ["op", "write out.raw v(out)"]
KEYS = [(0, r) for r in range(40)] + [(i, r) for i in (1, 57, 200, 293) for r in range(5)]


@pytest.fixture(scope="module")
def circuit(cfg):
    return Circuit(build_netlist(nominal_instance(cfg), cfg, CONTROL), cfg["circuit"])


def test_drawn_circuits_are_identical(cfg, circuit):
    population = variations(cfg)
    kinds = set()
    for key in KEYS:
        baseline = sample_instance(cfg, sample_rng(cfg, *key))
        netlist = circuit.netlist()
        draw = population.sample(sample_stream(cfg["seed"], *key), netlist)
        population.apply(netlist, draw.values)
        assert str(netlist) == build_netlist(baseline, cfg, CONTROL), key
        assert draw.labels["electrode_type"] == baseline.electrode_type
        assert draw.labels["electrode_kind"] == baseline.electrode_kind
        kinds.add(baseline.electrode_kind)
        # what the dataset records about the INA that is not a netlist parameter
        recorded = instance_parameters(baseline)
        for name, value in draw.labels.items():
            if name.startswith("p_"):
                assert value == recorded[name], (key, name)
    assert len(kinds) >= 5  # gel and several dry types were drawn


def test_every_netlist_parameter_of_the_population_is_declared(cfg, circuit):
    population = variations(cfg)
    targets = {t for v in population for t in v.targets()}
    nominal = circuit.parameters()
    assert targets <= set(nominal)
    # one sample changes every declared parameter and nothing else
    draw = population.sample(sample_stream(cfg["seed"], 0, 0), circuit.netlist())
    assert set(draw.values) == targets
    changed = {t for t in targets if draw.values[t] != nominal[t]}
    assert len(changed) >= len(targets) - 1


def test_experiment_reproduces_the_first_samples_of_the_baseline(cfg, circuit):
    """Experiment, with positional seeding, builds the netlists of the ECG dataset."""
    tasks = build_tasks({**cfg, "dataset": {"n_healthy": 2, "n_per_fault": 2}})
    nominal = nominal_instance(cfg)
    faults = [to_fault(f, nominal, cfg) for i, f, r in tasks if i and r == 0]
    experiment = Experiment(
        circuit, faults=faults, variations=variations(cfg), samples=2, seed=cfg["seed"]
    )
    plan = experiment.plan()
    assert len(plan) == len(tasks)
    checked = 0
    for sample, (index, fault, replica) in zip(plan, tasks, strict=True):
        assert (sample.fault_index, sample.replica) == (index, replica)
        if fault.kind == "ina_cmrr" or sample.sample_id % 5:
            continue  # the injected value depends on the drawn sign: built per sample
        baseline = fault.apply(sample_instance(cfg, sample_rng(cfg, index, replica)), cfg)
        assert experiment.realise(sample).netlist == build_netlist(baseline, cfg, CONTROL), fault.id
        checked += 1
    assert checked > 100


def test_scaled_population_shrinks_the_passives_only(cfg, circuit):
    """Tolerance scale of experiment E: the electrode and INA draws cannot be scaled."""
    population = variations(cfg)
    with pytest.raises(NotImplementedError):
        population.scaled(0.0)
    exact = population.scaled(0.0, strict=False)
    full = population.sample(sample_stream(1), circuit.netlist())
    none = exact.sample(sample_stream(1), circuit.netlist())
    nominal = circuit.parameters()
    joint = {t for v in population if v.metadata()["type"] == "joint" for t in v.targets()}
    for target, value in none.values.items():
        if target in joint:
            assert value == full.values[target]  # same random numbers, unchanged
        else:
            assert value == pytest.approx(nominal[target], abs=1e-12 * abs(nominal[target]))
    assert np.isfinite(list(none.values.values())).all()
