"""The ECG self-test measurement of a faulty circuit, through Circuit, Fault and Simulator.

Unlike the campaign tests, the faulty netlist is built here by spicefault: the fault
is injected into the deck of the healthy circuit and simulated by the backend.
"""

import ecgfd.simulate
import numpy as np
import pytest
from ecgfd.dataset import sample_rng
from ecgfd.faults import fault_catalogue
from ecgfd.sampling import sample_instance

from ecg_adapter import to_fault
from spicefault import Circuit, SimulationStatus, Simulator

pytestmark = [pytest.mark.ecgfd, pytest.mark.ngspice]


def measure_and_capture(inst, cfg, monkeypatch):
    """Run `ecgfd.simulate.measure`; return (deck it simulated, plots it obtained)."""
    seen = {}

    def run_deck(netlist, *args, **kwargs):
        seen["deck"] = netlist
        seen["plots"] = ecgfd.spice.run_deck(netlist, *args, **kwargs)
        return seen["plots"]

    monkeypatch.setattr(ecgfd.simulate, "run_deck", run_deck)
    ecgfd.simulate.measure(inst, cfg)
    return seen["deck"], seen["plots"]


def test_faulty_measurements_are_identical(cfg, monkeypatch):
    healthy = sample_instance(cfg, sample_rng(cfg, 0, 0))
    healthy_deck, _ = measure_and_capture(healthy, cfg, monkeypatch)
    circuit = Circuit(healthy_deck, cfg["circuit"])
    simulator = Simulator("ngspice")

    first_of_each_kind = {f.kind: f for f in reversed(fault_catalogue(cfg))}
    assert len(first_of_each_kind) >= 8
    for fault in first_of_each_kind.values():
        baseline_deck, baseline = measure_and_capture(fault.apply(healthy, cfg), cfg, monkeypatch)
        netlist = circuit.netlist()
        to_fault(fault, healthy, cfg).apply(netlist)
        assert str(netlist) == baseline_deck, fault.id

        result = simulator.run(netlist)
        assert result.status is SimulationStatus.SUCCESS, (fault.id, result.message)
        assert [p.name for p in result.plots] == [p.name for p in baseline]
        for ours, theirs in zip(result.plots, baseline, strict=True):
            for name in theirs.vectors:
                assert np.array_equal(ours[name], theirs[name]), (fault.id, ours.name, name)
