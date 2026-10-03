"""Every fault of the ECG catalogue, injected with the primitives, gives the baseline netlist."""

import pytest
from ecgfd.circuit import build_netlist, nominal_instance
from ecgfd.dataset import sample_rng
from ecgfd.faults import HEALTHY, fault_catalogue
from ecgfd.sampling import sample_instance

from ecg_adapter import inject
from spicefault.netlist import Netlist

pytestmark = pytest.mark.ecgfd

CONTROL = ["op", "write out.raw v(out)"]


def instances(cfg):
    """The nominal circuit and three Monte Carlo draws."""
    return [nominal_instance(cfg)] + [sample_instance(cfg, sample_rng(cfg, 0, r)) for r in range(3)]


def test_netlists_are_identical_for_the_whole_catalogue(cfg):
    catalogue = [HEALTHY, *fault_catalogue(cfg)]
    assert len(catalogue) == {"integrated": 294, "reference": 308}[cfg["circuit"]]
    for healthy in instances(cfg):
        healthy_text = build_netlist(healthy, cfg, CONTROL)
        for fault in catalogue:
            baseline = build_netlist(fault.apply(healthy, cfg), cfg, CONTROL)
            ours = str(inject(Netlist(healthy_text), fault, healthy, cfg))
            assert ours == baseline, fault.id
            assert (ours == healthy_text) == (fault.kind == "healthy"), fault.id


def test_healthy_netlist_survives_parsing(cfg):
    text = build_netlist(nominal_instance(cfg), cfg, CONTROL)
    net = Netlist(text)
    assert str(net) == text
    # the subcircuit definitions also contain R1 and C1; only the top-level ones count
    names = net.components()
    assert len(names) == len(set(names))
    assert net.value("R1") == 10e3
