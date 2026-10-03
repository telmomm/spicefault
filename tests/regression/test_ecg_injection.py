"""Every fault of the ECG catalogue, injected with the primitives, gives the baseline netlist."""

import json

import pytest
from ecgfd.circuit import build_netlist, nominal_instance
from ecgfd.dataset import sample_rng
from ecgfd.faults import HEALTHY, fault_catalogue
from ecgfd.sampling import sample_instance
from ecgfd.specs import _bench

import spicefault
from ecg_adapter import bench_condition, inject, to_fault
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


def test_fault_records_are_complete_and_reversible(cfg):
    healthy = nominal_instance(cfg)
    faults = [to_fault(f, healthy, cfg) for f in fault_catalogue(cfg)]
    assert len({f.fault_id for f in faults}) == len(faults)
    for fault in faults:
        record = json.loads(json.dumps(fault.metadata()))
        assert spicefault.Fault.from_metadata(record) == fault
        assert record["tags"]["origin"] in ("circuit", "electrode")
    # one cause acting on two components: both electrodes drying
    both = next(f for f in faults if f.fault_id.startswith("la+ra:"))
    assert both.components == ("Rd_la", "Cd_la", "Rd_ra", "Cd_ra")


def test_bench_is_an_operating_condition_applied_after_the_fault(cfg):
    """Specification tests: the faulty circuit moved to the test bench, as `ecgfd` builds it."""
    bench = bench_condition(cfg)
    for healthy in instances(cfg)[:2]:
        healthy_text = build_netlist(healthy, cfg, CONTROL)
        for fault in [HEALTHY, *fault_catalogue(cfg)]:
            baseline = build_netlist(*_bench(fault.apply(healthy, cfg), cfg), CONTROL)
            ours = inject(Netlist(healthy_text), fault, healthy, cfg)
            bench.apply(ours)
            assert str(ours) == baseline, fault.id
            # on the bench an electrode fault is no longer in the circuit
            healthy_bench = build_netlist(*_bench(healthy, cfg), CONTROL)
            assert (baseline == healthy_bench) == (fault.origin != "circuit"), fault.id
