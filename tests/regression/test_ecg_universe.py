"""The fault catalogue of the ECG study, generated as a fault universe from rules."""

import pytest
from ecgfd.circuit import build_netlist, get_circuit, nominal_instance
from ecgfd.faults import fault_catalogue

from ecg_adapter import fault_rules, variations
from spicefault import Circuit
from spicefault.faults import FaultUniverse

pytestmark = pytest.mark.ecgfd

CONTROL = ["op", "write out.raw v(out)"]


@pytest.fixture(scope="module")
def study(cfg):
    nominal = nominal_instance(cfg)
    circuit = Circuit(build_netlist(nominal, cfg, CONTROL), cfg["circuit"])
    return nominal, circuit, FaultUniverse(circuit, fault_rules(cfg))


def test_universe_is_the_baseline_catalogue(cfg, study):
    nominal, circuit, universe = study
    catalogue = fault_catalogue(cfg)
    assert len(universe) == len(catalogue) == {"integrated": 293, "reference": 307}[cfg["circuit"]]
    assert set(universe.faults.ids()) == {f.id for f in catalogue}
    assert universe.coverage() == 1.0
    # in catalogue order, as positional seeding needs
    ordered = universe.faults.select(f.id for f in catalogue)
    for ours, theirs in zip(ordered, catalogue, strict=True):
        netlist = circuit.netlist()
        ours.apply(netlist)
        assert str(netlist) == build_netlist(theirs.apply(nominal, cfg), cfg, CONTROL), theirs.id


def test_coverage_matrix(cfg, study):
    _, circuit, universe = study
    matrix = universe.coverage_matrix()
    n_levels = len(cfg["faults"]["parametric"])
    assert matrix.loc["R1"].dropna().to_dict() == {"open": 1, "short": 1, "parametric": n_levels}
    assert matrix.loc["C1", "cap_degradation"] == 2 and matrix.loc["XU4"].sum() == 4
    assert matrix.loc["Rs_la"].dropna().to_dict() == {"electrode_off": 1}
    assert matrix.loc["Rd_la", "electrode_high_z"] == 4  # la alone, and la with ra
    assert int(matrix.sum().sum()) == len(universe)
    # supplies, test sources and the patient coupling are outside the fault model
    without = set(universe.metadata()["components_without_faults"])
    assert {"Vcc", "Vcal", "Ilo", "Cbody", "Cd_la"} <= without
    assert len(without) + len(matrix) == len(circuit.components())


def test_small_parametric_faults_overlap_the_tolerance_band(cfg, study):
    """Capacitors have 5 % tolerance, and the smallest parametric faults are +-5 % and +-10 %."""
    _, circuit, universe = study
    report = universe.faults.tolerance_overlap(variations(cfg), circuit)
    n_passives = len(variations(cfg))
    assert len(report) == n_passives * len(cfg["faults"]["parametric"])
    inside = report[report["inside_fraction"] > 0]
    capacitors = [p.name for p in get_circuit(cfg).passives if p.kind == "C"]
    assert sorted(inside["component"].unique()) == sorted(capacitors)
    # with 5 % tolerance: half of the +-5 % faults and 1 in 22 of the +10 % faults are
    # healthy by definition; -10 % is always outside, since 0.9 * 1.05 < 0.95
    level = inside["fault_id"].str.split(":").str[-1]
    fractions = inside.groupby(level)["inside_fraction"].agg(["min", "max"])
    assert fractions.to_dict("index") == {
        "+0.05": pytest.approx({"min": 0.5, "max": 0.5}),
        "-0.05": pytest.approx({"min": 0.5, "max": 0.5}),
        "+0.1": pytest.approx({"min": 1 / 22, "max": 1 / 22}),
    }
