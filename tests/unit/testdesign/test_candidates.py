import numpy as np
import pytest

from spicefault import Circuit, Experiment
from spicefault.faults import FaultUniverse, open_rule
from spicefault.testdesign import candidates as cand

# an amplifier with supply rails, whose nodes are named by number as in many netlists
RAILS = Circuit(
    "rails\nVIN 20 0 dc 0 ac 1\nVCC vcc 0 dc 15\nVEE vee 0 dc -15\nR1 20 4 1k\n"
    "R2 4 10 1k\nR3 10 0 1k\nR4 vcc 13 10k\nR5 13 vee 10k\nR6 13 19 1k\nR7 19 0 1k\n.end\n",
    "rails",
)


def test_accessible_nodes_skip_ground_sources_and_rails():
    nodes = cand.accessible_nodes(RAILS)
    assert nodes == ["4", "10", "13", "19"]  # natural order, not 10 before 4
    assert "19" not in cand.accessible_nodes(RAILS, exclude=["19"])


def test_enumeration_is_node_by_stimulus_by_feature():
    stimuli = [cand.ac_tones("V1", [1e3, 1e4]), *cand.dc_levels("V1", [0.5, 1.0])]
    tests = cand.enumerate_tests(["a", "out"], stimuli)
    assert len(tests) == 2 * (4 + 1 + 1)
    ids = [t.test_id for t in tests]
    assert len(set(ids)) == len(ids)
    assert ids[0] == "ac|a|gain_db@1000"
    assert list(cand.listing(tests).columns) == ["test_id", "stimulus", "kind", "node", "feature"]
    with pytest.raises(ValueError):
        cand.enumerate_tests(["a"], [cand.ac_tones("V1", [1e3])] * 2)


def test_prepare_source_adds_a_flat_pulse_once(sallen_key):
    line = next(x for x in sallen_key.to_netlist().splitlines() if x.startswith("V1"))
    assert line.lower().count("pulse(") == 1
    again = cand.prepare_source(sallen_key, "V1")
    assert again.to_netlist() == sallen_key.to_netlist()
    with pytest.raises(KeyError):
        cand.prepare_source(sallen_key, "V9")


def test_one_condition_per_stimulus_with_its_analysis():
    stimuli = [cand.ac_tones("V1", [1e3]), cand.step("V1", 1.0, 1e-4, [1e-5])]
    tests = cand.enumerate_tests(["out"], stimuli)
    found = cand.conditions(tests)
    assert [c.name for c in found] == ["ac", "step"]
    assert found[0].config.analyses[0].startswith("ac dec")
    assert found[1].config.analyses[0].startswith("tran")
    assert found[1].settings[("V1", "pulse.v2")] == 1.0
    assert sum(len(c.measurements) for c in found) == len(tests)


@pytest.mark.ngspice
def test_matrix_joins_the_stimuli_of_each_drawn_circuit(sallen_key):
    stimuli = [
        cand.ac_tones("V1", [25e3]),
        cand.step("V1", 1.0, 2e-4, [1e-5]),
        *cand.dc_levels("V1", [1.0]),
    ]
    tests = cand.enumerate_tests(cand.accessible_nodes(sallen_key), stimuli)
    faults = [
        f
        for f in FaultUniverse(sallen_key, [open_rule("R")]).selected()
        if f.fault_id in ("R1:open", "R2:open")
    ]
    experiment = Experiment(
        sallen_key,
        faults=faults,
        conditions=cand.conditions(tests),
        samples=2,
        healthy_samples=2,
        seed=1,
    )
    matrix = cand.matrix(experiment.run().to_frame(), tests)
    assert matrix.shape == (6, 2 + len(tests))
    assert not matrix[[t.test_id for t in tests]].isna().any().any()
    # the band-pass passes no DC to its output, and R2 open lifts node a to the input
    healthy = matrix[matrix.fault_id == "healthy"].iloc[0]
    assert healthy["dc1|out|value"] == pytest.approx(0.0, abs=1e-6)
    r2 = matrix[matrix.fault_id == "R2:open"].iloc[0]
    assert r2["dc1|a|value"] == pytest.approx(1.0, abs=1e-3)
    assert not np.isclose(r2["ac|out|gain_db@25000"], healthy["ac|out|gain_db@25000"])
    assert "dc1|out|value" in cand.constant_tests(matrix, tests, atol=1e-6)
