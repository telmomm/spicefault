import hashlib

import pytest

from spicefault import Circuit

TEXT = """\
inverting amplifier
.subckt opamp inp inn out aol=2e5
R1 inp inn 1e12
E1 out 0 inp inn {aol}
.ends opamp
Vin in 0 dc 0 ac 1
R1 in fb 1k
R2 fb out 10k
XU1 0 fb out opamp aol=100000.0
K1 L1 L2 0.9
.end
"""


def test_components_nodes_and_parameters():
    circuit = Circuit(TEXT, "inverter")
    assert [c.name for c in circuit.components()] == ["Vin", "R1", "R2", "XU1", "K1"]
    r2 = circuit.component("r2")
    assert (r2.kind, r2.nodes, r2.parameters) == ("R", ("fb", "out"), {"value": 1e4})
    assert circuit.component("XU1").nodes == ("0", "fb", "out")
    assert circuit.component("K1").nodes == ()  # terminals of this type are not known
    assert circuit.nodes() == ["0", "fb", "in", "out"]
    assert circuit.parameters() == {
        ("Vin", "dc"): 0.0,
        ("R1", "value"): 1e3,
        ("R2", "value"): 1e4,
        ("XU1", "aol"): 1e5,
    }
    with pytest.raises(KeyError):
        circuit.component("R9")


def test_the_nominal_circuit_is_never_modified():
    circuit = Circuit(TEXT)
    netlist = circuit.netlist()
    netlist.set_parameter("R1", "value", "scale", 2.0)
    netlist.insert_series("R2", 2, 1e9)
    assert circuit.to_netlist() == TEXT and str(circuit.netlist()) == TEXT


def test_from_netlist_and_identity(tmp_path):
    path = tmp_path / "inverter.cir"
    path.write_text(TEXT)
    circuit = Circuit.from_netlist(path)
    assert circuit.name == "inverter" and circuit.to_netlist() == TEXT
    assert circuit.metadata() == {
        "circuit": "inverter",
        "netlist_sha256": hashlib.sha256(TEXT.encode()).hexdigest(),
    }
