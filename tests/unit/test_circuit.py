import hashlib

import pytest

from spicefault import Circuit, Experiment

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
        ("Vin", "ac"): 1.0,
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


def test_imported_netlist_encoding_micro_suffix_and_relative_include(tmp_path):
    model = tmp_path / "model library.lib"
    model.write_text(".model DTEST D(Is=1e-14)\n")
    path = tmp_path / "export.cir"
    source = "export\n.include \"model library.lib\"\nC1 out 0 4.7µ Rser=0.2\n.end\n"
    path.write_bytes(source.encode("latin-1"))
    circuit = Circuit.from_netlist(path)
    assert "4.7u Rser=0.2" in circuit.to_netlist()
    assert f'.include "{model.resolve()}"' in circuit.to_netlist()
    assert circuit.parameters()[("C1", "value")] == pytest.approx(4.7e-6)
    record = circuit.metadata()["includes"][0]
    assert record["path"] == str(model.resolve())
    assert record["sha256"] == hashlib.sha256(model.read_bytes()).hexdigest()
    assert circuit.check() == [
        "C1: Rser is a vendor-specific capacitor parameter; ngspice may reject it"
    ]


def test_imported_dot_analyses_are_wrapped_and_metadata_is_retained(tmp_path):
    path = tmp_path / "ltspice.net"
    path.write_text("export\nV1 in 0 1\nR1 in out 1k\nC1 out 0 4.7u\n.ac dec 10 1 10k\n.end\n")
    circuit = Circuit.from_netlist(path)
    deck = circuit.to_netlist()
    assert ".ac dec 10 1 10k" not in deck
    assert circuit.imported_analyses == ("ac dec 10 1 10k",)
    assert circuit.metadata()["imported_analyses"] == ["ac dec 10 1 10k"]
    assert Experiment(circuit).config.analyses == ("ac dec 10 1 10k",)


def test_check_reports_unaddressable_and_unsupported_items():
    circuit = Circuit(
        "export\n.subckt amp in out\nRinside in out {unknown}\n.ends amp\n"
        "K1 L1 L2 0.9\nR1 in out {unknown}\n.unsupported foo\n.end\n"
    )
    problems = circuit.check()
    assert "K1: terminal layout is unsupported" in problems
    assert "R1: no readable numeric value" in problems
    assert any("subcircuit element is not addressable: rinside" in item for item in problems)
    assert any("unsupported directive .unsupported" in item for item in problems)


def test_schematic_export_examples_are_readable():
    from pathlib import Path

    root = Path(__file__).parents[2] / "examples" / "from_schematic"
    kicad = Circuit.from_netlist(root / "kicad_rc.cir")
    ltspice = Circuit.from_netlist(root / "ltspice_rc.cir")
    assert kicad.component("C1").parameters["value"] == pytest.approx(100e-9)
    assert ltspice.component("R1").parameters["value"] == 10_000.0
    assert ltspice.component("C1").parameters["value"] == pytest.approx(4.7e-6)
    assert kicad.check() == []
    assert ltspice.check() == []


def test_check_reports_vendor_specific_rser_parameter():
    circuit = Circuit("export\nC1 out 0 100n Rser=0.2\n.end\n")
    assert circuit.check() == [
        "C1: Rser is a vendor-specific capacitor parameter; ngspice may reject it"
    ]


@pytest.mark.ngspice
def test_schematic_export_examples_simulate():
    from pathlib import Path

    root = Path(__file__).parents[2] / "examples" / "from_schematic"
    for path in sorted(root.glob("*.cir")):
        circuit = Circuit.from_netlist(path)
        result = Experiment(circuit).run().samples[0].result
        assert result.ok, result.message
        assert len(result.plots) == 1
