import json

import pytest

from spicefault import Fault
from spicefault.faults import InsertParallel, InsertSeries, SetParameter
from spicefault.netlist import Netlist

TEXT = "t\nR1 in out 10k\nC1 out 0 1e-06\n.end\n"

DEGRADATION = Fault(
    "cap_degradation",
    [SetParameter("C1", "value", "scale", 0.8), InsertSeries("C1", 2, 10.0)],
    magnitude=0.2,
    unit="capacitance loss",
    severity=0.2,
    severity_scale="relative capacitance loss",
    model_parameters={"esr": 10.0},
    nominal_state="within tolerance",
    fault_state="capacitance -20 %, ESR 10 ohm",
    tags={"origin": "circuit"},
)


def test_a_fault_is_its_list_of_primitives():
    net = Netlist(TEXT)
    DEGRADATION.apply(net)
    assert str(net) == "t\nR1 in out 10k\nRser_C1 C1_x 0 10.0\nC1 out C1_x 8e-07\n.end\n"
    assert DEGRADATION.components == ("C1",)


def test_default_identifier_comes_from_the_content():
    assert Fault("open", [InsertSeries("R1", 2, 1e9)]).fault_id == "R1:open"
    short = Fault("short", [InsertParallel("C1", 1, 2, 1.0)], fault_id="C1 shorted")
    assert short.fault_id == "C1 shorted"
    assert DEGRADATION.fault_id == "C1:cap_degradation:+0.2"
    both = Fault(
        "drift",
        [SetParameter("R1", "value", "relative", -0.1), SetParameter("C1", "value", "divide", 2)],
        magnitude=-0.1,
    )
    assert both.fault_id == "R1+C1:drift:-0.1"


def test_metadata_is_json_and_rebuilds_the_fault():
    record = json.loads(json.dumps(DEGRADATION.metadata()))
    assert record["schema_version"] == 1
    assert record["primitives"] == [
        {"op": "set_parameter", "component": "C1", "parameter": "value", "rule": "scale",
         "value": 0.8},
        {"op": "insert_series", "component": "C1", "terminal": 2, "resistance": 10.0},
    ]
    assert record["magnitude"] == {"value": 0.2, "unit": "capacitance loss"}
    assert Fault.from_metadata(record) == DEGRADATION
    hard = Fault("open", [InsertSeries("R1", 2, 1e9)])
    assert hard.metadata()["magnitude"] is None
    assert Fault.from_metadata(hard.metadata()) == hard
    with pytest.raises(ValueError, match="schema version"):
        Fault.from_metadata({**record, "schema_version": 2})


def test_invalid_definitions_are_rejected():
    with pytest.raises(ValueError, match="at least one primitive"):
        Fault("nothing", [])
    with pytest.raises(ValueError, match="unknown rule"):
        SetParameter("R1", "value", "double", 2.0)
