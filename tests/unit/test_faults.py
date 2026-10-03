import json

import pytest

from spicefault import Fault
from spicefault.faults import (
    CompositeFault,
    FaultSeverity,
    InsertParallel,
    InsertSeries,
    LeakageFault,
    OpenCircuit,
    ParametricFault,
    SetParameter,
    ShortCircuit,
)
from spicefault.netlist import Netlist

TEXT = "t\nR1 in out 10k\nC1 out 0 1e-06\nXU1 out fb o opamp vos=0.0 aol=200000.0\n.end\n"


def injected(fault: Fault) -> str:
    net = Netlist(TEXT)
    fault.apply(net)
    return str(net)


DEGRADATION = CompositeFault(
    "cap_degradation",
    [ParametricFault("C1", factor=0.8), Fault("esr", [InsertSeries("C1", 2, 10.0)])],
    magnitude=0.2,
    unit="capacitance loss",
    severity=FaultSeverity.declared(0.2, "relative capacitance loss"),
    nominal_state="within tolerance",
    tags={"origin": "circuit"},
)

EVERY_TYPE = [
    Fault("custom", [SetParameter("R1", "value", "divide", 3.0)], severity=0.7),
    OpenCircuit("R1"),
    OpenCircuit("XU1", terminal=3, r_open=1e8, tags={"stage": "gain"}),
    ShortCircuit("C1"),
    ShortCircuit("XU1", terminals=(1, 2), r_short=0.1, fault_type="input_short"),
    LeakageFault("C1", 1e6),
    LeakageFault("C1", 1e5, r_min=1e3, r_max=1e9),
    ParametricFault("R1", deviation=-0.2),
    ParametricFault("XU1", "aol", factor=0.01),
    ParametricFault("C1", divisor=5.0),
    ParametricFault("XU1", "vos", fault_value=0.02, nominal_value=0.0, reference=0.5e-3),
    ParametricFault("R1", "resistance", fault_value=15e3, nominal_value=10e3),
    DEGRADATION,
]


# --- each type: what it does to the netlist ----------------------------------------


def test_open_circuit():
    assert injected(OpenCircuit("R1")) == TEXT.replace(
        "R1 in out 10k", "Rser_R1 R1_x out 1000000000.0\nR1 in R1_x 10k"
    )
    fault = OpenCircuit("XU1", terminal=3, r_open=1e8)
    assert "Rser_XU1 XU1_x o 100000000.0\nXU1 out fb XU1_x opamp" in injected(fault)
    assert (OpenCircuit("R1").fault_id, fault.fault_id) == ("R1:open", "XU1:open:t3")
    assert fault.model_parameters == {"r_open": 1e8} and fault.severity is None
    assert (fault.nominal_state, fault.fault_state) == ("connected", "open")


def test_short_circuit():
    assert injected(ShortCircuit("C1")) == TEXT.replace(
        "C1 out 0 1e-06", "C1 out 0 1e-06\nRpar_C1 out 0 1.0"
    )
    fault = ShortCircuit("XU1", terminals=(1, 3), r_short=0.1)
    assert "aol=200000.0\nRpar_XU1 out o 0.1\n" in injected(fault)
    assert (ShortCircuit("C1").fault_id, fault.fault_id) == ("C1:short", "XU1:short:t1-3")
    assert fault.model_parameters == {"r_short": 0.1} and fault.magnitude is None


def test_leakage_is_a_graded_short():
    fault = LeakageFault("C1", 1e6)
    assert injected(fault) == TEXT.replace(
        "C1 out 0 1e-06", "C1 out 0 1e-06\nRpar_C1 out 0 1000000.0"
    )
    assert (fault.fault_id, fault.magnitude, fault.unit) == ("C1:leakage:1e+06", 1e6, "ohm")
    assert fault.severity is None and fault.model_parameters == {}
    graded = [LeakageFault("C1", r, r_min=1e3, r_max=1e9).severity.value for r in (1e9, 1e6, 1e3)]
    assert graded == pytest.approx([0.0, 0.5, 1.0])
    with pytest.raises(ValueError, match="both r_min and r_max"):
        LeakageFault("C1", 1e6, r_min=1e3)
    with pytest.raises(ValueError, match="r_min <= resistance <= r_max"):
        LeakageFault("C1", 1e2, r_min=1e3, r_max=1e9)


@pytest.mark.parametrize(
    "kwargs, value, fault_id, severity",
    [
        ({"deviation": 0.2}, 1e4 * (1.0 + 0.2), "R1:parametric:+0.2", 0.2),
        ({"deviation": -0.5}, 5e3, "R1:parametric:-0.5", 0.5),
        ({"factor": 0.8}, 8e3, "R1:parametric:x0.8", 0.2),
        ({"divisor": 4.0}, 2500.0, "R1:parametric:/4", 0.75),
        ({"fault_value": 15e3, "nominal_value": 10e3}, 15e3, "R1:parametric:=15000", 0.5),
    ],
)
def test_parametric_fault_forms(kwargs, value, fault_id, severity):
    fault = ParametricFault("R1", **kwargs)
    assert injected(fault) == TEXT.replace("10k", repr(value))
    assert fault.fault_id == fault_id and fault.severity.value == pytest.approx(severity)
    assert fault.nominal_state == "within tolerance"


def test_parametric_fault_on_an_instance_parameter():
    gain = ParametricFault("XU1", "aol", factor=0.01)
    assert "aol=2000.0" in injected(gain) and gain.fault_id == "XU1.aol:parametric:x0.01"
    # a nominally zero parameter has no relative deviation: the reference is declared
    offset = ParametricFault("XU1", "vos", fault_value=0.02, nominal_value=0.0, reference=0.5e-3)
    assert "vos=0.02" in injected(offset) and offset.fault_id == "XU1.vos:parametric:=0.02"
    assert offset.severity == FaultSeverity(
        40.0, "abs_deviation_over_reference", {"nominal_value": 0.0, "reference": 0.5e-3}
    )
    assert ParametricFault("XU1", "vos", fault_value=0.02).severity is None
    assert ParametricFault("XU1", "vos", fault_value=0.02, nominal_value=0.0).severity is None
    declared = ParametricFault("XU1", "vos", fault_value=0.02, severity=0.9)
    assert declared.severity == FaultSeverity(0.9, "declared")


def test_parametric_fault_takes_exactly_one_form():
    with pytest.raises(ValueError, match="exactly one"):
        ParametricFault("R1")
    with pytest.raises(ValueError, match="exactly one"):
        ParametricFault("R1", deviation=0.1, factor=1.1)


def test_composite_fault_is_one_mechanism():
    assert injected(DEGRADATION) == TEXT.replace(
        "C1 out 0 1e-06", "Rser_C1 C1_x 0 10.0\nC1 out C1_x 8e-07"
    )
    assert DEGRADATION.fault_id == "C1:cap_degradation:+0.2"
    assert DEGRADATION.components == ("C1",) and len(DEGRADATION.primitives) == 2
    assert DEGRADATION.fault_state == "realised value times 0.8"
    both = CompositeFault(
        "drying", [ParametricFault("R1", factor=5.0), ParametricFault("C1", divisor=5.0)]
    )
    assert both.fault_id == "R1+C1:drying" and both.components == ("R1", "C1")
    merged = CompositeFault("broken", [OpenCircuit("R1"), ShortCircuit("C1")])
    assert merged.model_parameters == {"r_open": 1e9, "r_short": 1.0}


def test_reported_type_can_differ_from_the_class():
    fault = ParametricFault("XU1", "vos", fault_value=0.02, fault_type="opamp_vos")
    assert fault.fault_type == "opamp_vos" and fault.metadata()["class"] == "parametric"
    lead_off = OpenCircuit("R1", fault_type="electrode_off", fault_id="la:electrode_off")
    assert (lead_off.fault_type, lead_off.fault_id) == ("electrode_off", "la:electrode_off")


# --- metadata and serialisation -----------------------------------------------------


def test_metadata_of_an_open_circuit():
    assert OpenCircuit("R17").metadata() == {
        "schema_version": 1,
        "fault_id": "R17:open",
        "class": "open_circuit",
        "fault_type": "open_circuit",
        "components": ["R17"],
        "primitives": [
            {"op": "insert_series", "component": "R17", "terminal": 2, "resistance": 1e9}
        ],
        "magnitude": None,
        "severity": None,
        "model_parameters": {"r_open": 1e9},
        "nominal_state": "connected",
        "fault_state": "open",
        "tags": {},
    }


@pytest.mark.parametrize("fault", EVERY_TYPE, ids=lambda f: f.fault_id)
def test_a_fault_is_rebuilt_from_its_record(fault):
    record = json.loads(json.dumps(fault.metadata()))
    rebuilt = Fault.from_metadata(record)
    assert rebuilt == fault and type(rebuilt) is type(fault)
    assert rebuilt.metadata() == record and injected(rebuilt) == injected(fault)


def test_record_details():
    record = DEGRADATION.metadata()
    assert record["class"] == "composite" and record["fault_type"] == "cap_degradation"
    assert record["primitives"] == [
        {"op": "set_parameter", "component": "C1", "parameter": "value", "rule": "scale",
         "value": 0.8},
        {"op": "insert_series", "component": "C1", "terminal": 2, "resistance": 10.0},
    ]
    assert record["magnitude"] == {"value": 0.2, "unit": "capacitance loss"}
    assert record["severity"] == {
        "value": 0.2, "scale": "relative capacitance loss", "parameters": {},
    }
    with pytest.raises(ValueError, match="schema version"):
        Fault.from_metadata({**record, "schema_version": 2})


def test_default_identifier_of_a_plain_fault():
    assert Fault("open", [InsertSeries("R1", 2, 1e9)]).fault_id == "R1:open"
    assert Fault("short", [InsertParallel("C1", 1, 2, 1.0)], fault_id="x").fault_id == "x"
    assert Fault("drift", [SetParameter("R1", "value", "relative", -0.1)], magnitude=-0.1
                 ).fault_id == "R1:drift:-0.1"  # fmt: skip


def test_invalid_definitions_are_rejected():
    with pytest.raises(ValueError, match="at least one primitive"):
        Fault("nothing", [])
    with pytest.raises(ValueError, match="unknown rule"):
        SetParameter("R1", "value", "double", 2.0)


# --- severity -----------------------------------------------------------------------


def test_severity_scales():
    assert FaultSeverity.relative_deviation(-0.2) == FaultSeverity(0.2, "abs_relative_deviation")
    s = FaultSeverity.deviation_over_reference(15e3, 10e3, 10e3)
    assert s.value == 0.5 and s.parameters == {"nominal_value": 10e3, "reference": 10e3}
    with pytest.raises(ValueError, match="cannot be zero"):
        FaultSeverity.deviation_over_reference(0.02, 0.0, 0.0)
    log = FaultSeverity.log_resistance(1e5, 1e3, 1e9)
    assert log.value == pytest.approx(4 / 6) and log.scale == "log_resistance"
    assert FaultSeverity.from_metadata(json.loads(json.dumps(log.metadata()))) == log
    assert Fault("x", [InsertSeries("R1", 2, 1.0)], severity=1).severity == FaultSeverity(
        1.0, "declared"
    )
