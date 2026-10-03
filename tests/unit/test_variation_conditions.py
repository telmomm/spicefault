import numpy as np
import pytest

from spicefault import OperatingCondition, VariationSet
from spicefault.experiments import sample_stream
from spicefault.netlist import Netlist
from spicefault.variation import ToleranceVariation

TEXT = (
    "t\nVcc vcc 0 dc 5\nR1 vcc out 10k\nC1 out 0 1e-06\n"
    "XU1 out o opamp vos=0.0 aol=200000.0\n.end\n"
)
VARIATIONS = VariationSet(
    [
        ToleranceVariation("R1", 0.01),
        ToleranceVariation("C1", 0.05, "truncnorm"),
        ToleranceVariation("XU1", 0.5, parameter="aol"),
    ]
)


def test_a_stream_defines_the_drawn_circuit():
    a = VARIATIONS.sample(sample_stream(7, 1, 2), Netlist(TEXT))
    assert a == VARIATIONS.sample(sample_stream(7, 1, 2), Netlist(TEXT))
    assert a != VARIATIONS.sample(sample_stream(7, 1, 3), Netlist(TEXT))
    assert list(a) == [("R1", "value"), ("C1", "value"), ("XU1", "aol")]


def test_drawn_values_stay_within_tolerance_and_are_written():
    net = Netlist(TEXT)
    draws = [VARIATIONS.sample(sample_stream(0, r), net) for r in range(500)]
    for key, nominal, tol in ((("R1", "value"), 1e4, 0.01), (("XU1", "aol"), 2e5, 0.5)):
        values = np.array([d[key] for d in draws])
        assert abs(values / nominal - 1).max() <= tol
        assert abs(values / nominal - 1).max() > 0.9 * tol
    VARIATIONS.apply(net, draws[0])
    assert net.value("R1") == draws[0][("R1", "value")]
    assert net.value("XU1", "aol") == draws[0][("XU1", "aol")]
    assert net.value("XU1", "vos") == 0.0


def test_variation_definitions_are_checked():
    with pytest.raises(ValueError, match="more than one variation"):
        VariationSet([ToleranceVariation("R1", 0.01), ToleranceVariation("r1", 0.05)])
    with pytest.raises(ValueError, match="unknown tolerance distribution"):
        ToleranceVariation("R1", 0.01, "triangular")
    assert len(VariationSet()) == 0 and VariationSet().sample(sample_stream(0), Netlist(TEXT)) == {}
    assert VARIATIONS.metadata()[1] == {
        "type": "tolerance", "component": "C1", "parameter": "value", "tolerance": 0.05,
        "distribution": "truncnorm",
    }


def test_operating_condition_sets_values_and_temperature():
    net = Netlist(TEXT)
    OperatingCondition().apply(net)
    assert str(net) == TEXT  # the nominal condition changes nothing
    low = OperatingCondition("low supply, hot", 85, {("Vcc", "dc"): 4.5, ("R1", "value"): 2e4})
    low.apply(net)
    assert str(net) == TEXT.replace("dc 5", "dc 4.5").replace("10k", "20000.0").replace(
        ".end", ".options temp=85.0\n.end"
    )
    assert low.metadata() == {
        "name": "low supply, hot",
        "temperature": 85,
        "settings": [
            {"component": "Vcc", "parameter": "dc", "value": 4.5},
            {"component": "R1", "parameter": "value", "value": 2e4},
        ],
    }
