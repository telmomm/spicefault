import pytest

from spicefault.netlist import Netlist, apply_rule, parse_value

DECK = """\
* title line, never an element
.subckt opamp inp inn out aol=2e5
R1 inp inn 1e12
C1 n1 0 1n
.ends opamp

Vcc vcc 0 dc 5
R1 in out 10k
C1 out 0 1e-07
XU1 out fb o1 opamp aol=200000.0 vos=0.0
+ gbw=1meg

.control
op
R2 is not an element here
.endc
.end
"""


@pytest.mark.parametrize(
    "token, value",
    [("10k", 1e4), ("1meg", 1e6), ("4.7uF", 4.7e-6), ("1e-09", 1e-9), ("-2.5m", -2.5e-3),
     ("100", 100.0), (".5", 0.5), ("1MEG", 1e6), ("3mil", 76.2e-6)],
)
def test_parse_value(token, value):
    assert parse_value(token) == pytest.approx(value, rel=1e-15)


def test_parse_value_rejects_expressions():
    with pytest.raises(ValueError):
        parse_value("{aol*1k}")


def test_rules():
    assert apply_rule(10.0, "absolute", 3.0) == 3.0
    assert apply_rule(10.0, "relative", 0.2) == 10.0 * (1.0 + 0.2)
    assert apply_rule(10.0, "scale", 0.5) == 5.0
    assert apply_rule(10.0, "divide", 4.0) == 2.5
    with pytest.raises(ValueError):
        apply_rule(10.0, "double", 1.0)


def test_only_top_level_elements_are_addressed():
    net = Netlist(DECK)
    assert net.components() == ["Vcc", "R1", "C1", "XU1"]
    assert net.nodes("R1") == ["in", "out"]  # not the R1 inside the subcircuit
    assert net.nodes("xu1") == ["out", "fb", "o1"]
    assert net.value("R1") == 1e4
    assert net.value("XU1", "gbw") == 1e6  # on a continuation line
    with pytest.raises(KeyError):
        net.nodes("R2")
    with pytest.raises(KeyError):
        net.value("XU1", "rout")


def test_untouched_netlist_is_unchanged():
    assert str(Netlist(DECK)) == DECK


def test_set_parameter_changes_only_its_token():
    net = Netlist(DECK)
    assert net.set_parameter("R1", "value", "relative", 0.2) == 1e4 * (1.0 + 0.2)
    assert net.set_parameter("XU1", "vos", "absolute", 0.02) == 0.02
    assert net.set_parameter("XU1", "gbw", "scale", 0.5) == 5e5
    expected = (
        DECK.replace("R1 in out 10k", "R1 in out 12000.0")
        .replace("vos=0.0", "vos=0.02")
        .replace("gbw=1meg", "gbw=500000.0")
    )
    assert str(net) == expected


def test_insert_series_opens_one_terminal():
    net = Netlist(DECK)
    assert net.insert_series("C1", 2, 1e9) == "Rser_C1"
    assert str(net) == DECK.replace(
        "C1 out 0 1e-07", "Rser_C1 C1_x 0 1000000000.0\nC1 out C1_x 1e-07"
    )
    with pytest.raises(ValueError):
        net.insert_series("C1", 1, 1e9)


def test_insert_series_on_a_subcircuit_terminal():
    net = Netlist(DECK)
    net.insert_series("XU1", 3, 1e9)
    assert net.nodes("XU1") == ["out", "fb", "XU1_x"]
    assert "Rser_XU1 XU1_x o1 1000000000.0\nXU1 out fb XU1_x opamp" in str(net)


def test_insert_parallel_goes_after_the_whole_element():
    net = Netlist(DECK)
    assert net.insert_parallel("R1", 1, 2, 1.0) == "Rpar_R1"
    net.insert_parallel("XU1", 1, 3, 1.0)
    text = str(net)
    assert "R1 in out 10k\nRpar_R1 in out 1.0\n" in text
    assert "+ gbw=1meg\nRpar_XU1 out o1 1.0\n" in text
    assert net.components() == ["Vcc", "R1", "Rpar_R1", "C1", "XU1", "Rpar_XU1"]


def test_unsupported_element_type():
    net = Netlist("t\nQ1 c b e npn\n.end\n")
    with pytest.raises(NotImplementedError):
        net.nodes("Q1")
