import numpy as np
import pytest

from spicefault.netlist import Netlist
from spicefault.simulation import SimulationError, ngspice_version, run_deck

pytestmark = pytest.mark.ngspice

# 10 kOhm and 100 nF: divider at DC, low-pass with a corner at 159.15 Hz
DECK = """\
RC low-pass
V1 in 0 dc 1 ac 1
R1 in out 10k
R2 out 0 10k
C1 out 0 100n
.control
set appendwrite
op
write out.raw v(out)
ac dec 20 1 1e5
write out.raw v(out)
.endc
.end
"""


def test_version_is_reported():
    assert ngspice_version().startswith("ngspice-")


def test_plots_are_returned_in_order_and_match_theory():
    op, ac = run_deck(DECK)
    assert op.name.startswith("Operating Point") and ac.name.startswith("AC Analysis")
    assert op["v(out)"][0] == pytest.approx(0.5, rel=1e-9)
    f = ac["frequency"].real
    # Thevenin: 0.5 V behind 5 kOhm, loaded by 100 nF
    expected = 0.5 / (1 + 2j * np.pi * f * 5e3 * 100e-9)
    assert np.allclose(ac["v(out)"], expected, rtol=1e-6)


@pytest.mark.parametrize(
    "inject, expected",
    [
        (lambda n: n.insert_series("R2", 2, 1e9), (1e9 + 1e4) / (1e9 + 2e4)),  # R2 open
        (lambda n: n.insert_parallel("R2", 1, 2, 1.0), 1 / (1e4 * (1 + 1e-4) + 1)),  # R2 short
        (lambda n: n.set_parameter("R1", "value", "relative", 0.5), 10 / 25),
    ],
)
def test_injected_faults_give_the_analytical_operating_point(inject, expected):
    net = Netlist(DECK)
    inject(net)
    assert run_deck(str(net))[0]["v(out)"][0] == pytest.approx(expected, rel=1e-6)


def test_deck_without_output_raises():
    with pytest.raises(SimulationError, match="no output"):
        run_deck("empty\nV1 in 0 dc 1\nR1 in 0 1k\n.control\nop\n.endc\n.end\n")


def test_timeout_raises():
    slow = DECK.replace("ac dec 20 1 1e5", "tran 10n 1 0 10n")
    with pytest.raises(SimulationError, match="timed out"):
        run_deck(slow, timeout=0.5)
