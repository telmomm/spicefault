from pathlib import Path

import pytest

from spicefault import Circuit
from spicefault.testdesign import candidates

DATA = Path(__file__).resolve().parents[2] / "data"
NETLIST = DATA / "sallen_key_bandpass.cir"


@pytest.fixture(scope="module")
def sallen_key():
    """The Sallen-Key band-pass with a pulse on its source, so that it can take steps."""
    return candidates.prepare_source(Circuit.from_netlist(NETLIST, name="sk"), "V1")


@pytest.fixture(scope="module")
def rc_divider():
    """An RC divider whose sensitivities and impedance have a closed form."""
    return Circuit.from_netlist(DATA / "rc_divider.cir", name="rc")
