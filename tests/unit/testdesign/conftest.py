from pathlib import Path

import pytest

from spicefault import Circuit
from spicefault.testdesign import candidates

NETLIST = Path(__file__).resolve().parents[2] / "data" / "sallen_key_bandpass.cir"


@pytest.fixture(scope="module")
def sallen_key():
    """The Sallen-Key band-pass with a pulse on its source, so that it can take steps."""
    return candidates.prepare_source(Circuit.from_netlist(NETLIST, name="sk"), "V1")
