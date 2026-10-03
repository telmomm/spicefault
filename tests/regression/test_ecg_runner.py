"""The same ECG decks give the same vectors through both ngspice runners."""

import ecgfd.spice
import numpy as np
import pytest
from ecgfd.circuit import Stimulus, build_netlist, nominal_instance
from ecgfd.dataset import sample_rng
from ecgfd.faults import HEALTHY, Fault
from ecgfd.sampling import sample_instance

from spicefault.simulation import RAW_NAME, ngspice_version, run_deck

pytestmark = [pytest.mark.ecgfd, pytest.mark.ngspice]

CONTROL = [
    "set appendwrite",
    "op",
    f"write {RAW_NAME} v(out) v(ina_out)",
    "alter @vcal[acmag]=1",
    "ac dec 20 0.01 1e4",
    f"write {RAW_NAME} v(out)",
    "tran 1e-3 0.5 0 2.5e-4",
    f"write {RAW_NAME} v(out)",
]
FAULTS = [HEALTHY, Fault("open", "R1"), Fault("short", "C5"), Fault("parametric", "R11", 0.2)]


def test_version_strings_agree():
    assert RAW_NAME == ecgfd.spice.RAW_NAME
    assert ngspice_version() == ecgfd.spice.ngspice_version()


@pytest.mark.parametrize("fault", FAULTS, ids=lambda f: f.id)
def test_plots_are_identical(cfg, fault):
    stim = Stimulus(cal_amplitude=1e-3)
    for inst in (nominal_instance(cfg), sample_instance(cfg, sample_rng(cfg, 1, 0))):
        deck = build_netlist(fault.apply(inst, cfg), cfg, CONTROL, stim)
        ours, theirs = run_deck(deck), ecgfd.spice.run_deck(deck)
        assert [p.name for p in ours] == [p.name for p in theirs]
        assert len(ours) == 3
        for a, b in zip(ours, theirs, strict=True):
            assert list(a.vectors) == list(b.vectors)
            for name in a.vectors:
                assert a[name].dtype == b[name].dtype
                assert np.array_equal(a[name], b[name]), (a.name, name)
