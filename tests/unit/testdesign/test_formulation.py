import numpy as np
import pandas as pd
import pytest

from spicefault import OperatingCondition, SimulationConfig
from spicefault.faults import FaultUniverse, open_rule, short_rule
from spicefault.measurements import Measurement
from spicefault.testdesign import metrics as mt
from spicefault.testdesign.metrics import adaptive
from spicefault.testdesign.metrics import formulation as fm
from spicefault.variation import tolerances


def conditions():
    op = SimulationConfig(analyses=("op",), outputs=("a",))
    a = (Measurement.value("v(a)", name="a"),)
    return (
        OperatingCondition("low", settings={("V1", "dc"): 1.0}, config=op, measurements=a),
        OperatingCondition("high", settings={("V1", "dc"): 2.0}, config=op, measurements=a),
    )


def synthetic(n_healthy=400, n_fault=40, seed=0):
    """Test t moves by 10 sigma under `seen` and not under `hidden`."""
    rng = np.random.default_rng(seed)
    rows = []
    for fault, shift in (("healthy", 0), ("seen", 10), ("hidden", 0)):
        for r in range(n_healthy if fault == "healthy" else n_fault):
            rows.append({"fault_id": fault, "replica": r, "t": rng.normal() + shift})
    return pd.DataFrame(rows)


def test_prefixed_names_are_unique_per_condition():
    renamed, ids = fm.prefixed(conditions())
    assert ids == ["low:a", "high:a"]
    assert [m.name for c in renamed for m in c.measurements] == ids


def test_parts_split_the_faulty_replicas_and_keep_the_fault_free():
    m = synthetic()
    sel, ev = fm.part(m, "selection"), fm.part(m, "evaluation")
    assert (sel.fault_id == "healthy").sum() == (ev.fault_id == "healthy").sum() == 400
    sel_faulty = set(sel.index[sel.fault_id != "healthy"])
    ev_faulty = set(ev.index[ev.fault_id != "healthy"])
    assert sel_faulty.isdisjoint(ev_faulty) and len(sel_faulty) == len(ev_faulty) == 40
    assert sel[sel.fault_id == "seen"].replica.max() < ev[ev.fault_id == "seen"].replica.min()
    with pytest.raises(ValueError):
        fm.part(m, "all")


def test_severity_weights_the_coverage():
    m = synthetic()
    faults = mt.fault_table(
        ["seen", "hidden"], rate=1.0, severity=pd.Series({"seen": 3.0, "hidden": 0.0}),
        iec=pd.Series({"seen": "dangerous", "hidden": "safe"}),
    )  # fmt: skip
    s = fm.score(fm.part(m, "evaluation"), ["t"], faults, localise=True)
    assert s["dc_w"] == pytest.approx(1.0)  # the undetected fault weighs nothing
    assert s["sff"] == pytest.approx(1.0)  # and it is safe
    unweighted = fm.score(fm.part(m, "evaluation"), ["t"], mt.fault_table(["seen", "hidden"]))
    # the hidden fault raises only false alarms (20 circuits, so a few of them)
    assert unweighted["dc"] == pytest.approx(0.5, abs=0.1)
    assert s.lower("dc_w") < 1.0 and s.h_prior is not None
    quick = fm.score(fm.part(m, "evaluation"), ["t"], faults, bounds=False)
    assert quick["dc_w"] == s["dc_w"] and "lower" not in quick.metrics


@pytest.mark.ngspice
def test_sampler_reads_replicas_once_and_certifies(rc_divider):
    universe = FaultUniverse(rc_divider, [open_rule("R"), short_rule("R")])
    conds, ids = fm.prefixed(conditions())
    sampler = fm.ConditionSampler(
        rc_divider, universe.selected(), conds, ids, mt.GaussianLimits(0.01),
        variations=tolerances(rc_divider, {"R": 0.01}), n_calibration=100, n_max=200, seed=3,
    )  # fmt: skip
    assert sampler.simulated == 100
    first = sampler("R1:short", 0, 20)
    assert first.all()  # a shorted divider resistor moves v(a) far outside the limits
    np.testing.assert_array_equal(sampler("R1:short", 0, 20), first)
    assert sampler.simulated == 120
    assert sampler("healthy", 0, 50).mean() < 0.2
    faults = mt.fault_table([f.fault_id for f in universe.selected()])
    c = adaptive.certify_sampled(
        sampler, faults, target=0.9, metric="dc", sizes=[50, 100, 200], seed=1
    )
    assert c.decision == "certified"
    # every replica read by the rule was simulated once, besides the calibration ones
    assert sampler.simulated == 100 + len(sampler.cache)


@pytest.mark.ngspice
def test_sampler_reads_through_a_transform(rc_divider):
    universe = FaultUniverse(rc_divider, [short_rule("R")])
    conds, ids = fm.prefixed(conditions()[:1])
    seen = []

    def transform(m):
        seen.append(len(m))
        # an instrument whose channel reads the nominal 0.5 V whenever R1 is shorted
        dead = m.fault_id == "R1:short"
        return m.assign(**{ids[0]: m[ids[0]].where(~dead, 0.5)})

    with fm.ConditionSampler(
        rc_divider, universe.selected(), conds, ids, mt.GaussianLimits(0.01),
        variations=tolerances(rc_divider, {"R": 0.01}), n_calibration=40, n_max=20,
        transform=transform,
    ) as sampler:  # fmt: skip
        assert sampler("R2:short", 0, 10).all()
        assert not sampler("R1:short", 0, 10).any()
    assert seen == [40, 10, 10]
