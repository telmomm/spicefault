import zlib

import numpy as np
import pandas as pd
import pytest

from spicefault.faults import FaultUniverse, parametric_rule
from spicefault.reliability import LimitTest
from spicefault.testdesign import candidates as cand
from spicefault.testdesign import metrics as mt
from spicefault.testdesign.metrics import adaptive as ad
from spicefault.variation import tolerances


def bernoulli(p: dict, seed: int = 0):
    """A sampler whose fault f raises an alarm with probability p[f]; replica r is always
    the same draw."""

    def sampler(fault_id, start, stop):
        rng = np.random.default_rng([seed, zlib.crc32(fault_id.encode())])
        return rng.random(stop)[start:stop] < p[fault_id]

    return sampler


def test_schedule():
    assert ad.schedule(25, 1600) == [25, 50, 100, 200, 400, 800, 1600]
    assert ad.schedule(30, 100) == [30, 60, 100]


def test_certifies_with_less_than_the_fixed_design():
    p = {f"f{i}": 1.0 for i in range(10)}
    c = ad.certify(bernoulli(p), mt.fault_table(list(p)), target=0.95)
    assert c.decision == "certified"
    assert c.lower >= 0.95
    assert c.simulated < 10 * 1600
    # with K = 10 faults and L = 7 stages, all detected, the bound is (0.05 / 70) ** (1 / n)
    n = c.per_fault.n.iloc[0]
    assert c.per_fault.lower.iloc[0] == pytest.approx((0.05 / 70) ** (1 / n), rel=1e-6)


def test_stops_early_when_the_target_is_out_of_reach():
    p = {f"f{i}": 0.5 for i in range(5)}
    c = ad.certify(bernoulli(p), mt.fault_table(list(p)), target=0.9)
    assert c.decision == "not reachable"
    assert c.per_fault.n.max() <= 100


def test_grows_only_the_faults_that_cost_most():
    p = {"weak": 0.97, **{f"f{i}": 1.0 for i in range(8)}}
    c = ad.certify(bernoulli(p, seed=3), mt.fault_table(list(p)), target=0.95)
    assert c.decision == "certified"
    assert c.per_fault.n["weak"] >= c.per_fault.n.drop("weak").min()


def test_wrong_claims_stay_below_alpha():
    # the true coverage, 0.92, is below the target, 0.93: certifying would be wrong
    wrong = 0
    for seed in range(200):
        p = {f"f{i}": 0.92 for i in range(3)}
        c = ad.certify(
            bernoulli(p, seed), mt.fault_table(list(p)), target=0.93, sizes=ad.schedule(50, 800)
        )
        wrong += c.decision == "certified"
    assert wrong / 200 <= 0.05


def test_estimate_reaches_the_width():
    p = {"a": 0.5, "b": 0.99}
    e = ad.estimate(bernoulli(p), mt.fault_table(list(p)), width=0.1)
    assert e.decision == "estimated"
    assert ((e.per_fault.upper - e.per_fault.lower) <= 0.1).all()
    assert e.per_fault.n["b"] < e.per_fault.n["a"]


def test_false_alarm_rule():
    met = ad.certify_false_alarm(bernoulli({"healthy": 0.0}), target=0.05)
    assert met["decision"] == "met" and met["upper"] <= 0.05
    exceeded = ad.certify_false_alarm(bernoulli({"healthy": 0.3}), target=0.05)
    assert exceeded["decision"] == "exceeded"


@pytest.mark.ngspice
@pytest.mark.parametrize("workers", [1, 2])
def test_spice_sampler_replicas_do_not_depend_on_how_they_are_asked_for(sallen_key, workers):
    tests = cand.enumerate_tests(["out"], [cand.ac_tones("V1", [25e3], phase=False)])
    universe = FaultUniverse(sallen_key, [parametric_rule((-0.5, 0.5), "R")])
    variations = tolerances(sallen_key, {"R": 0.05, "C": 0.10}, components=("R2", "R3", "C1", "C2"))
    fault = "R2:parametric:+0.5"
    with ad.SpiceSampler(
        sallen_key,
        universe.selected(),
        tests,
        LimitTest(0.05),
        variations=variations,
        n_calibration=20,
        n_max=40,
        seed=1,
        workers=workers,
    ) as s:
        late = s(fault, 5, 12)  # simulated first, out of order
        parts = np.concatenate([s(fault, 0, 5), late])
        whole = s(fault, 0, 12)
        assert (whole == parts).all()
        assert whole.all()  # +50 % on R2 always leaves the band at the centre frequency
        assert len(s(mt.HEALTHY, 0, 10)) == 10
        assert s.simulated == 20 + 12 + 10  # calibration, then each replica once
        assert s.detector.low is not None


def test_sampled_certification_is_exact_and_needs_few_circuits():
    p = {f"f{i}": 1.0 for i in range(50)}
    faults = mt.fault_table(list(p))
    sampled = ad.certify_sampled(bernoulli(p), faults, target=0.99, sizes=ad.schedule(50, 6400))
    per_fault = ad.certify(bernoulli(p), faults, target=0.99)
    assert sampled.decision == "certified" and sampled.lower >= 0.99
    assert sampled.simulated < per_fault.simulated / 20


def test_sampled_skips_faults_that_cannot_lower_the_metric():
    p = {"d": 1.0, "s": 0.0}
    faults = mt.fault_table(["d", "s"], iec=pd.Series({"d": "dangerous", "s": "safe"}))
    c = ad.certify_sampled(bernoulli(p), faults, target=0.9, metric="sff")
    assert c.decision == "certified"
    assert c.per_fault.n["s"] == 0 and c.per_fault.drawn["s"] > 0


def test_sampled_wrong_claims_stay_below_alpha():
    wrong = 0
    for seed in range(300):
        p = {"a": 0.95, "b": 0.85}  # equal weights: true coverage 0.90, target 0.91
        c = ad.certify_sampled(
            bernoulli(p, seed),
            mt.fault_table(list(p)),
            target=0.91,
            sizes=ad.schedule(50, 3200),
            seed=seed,
        )
        wrong += c.decision == "certified"
    assert wrong / 300 <= 0.05
