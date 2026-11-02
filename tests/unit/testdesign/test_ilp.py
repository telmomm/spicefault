import importlib.util
from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from spicefault.testdesign import candidates as cand
from spicefault.testdesign import costmodel as cm
from spicefault.testdesign import metrics as mt
from spicefault.testdesign.metrics import adaptive
from spicefault.testdesign.select import ilp

PRICE = {"external_opamp_channel": 0.12, "mux_8": 0.25, "injection_switch": 0.40}
needs_solver = pytest.mark.skipif(
    not all(importlib.util.find_spec(m) for m in ("pulp", "highspy")),
    reason="PuLP and HiGHS not installed (spicefault[ilp])",
)


def random_instance(seed, n_tests=8, n_circuits=60, channels=2, opamps=0):
    rng = np.random.default_rng(seed)
    nodes = [f"n{i}" for i in range(4)]
    node = tuple(rng.choice(nodes) for _ in range(n_tests))
    stimulus = tuple(rng.choice(["ac", "step"]) for _ in range(n_tests))
    acquisition = tuple(f"{s}|{n}" for s, n in zip(stimulus, node, strict=True))
    weight = rng.random(n_circuits)
    weight = 0.9 * weight / weight.sum()
    return ilp.Instance(
        tests=tuple(f"t{i}" for i in range(n_tests)),
        detected=rng.random((n_circuits, n_tests)) < 0.3,
        weight=weight, free=0.1, node=node,
        sources=tuple(("V1",) if s == "ac" else ("V1", "S2") for s in stimulus),
        ac=tuple(s == "ac" for s in stimulus), acquisition=acquisition,
        duration={k: float(rng.integers(1, 10)) for k in set(acquisition)},
        buffer={n: bool(rng.random() < 0.5) for n in nodes},
        channels=channels, opamps=opamps, price=PRICE,
        false_alarm=np.zeros(n_tests), held_out=rng.random((100, n_tests)) < 0.02,
    )  # fmt: skip


@needs_solver
@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("share", [0.5, 0.8, 0.95])
def test_ilp_matches_enumeration(seed, share):
    inst = random_instance(seed)
    best = inst.coverage(inst.tests)
    target = inst.free + share * (best - inst.free)
    exact, enum = ilp.solve(inst, target, max_tests=8), ilp.brute_force(inst, target, 8)
    assert exact.status == enum.status == "optimal"
    assert exact.euros == pytest.approx(enum.euros)
    assert exact.time == pytest.approx(enum.time)
    assert len(exact.tests) == len(enum.tests)
    assert exact.coverage == pytest.approx(enum.coverage)
    assert exact.coverage >= target - 1e-9


@needs_solver
@pytest.mark.parametrize("seed", range(4))
def test_false_alarm_constraint_matches_enumeration(seed):
    inst = random_instance(seed)
    target = inst.free + 0.6 * (inst.coverage(inst.tests) - inst.free)
    exact = ilp.solve(inst, target, 8, max_false_alarm=0.04)
    enum = ilp.brute_force(inst, target, 8, max_false_alarm=0.04)
    assert exact.status == enum.status
    if exact.status == "optimal":
        assert exact.false_alarm <= 0.04 + 1e-12
        found = (exact.euros, exact.time, len(exact.tests))
        assert found == pytest.approx((enum.euros, enum.time, len(enum.tests)))


@needs_solver
def test_infeasible_target_and_maximise():
    inst = random_instance(1)
    assert ilp.solve(inst, 1.01).status == "infeasible"
    top = ilp.solve(inst, maximise=True, max_tests=2)
    assert top.coverage == pytest.approx(max(inst.coverage(p) for p in combinations(inst.tests, 2)))
    with pytest.raises(ValueError):
        ilp.solve(inst)


@needs_solver
def test_budget():
    inst = random_instance(3, channels=1)
    for budget in (0.40, 0.52, 0.65):
        top = ilp.solve(inst, maximise=True, max_tests=3, max_euros=budget)
        # the empty plan costs nothing and covers the free share
        affordable = (
            inst.coverage(p) for k in (1, 2, 3) for p in combinations(inst.tests, k)
            if inst.euros(p) <= budget + 1e-9
        )  # fmt: skip
        best = max(affordable, default=inst.free)
        assert top.status == "optimal"
        assert top.coverage == pytest.approx(max(best, inst.free)) and top.euros <= budget + 1e-9


def test_cost_terms_and_subset():
    inst = random_instance(2, channels=1)
    t = inst.tests
    assert inst.euros([t[0]]) >= 0.40  # a stimulus switch at least
    assert inst.coverage([]) == pytest.approx(0.1) and inst.false_alarm_rate([]) == 0.0
    small = inst.subset([t[5], t[2]])
    assert small.tests == (t[5], t[2])
    for plan in ([t[5]], [t[2], t[5]]):
        assert small.coverage(plan) == inst.coverage(plan)
        assert small.euros(plan) == inst.euros(plan) and small.time(plan) == inst.time(plan)
        assert small.false_alarm_rate(plan) == inst.false_alarm_rate(plan)


def test_instance_matches_the_cost_model_and_the_metric():
    """An instance built from a test matrix: its euros and time are those of
    `costmodel.cost`, and its coverage that of the fault weights."""
    stimuli = [cand.ac_tones("V1", [1e3]), *cand.dc_levels("V1", [1.0])]
    tests = cand.enumerate_tests(["a", "b"], stimuli)
    rng = np.random.default_rng(0)
    rows = [{"fault_id": "healthy", "replica": r} for r in range(200)]
    rows += [{"fault_id": f, "replica": r} for f in ("f1", "f2") for r in range(10)]
    m = pd.DataFrame(rows)
    for t in tests:
        m[t.test_id] = rng.normal(size=len(m))
    # f1 moves the DC level at a far away; f2 moves nothing
    m.loc[m.fault_id == "f1", "dc1|a|value"] += 100
    faults = mt.fault_table(["f1", "f2"], rate=pd.Series({"f1": 3.0, "f2": 1.0}))
    access = {"a": cm.Access("a", 1e4, True), "b": cm.Access("b", 10.0, False)}
    board = cm.STM32G474

    def duration(t):
        return cm.cost([t], board, access, PRICE).time

    inst = ilp.instance(m, tests, faults, "dc_w", access, board, PRICE, 1e-6, duration)
    probability, _ = adaptive.draw_weights(faults, "dc_w")
    assert inst.weight.sum() == pytest.approx(1.0) and inst.free == 0.0
    assert inst.coverage(["dc1|a|value"]) == pytest.approx(probability["f1"])
    for plan in (tests[:2], tests[2:], tests):
        c = cm.cost(plan, board, access, PRICE)
        ids = [t.test_id for t in plan]
        assert inst.euros(ids) == pytest.approx(c.euros)
        assert inst.time(ids) == pytest.approx(c.time)
