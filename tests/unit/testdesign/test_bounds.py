import numpy as np
import pandas as pd
import pytest

from spicefault import Experiment
from spicefault.testdesign import bounds as bd
from spicefault.testdesign import candidates as cand


def test_groups_of_a_synthetic_matrix():
    # c0 and c1 are proportional, c2 alone, c3 never seen, c4 = c2 + c0 makes {c0, c2, c4}
    S = np.array([
        [1.0, 2.0, 0.0, 0.0, 1.0],
        [2.0, 4.0, 1.0, 0.0, 3.0],
        [0.0, 0.0, 1.0, 0.0, 1.0],
    ])  # fmt: skip
    names = ["c0", "c1", "c2", "c3", "c4"]
    groups, ambiguous = bd.ambiguity_groups(S, names)
    assert ("c3",) in groups and ("c0", "c1") in groups
    assert ("c1", "c2", "c4") in groups or ("c0", "c2", "c4") in groups
    assert ambiguous == set(names)
    b = bd.bound(pd.DataFrame(S, columns=names), pd.Series(1.0, index=range(3)))
    assert b.rank == 2 and b.identifiable == ()


def test_full_rank_has_no_group_and_noise_limits_the_resolvable_rank():
    S = pd.DataFrame(np.diag([100.0, 10.0, 1.0]), columns=["a", "b", "c"], index=["t0", "t1", "t2"])
    b = bd.bound(S, pd.Series(1.0, index=S.index), deviation=0.1, k=3.0)
    assert b.rank == 3 and b.groups == () and b.identifiable == ("a", "b", "c")
    # 10 % moves the readings by 9.5, 0.95 and 0.095 spreads: only the first is resolvable
    assert b.resolvable == 1
    small = bd.bound(S.loc[["t0"]], pd.Series(1.0, index=["t0"]))
    # t0 alone still identifies a; b and c are never seen
    assert small.identifiable == ("a",)
    assert bd.gap(b, small) == {"rank": 2, "resolvable": 0, "identifiable": 2}


def test_perturbations_are_named_as_sensitivity_reads_them():
    names = [f.fault_id for f in bd.perturbations(["R1"], 0.01)]
    assert names == ["R1:parametric:+0.01", "R1:parametric:-0.01"]


@pytest.mark.ngspice
def test_rc_divider_sensitivity(rc_divider):
    """H = 1 / (1 + R1/R2 + j w R1 C): the gain depends on R1/R2 and R1 C only, so the
    rank is 2 and {R1, R2, C1} is the one group; at DC, C1 is invisible."""
    circuit = cand.prepare_source(rc_divider, "V1")
    stimuli = [cand.ac_tones("V1", [500, 2000]), *cand.dc_levels("V1", [1.0])]
    tests = cand.enumerate_tests(["a"], stimuli)
    comps = ["R1", "R2", "C1"]
    exp = Experiment(
        circuit, faults=bd.perturbations(comps), conditions=cand.conditions(tests),
        samples=1, healthy_samples=1, seed=0,
    )  # fmt: skip
    m = cand.matrix(exp.run().to_frame(), tests)
    S = bd.sensitivity(m, [t.test_id for t in tests], comps)
    # DC: v = R2/(R1+R2); d v / d ln R1 = -1/4 at R1 = R2
    assert S.loc["dc1|a|value", "R1"] == pytest.approx(-0.25, rel=1e-3)
    assert S.loc["dc1|a|value", "R2"] == pytest.approx(0.25, rel=1e-3)
    assert S.loc["dc1|a|value", "C1"] == pytest.approx(0.0, abs=1e-9)
    # phase of H: -atan(w R1 C / (1 + R1/R2)); d / d ln C1 = -x / (1 + x^2) rad, x = w R C / 2
    x = 2 * np.pi * 500 * 1e3 * 100e-9 / 2
    expected = np.degrees(-x / (1 + x**2))
    assert S.loc["ac|a|phase@500", "C1"] == pytest.approx(expected, rel=1e-3)
    b = bd.bound(S, pd.Series(1.0, index=S.index))
    assert b.rank == 2
    assert ("C1", "R1", "R2") in {tuple(sorted(g)) for g in b.groups}
    dc = bd.bound(S.loc[["dc1|a|value"]], pd.Series(1.0, index=["dc1|a|value"]))
    assert ("C1",) in dc.groups and dc.rank == 1
