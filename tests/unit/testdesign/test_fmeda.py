import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spicefault import OperatingCondition, SimulationConfig
from spicefault.faults import FaultUniverse, short_rule
from spicefault.measurements import Measurement
from spicefault.testdesign import candidates as cand
from spicefault.testdesign import faultprior as fp
from spicefault.testdesign import fmeda
from spicefault.testdesign import metrics as mt
from spicefault.testdesign.metrics import adaptive
from spicefault.testdesign.metrics import formulation as fm
from spicefault.variation import tolerances

DATA = Path(__file__).resolve().parents[2] / "data"


def universe():
    return pd.read_csv(DATA / "fmeda_universe.csv")


def table():
    """The fault table of `faultprior.prior` joined with the fault universe."""
    u = universe()
    prior = fp.prior(u, "gf")
    return prior.join(u.set_index("fault_id")[["component", "fault_type", "effect", "deviation"]])


def risk(t):
    return mt.fault_table(list(t.index), t.rate, t.severity, t.iec, t.iso)


def detection(t, seed, n=20):
    """Circuits and alarms of each fault, drawn at random."""
    rng = np.random.default_rng(seed)
    detected = rng.binomial(n, rng.uniform(0, 1, len(t)))
    per_fault = pd.DataFrame({"n": n, "detected": detected}, index=t.index)
    return per_fault.assign(p=per_fault.detected / per_fault.n)


def bernoulli(p: dict, seed: int = 0):
    def sampler(fault_id, start, stop):
        rng = np.random.default_rng([seed, zlib.crc32(fault_id.encode())])
        return rng.random(stop)[start:stop] < p[fault_id]

    return sampler


def conditions():
    op = SimulationConfig(analyses=("op",), outputs=("a",))
    a = (Measurement.value("v(a)", name="a"),)
    return (
        OperatingCondition("low", settings={("V1", "dc"): 1.0}, config=op, measurements=a),
        OperatingCondition("high", settings={("V1", "dc"): 2.0}, config=op, measurements=a),
    )


def recalculated(path):
    """Every cell of a workbook, recalculated by `formulas` (no Excel), as SHEET!A1."""
    formulas = pytest.importorskip("formulas")
    solution = formulas.ExcelModel().loads(str(path)).finish().calculate()
    out = {}
    for key, value in solution.items():
        cell = str(key).split("]")[-1].replace("'", "").upper()
        out[cell] = value.value[0, 0] if hasattr(value, "value") else value
    return out


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_totals_with_the_lower_bounds_are_the_bonferroni_bound(seed):
    t = table()
    per_fault = detection(t, seed)
    s = fmeda.summary(fmeda.rows(fmeda.faults(t, per_fault)))
    ref = mt.coverage(mt.Detection((), per_fault, {}, pd.Series(), None, 0.95), risk(t))
    for ours, theirs in (("claimed", "lower"), ("estimate", "estimate"), ("upper", "upper")):
        assert np.allclose(s[ours], ref.loc[s.index, theirs], equal_nan=True)
    assert (s.lower == s.claimed).all()


def test_a_mode_is_split_by_severity_and_class():
    r = fmeda.rows(fmeda.faults(table(), detection(table(), 1)))
    change = r[(r.component == "R1") & (r["mode"] == "Parameter Change")]
    assert sorted(change.severity_class) == [0, 3] and set(change.faults) == {2}
    assert change.class_share.sum() == pytest.approx(1.0)
    assert set(change.iso) == {"safe", "single_point"}


def test_every_fault_in_one_row_and_the_rates_add_up():
    t = table()
    f = fmeda.faults(t, detection(t, 1))
    r = fmeda.rows(f)
    assessed = r[r.iec != fmeda.NOT_ASSESSED]
    assert assessed.faults.sum() == len(f)
    first = t.groupby("component").first()
    found = assessed.groupby("component")["lambda"].sum()
    assert np.allclose(found, (first.component_rate * first.modelled_share).loc[found.index])
    whole = r.groupby("component")["lambda"].sum()
    assert np.allclose(whole, first.component_rate.loc[whole.index])
    # a FET short and open are modelled, its stuck outputs and parameter change are not
    q1 = r[(r.component == "Q1") & (r.iec == fmeda.NOT_ASSESSED)]
    assert set(q1["mode"]) == {"Output Low", "Output High", "Parameter Change"}
    assert (q1.dc_claimed == 0).all() and (q1[list(fmeda.LAMBDAS)] == 0).all().all()


def test_modes_not_assessed_as_single_point_lower_the_spfm():
    t = table()
    r = fmeda.rows(fmeda.faults(t, detection(t, 1)))
    s = fmeda.summary(r, {"spfm": 0.5})
    na = r[r.iec == fmeda.NOT_ASSESSED]["lambda"].sum()
    assessed = r[r.iec != fmeda.NOT_ASSESSED]
    residual = assessed.lambda_spf.sum() + assessed.lambda_rf.sum()
    worst = 1 - (residual + na) / (assessed["lambda"].sum() + na)
    assert s.loc["spfm", "not assessed as single point"] == pytest.approx(worst)
    assert worst < s.claimed["spfm"]
    assert s.met["spfm"] == (s.claimed["spfm"] >= 0.5) and pd.isna(s.met["dc"])


def test_the_claimed_metric_is_the_lower_bound_of_certify():
    t = table()
    rng = np.random.default_rng(0)
    p = dict(zip(t.index, rng.uniform(0.8, 1.0, len(t)), strict=True))
    c = adaptive.certify(bernoulli(p), risk(t), 0.5, "spfm", sizes=[25, 50, 100])
    s = fmeda.summary(fmeda.rows(fmeda.faults(t, c.per_fault, claim="mode")))
    assert s.claimed["spfm"] == pytest.approx(c.lower, abs=1e-12)


def test_bounds_of_certify_are_kept_only_for_a_claim_per_mode():
    t = table()
    given = detection(t, 1).assign(lower=0.123, upper=0.987)
    f = fmeda.faults(t, given, claim="mode")
    assert (f.lower == 0.123).all() and (f.upper == 0.987).all()
    assert not (fmeda.faults(t, given, claim="metric").lower == 0.123).any()


def test_claim_per_metric_claims_the_estimate():
    t = table()
    per_fault = detection(t, 1)
    per_fault.loc[t.iso != "single_point", ["n", "detected"]] = 0
    r = fmeda.rows(fmeda.faults(t, per_fault, claim="metric"), claim="metric")
    simulated = per_fault[per_fault.n > 0]
    p = (simulated.detected / simulated.n).reindex(t.index).fillna(0)
    assert fmeda.summary(r).claimed["spfm"] == pytest.approx(mt.metric_values(p, risk(t))["spfm"])


def test_unknown_claim_and_unknown_fault():
    t = table()
    with pytest.raises(ValueError):
        fmeda.faults(t, detection(t, 1), claim="row")
    with pytest.raises(KeyError):
        fmeda.faults(t.iloc[1:], detection(t, 1))


def test_tests_that_alarm_are_the_safety_mechanism():
    t = table()
    tests = {"R1:open": {"b", "a"}, "R1:short": "c"}
    r = fmeda.rows(fmeda.faults(t, detection(t, 1), tests=tests))
    stuck = r[(r.component == "R1") & (r["mode"] == "Open")]
    assert stuck.safety_mechanism.iloc[0] == "a b"
    assert r[(r.component == "R1") & (r["mode"] == "Short")].safety_mechanism.iloc[0] == "c"


@pytest.mark.parametrize("seed", [1, 2])
def test_workbook_formulas_give_the_summary(seed, tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    t = table()
    f = fmeda.faults(t, detection(t, seed))
    r = fmeda.rows(f)
    s = fmeda.summary(r, {"spfm": 0.6})
    sampled = pd.DataFrame({"metric": ["SPFM"], "lower": [0.61]})
    path = fmeda.write(tmp_path / "f.xlsx", r, f, s, {"Case": "test"}, sampled=sampled)
    cells = recalculated(path)
    for i, m in enumerate(fmeda.METRICS, start=2):
        assert cells[f"SUMMARY!B{i}"] == pytest.approx(s.claimed[m], abs=1e-12)
    assert cells["SUMMARY!H5"] == ("yes" if s.claimed["spfm"] >= 0.6 else "no")
    na = r[r.iec == fmeda.NOT_ASSESSED]["lambda"].sum()
    assert cells["SUMMARY!B9"] == pytest.approx(na)
    assert cells["SUMMARY!B10"] == pytest.approx(na / r["lambda"].sum())
    wb = openpyxl.load_workbook(path)
    assert wb.sheetnames == ["Cover", "FMEDA", "Faults", "Summary", "Assumptions"]
    # every fault points to the FMEDA row of its component and mode
    faults = wb["Faults"]
    for i in range(2, faults.max_row + 1):
        row = faults.cell(i, 2).value
        assert wb["FMEDA"].cell(row, 1).value == faults.cell(i, 3).value
        assert wb["FMEDA"].cell(row, 4).value == faults.cell(i, 4).value


def test_same_content_same_file(tmp_path):
    pytest.importorskip("openpyxl")
    a, b = fmeda.template(tmp_path / "a.xlsx"), fmeda.template(tmp_path / "b.xlsx")
    assert a.read_bytes() == b.read_bytes()
    t = table()
    f = fmeda.faults(t, detection(t, 1))
    r = fmeda.rows(f)
    s = fmeda.summary(r)
    one = fmeda.write(tmp_path / "1.xlsx", r, f, s, {"Case": "test"}).read_bytes()
    assert fmeda.write(tmp_path / "2.xlsx", r, f, s, {"Case": "test"}).read_bytes() == one


def test_empty_template_gives_blank_totals(tmp_path):
    cells = recalculated(fmeda.template(tmp_path / "t.xlsx", blank=5))
    assert [cells[f"SUMMARY!B{i}"] for i in range(2, 7)] == [""] * 5


def test_table_fmeda_filled_by_hand_in_the_template(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    path = fmeda.template(tmp_path / "t.xlsx")
    wb = openpyxl.load_workbook(path)
    ws = wb["FMEDA"]
    # Ref, Part, λ part, mode, mode share, class share; IEC, ISO; DC basis; DC claimed
    hand = [
        ("R1", "resistor", 10.0, "Open", 0.59, 1.0, "dangerous", "single point",
         "table, ISO 26262-5 D, high", 0.99),
        ("C1", "capacitor", 20.0, "Short", 0.49, 1.0, "dangerous", "single point",
         "table, ISO 26262-5 D, medium", 0.90),
        ("R2", "resistor", 10.0, "Parameter Change", 0.36, 1.0, "safe", "safe", "none", 0.0),
        ("U1", "resistor", 10.0, "Open", 0.50, 1.0, "safe", "multiple point",
         "table, ISO 26262-5 D, low", 0.60),
    ]  # fmt: skip
    for i, values in enumerate(hand, start=2):
        for col, v in zip("ABCDEFKLNT", values, strict=True):
            ws[f"{col}{i}"] = v
        ws[f"J{i}"] = 1.0
    wb.save(path)
    cells = recalculated(path)
    lam = np.array([5.9, 9.8, 3.6, 5.0])
    residual = 5.9 * 0.01 + 9.8 * 0.10
    # ISO 26262-5 Annex C: SPFM = 1 − Σ λ_SPF,RF / Σ λ; LFM = 1 − Σ λ_MPF,L / (Σ λ − Σ λ_SPF,RF)
    assert cells["SUMMARY!B5"] == pytest.approx(1 - residual / lam.sum())
    assert cells["SUMMARY!B6"] == pytest.approx(1 - 5.0 * 0.4 / (lam.sum() - residual))
    # IEC 61508-2 Annex C: SFF = (Σ λ_S + Σ λ_DD) / (Σ λ_S + Σ λ_D)
    assert cells["SUMMARY!B4"] == pytest.approx((3.6 + 5.0 + 5.9 * 0.99 + 9.8 * 0.9) / lam.sum())


def test_a_detector_without_limits_cannot_record():
    class Any:
        def fit(self, healthy):
            return self

        def flag(self, x):
            return np.zeros(len(x), dtype=bool)

    with pytest.raises(TypeError):
        mt.outside(Any(), np.zeros((1, 1)))
    limits = mt.GaussianLimits(0.01).fit(np.random.default_rng(0).normal(size=(100, 2)))
    x = np.array([[0.0, 0.0], [0.0, 50.0]])
    assert mt.outside(limits, x).tolist() == [False, True]


@pytest.mark.ngspice
def test_samplers_record_the_tests_that_alarm(rc_divider):
    universe = FaultUniverse(rc_divider, [short_rule("R")])
    variations = tolerances(rc_divider, {"R": 0.01})
    conds, ids = fm.prefixed(conditions())
    with fm.ConditionSampler(
        rc_divider, universe.selected(), conds, ids, mt.GaussianLimits(0.01),
        variations=variations, n_calibration=40, n_max=20, record=True,
    ) as sampler:  # fmt: skip
        sampler("R1:short", 0, 5)
        sampler(mt.HEALTHY, 0, 5)
        assert sampler.alarming == {"R1:short": set(ids)}
    circuit = cand.prepare_source(rc_divider, "V1")
    tests = cand.enumerate_tests(["a"], cand.dc_levels("V1", [1.0]))
    with adaptive.SpiceSampler(
        circuit, universe.selected(), tests, mt.GaussianLimits(0.01),
        variations=variations, n_calibration=40, n_max=20, record=True,
    ) as sampler:  # fmt: skip
        sampler("R2:short", 0, 5)
        assert sampler.alarming == {"R2:short": {tests[0].test_id}}
    plain = fm.ConditionSampler(
        rc_divider, universe.selected(), conds, ids, mt.GaussianLimits(0.01),
        variations=variations, n_calibration=40, n_max=20,
    )  # fmt: skip
    assert plain.alarming is None
