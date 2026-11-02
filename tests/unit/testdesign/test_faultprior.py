import pandas as pd
import pytest

from spicefault.testdesign import faultprior as fp
from spicefault.testdesign import metrics as mt


def universe():
    rows = [
        ("R1:open", "R1", "resistor", "open_circuit", None, "none", 0),
        ("R1:short", "R1", "resistor", "short_circuit", None, "masks", 3),
        *[(f"R1:p{d}", "R1", "resistor", "parametric", d, "none", 0)
          for d in (-0.5, -0.1, 0.1, 0.5)],
        ("F1:open", "F1", "fuse", "open_circuit", None, "masks", 3),
        ("S1:open", "S1", "sensor", "open_circuit", None, "self-test only", 0),
        ("S1:short", "S1", "sensor", "short_circuit", None, "lost", 2),
    ]
    columns = ["fault_id", "component", "part", "fault_type", "deviation", "effect", "severity"]
    return pd.DataFrame(rows, columns=columns)


def test_component_rate_in_fit():
    # MIL-HDBK-217F Notice 2, film chip resistor, ground benign: 0.0037 per 10^6 h, pi_Q 10
    assert fp.component_rate("resistor", "gb") == pytest.approx(0.0037 * 10 * 1000)
    assert fp.component_rate("fuse", "gm", fp.rates()) == pytest.approx(80.0)
    with pytest.raises(ValueError):
        fp.component_rate("resistor", "space")


def test_tables_can_be_replaced(tmp_path):
    path = tmp_path / "rates.csv"
    path.write_text("category,gb,gf,gm,pi_q\nresistor,0.001,0.002,0.003,1\n")
    assert fp.component_rate("resistor", "gf", fp.rates(path)) == pytest.approx(2.0)


def test_mode_shares_leave_out_unmodelled_modes():
    m = fp.modes()
    assert fp.mode_shares("resistor") == pytest.approx(
        {"open_circuit": 0.59, "parametric": 0.36, "short_circuit": 0.05}
    )
    assert fp.mode_shares("fuse", m) == pytest.approx({"open_circuit": 0.08})
    for category in set(m.category):
        assert m[m.category == category].alpha.sum() == pytest.approx(1.0)
    with pytest.raises(KeyError):
        fp.mode_shares("inductor")


def test_rates_split_by_mode_and_level():
    r = fp.fault_rates(universe(), "gb", parts={"S1": "ntc"})
    lam = 37.0
    assert r.loc["R1:open", "rate"] == pytest.approx(lam * 0.59)
    assert r.loc["R1:p0.1", "rate"] == pytest.approx(lam * 0.36 / 4)
    resistor = [i for i in r.index if i.startswith("R1")]
    assert r.loc[resistor, "rate"].sum() == pytest.approx(lam)
    assert r.loc["R1:open", "modelled_share"] == pytest.approx(1.0)
    assert r.loc["F1:open", "modelled_share"] == pytest.approx(0.08)
    # the NTC takes the thermistor rate and the transducer modes
    assert r.loc["S1:short", "rate"] == pytest.approx(0.0014 * 10 * 1000 * 0.05)
    assert r.loc["S1:open", "modelled_share"] == pytest.approx(0.17)
    assert list(r.index) == list(universe().fault_id)


def test_unknown_part_is_refused():
    with pytest.raises(KeyError):
        fp.fault_rates(universe(), "gb")  # the sensor is not said to be an NTC or a gauge


def test_classes_and_weights_fit_metrics():
    p = fp.prior(universe(), "gf", parts={"S1": "strain_gauge"})
    short = p.loc["R1:short", ["iec", "iso", "severity"]].tolist()
    assert short == ["dangerous", "single_point", 3.0]
    assert p.loc["S1:open", "iso"] == "multiple_point" and p.loc["S1:open", "iec"] == "safe"
    assert p.loc["S1:short", "severity"] == 2.0
    table = mt.fault_table(p.index, p.rate, p.severity, p.iec, p.iso)
    assert table.rate.sum() == pytest.approx(p.rate.sum())
    steep = fp.prior(universe(), "gf", weights=(0, 1, 10, 100), parts={"S1": "strain_gauge"})
    assert steep.loc["R1:short", "severity"] == 100
    assert steep.loc["R1:short", "severity_class"] == 3
