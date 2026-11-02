import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from spicefault.testdesign import candidates as cand
from spicefault.testdesign import costmodel as cm

MCU = cm.STM32G474
TONES = cand.ac_tones("V1", [1e3, 2e5])
STEP = cand.step("V1", 1.0, 1e-3, [1e-6, 1e-4])
DC = cand.dc_levels("V1", [1.0])[0]
PRICE = {"external_opamp_channel": 0.1, "mux_8": 0.25, "injection_switch": 0.4}


def make_tests(nodes, stimuli=(TONES, STEP, DC)):
    return cand.enumerate_tests(nodes, list(stimuli))


def matrix(tests, values, n=400):
    """`n` fault-free rows and `n` rows of fault `f`, each test at its value in `values`
    (fault-free, faulty)."""
    rows = []
    for fault, i in (("healthy", 0), ("f", 1)):
        for r in range(n):
            readings = {t.test_id: values[t.test_id][i] for t in tests}
            rows.append({"fault_id": fault, "replica": r, **readings})
    return pd.DataFrame(rows)


def test_adc_noise_from_sinad():
    adc = MCU.adc
    assert adc.lsb == pytest.approx(3.3 / 4096)
    # the noise and the quantisation together give back the SINAD of the datasheet
    total = math.hypot(adc.noise, adc.lsb / math.sqrt(12))
    assert 20 * math.log10(3.3 / (2 * math.sqrt(2)) / total) == pytest.approx(65.6)
    assert adc.rate(47.5) == pytest.approx(1e6)


def test_feasibility():
    found = {t.test_id: cm.feasible(t, MCU) for t in make_tests(["a"])}
    assert found["ac|a|gain_db@1000"] == ""
    assert "tone above" in found["ac|a|gain_db@200000"]
    assert "instant before" in found["step|a|v@1e-06"]
    assert found["step|a|v@0.0001"] == "" and found["dc1|a|value"] == ""


def test_observation_keeps_large_signals_and_drowns_small_ones():
    ts = make_tests(["big", "tiny"], [TONES])
    values = {t.test_id: (0.0, -1.0) if t.node == "big" else (-100.0, -101.0) for t in ts}
    values.update({t.test_id: (10.0, 20.0) for t in ts if "phase" in t.test_id})
    read = cm.observe(matrix(ts, values), ts, MCU, amplitude=1.0)
    big, tiny = "ac|big|gain_db@1000", "ac|tiny|gain_db@1000"
    by = read.groupby("fault_id")
    # 1 dB at 0 dB is far above the noise; 1 dB at -100 dB (10 µV) is buried in it
    assert by[big].mean()["healthy"] - by[big].mean()["f"] == pytest.approx(1.0, abs=0.05)
    assert by[big].std()["healthy"] < 0.05
    assert by[tiny].std()["healthy"] > 1.0
    phase = by["ac|tiny|phase@1000"].std()["healthy"]
    assert phase > 10 * by["ac|big|phase@1000"].std()["healthy"]


def test_observation_is_deterministic_and_order_free():
    ts = make_tests(["a"], [DC])
    m = matrix(ts, {"dc1|a|value": (0.5, 0.6)}, n=50)
    one = cm.observe(m, ts, MCU, seed=3)
    two = cm.observe(m.sample(frac=1, random_state=1), ts, MCU, seed=3).loc[one.index]
    pd.testing.assert_frame_equal(one, two)
    assert not one.equals(cm.observe(m, ts, MCU, seed=4))


def test_draws_depend_only_on_the_row():
    faults = np.array(["f", "f", "g", "f"])
    replicas = np.array([0, 3, 0, 7])
    all_rows = cm.draws(1, "t", faults, replicas)
    # the same row drawn within a smaller matrix gets the same value
    assert cm.draws(1, "t", faults[[1]], replicas[[1]])[0] == all_rows[1]
    assert cm.draws(1, "t", faults[[2, 0]], replicas[[2, 0]]).tolist() == all_rows[[2, 0]].tolist()
    assert cm.draws(1, "u", faults, replicas)[1] != all_rows[1]


def test_voltage_is_quantised_and_clipped():
    ts = make_tests(["a"], [DC])
    read = cm.observe(matrix(ts, {"dc1|a|value": (0.1, 5.0)}, n=50), ts, MCU)
    volts = read["dc1|a|value"] + MCU.adc.vref / 2
    assert np.allclose(np.round(volts / MCU.adc.lsb), volts / MCU.adc.lsb)
    assert read[read.fault_id == "f"]["dc1|a|value"].max() <= MCU.adc.vref / 2


def test_amplitudes_fit_the_adc_and_the_dac():
    ts = make_tests(["a"], [TONES, DC])
    values = {t.test_id: (6.0, 0.0) for t in ts}
    values["dc1|a|value"] = (0.1, 0.0)
    a = cm.amplitudes(matrix(ts, values, n=5), ts, MCU)
    # a gain of 6 dB (x 2) fills 90 % of the half-range at 0.74 V; the DC level would
    # allow 14.85 V, but the DAC gives only 1.45 V
    assert a["ac"] == pytest.approx(0.9 * 1.65 / 10 ** (6 / 20))
    assert a["dc1"] == pytest.approx(MCU.dac.amplitude)


def test_cost_uses_internal_resources_first():
    nodes = [f"n{i}" for i in range(20)]
    access = {n: cm.Access(n, 1e4, buffer=i < 8) for i, n in enumerate(nodes)}
    ts = [t for t in make_tests(nodes, [TONES]) if t.feature.name == "gain_db@1000"]
    c = cm.cost(ts[:4], MCU, access, PRICE)
    assert c.items == {"external_opamp_channel": 0, "mux_8": 0, "injection_switch": 1}
    assert c.euros == pytest.approx(0.4)
    c = cm.cost(ts, MCU, access, PRICE)
    # 8 buffers, 6 internal; 21 channels with the stimulus, 16 free: one multiplexer
    assert c.items == {"external_opamp_channel": 2, "mux_8": 1, "injection_switch": 1}
    assert c.euros == pytest.approx(0.2 + 0.25 + 0.4)
    assert c.time == pytest.approx(20 * (5 / 1e3 + 1024 / 1e6))
    tight = replace(MCU, opamps=0)
    assert cm.cost(ts[:4], tight, access, PRICE).items["external_opamp_channel"] == 4


def test_prices_from_a_table(tmp_path):
    path = tmp_path / "prices.csv"
    path.write_text("item,price_eur\nexternal_opamp_channel,0.1\nmux_8,0.25\ninjection_switch,0.4\n")
    assert cm.prices(path) == PRICE


@pytest.mark.ngspice
def test_node_impedance_and_access(rc_divider):
    z = cm.node_impedance(rc_divider, "a", [1.0, 1e5])
    assert z[0] == pytest.approx(500, rel=1e-3)
    assert z[1] == pytest.approx(1 / (2 * math.pi * 1e5 * 100e-9), rel=0.01)
    found = cm.access(rc_divider, ["in", "a"], MCU)
    # the source node is driven by an ideal source; 500 ohm settles on a slow channel
    assert found["in"].impedance < 1 and not found["a"].buffer
    slow = replace(MCU, adc=replace(MCU.adc, max_source={47.5: (400, 300)}))
    assert cm.access(rc_divider, ["a"], slow)["a"].buffer
