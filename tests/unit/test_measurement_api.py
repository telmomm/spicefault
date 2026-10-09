import json

import numpy as np
import pytest

from spicefault import (
    Measurement,
    SimulationConfig,
    SimulationResult,
    SimulationStatus,
    Simulator,
    Waveform,
)
from spicefault.simulation import Plot, build_deck

# a transient on an uneven grid, dense in the first tenth as a simulator would make it
T = np.concatenate([np.linspace(0, 0.1, 400, endpoint=False), np.linspace(0.1, 1.0, 50)])
RAMP = 2.0 * T + 1.0  # mean 2, from 1 to 3
SINE = 1.5 + np.sin(2 * np.pi * 5 * np.linspace(0, 1, 20001))
FREQ = np.logspace(0, 5, 501)
POLE = 10.0 / (1 + 1j * FREQ / 1e3)

RESULT = SimulationResult(
    SimulationStatus.SUCCESS,
    (
        Plot("Operating Point", {"v(out)": np.array([0.5]), "v(in)": np.array([1.0])}),
        Plot("AC Analysis", {"frequency": FREQ + 0j, "v(out)": POLE}),
        Plot("Transient Analysis", {"time": T, "v(out)": RAMP}),
        Plot("Transient Analysis", {"time": np.linspace(0, 1, 20001), "v(out)": SINE}),
        Plot("AC Analysis", {"frequency": FREQ + 0j, "v(out)": 2 * POLE}),
    ),
)


def test_time_statistics_are_weighted_by_time():
    assert Measurement.mean("v(out)")(RESULT) == pytest.approx(2.0, rel=1e-12)
    assert RAMP.mean() == pytest.approx(1.21, abs=0.01)  # what an average of the points gives
    # mean square of a + b t over [0, 1]: a^2 + a b + b^2 / 3, integrated by trapezoids
    assert Measurement.rms("v(out)")(RESULT) == pytest.approx((1 + 2 + 4 / 3) ** 0.5, rel=1e-4)
    assert Measurement.variance("v(out)")(RESULT) == pytest.approx(4 / 12, rel=1e-3)
    assert Measurement.peak("V(OUT)")(RESULT) == 3.0
    assert Measurement.minimum("v(out)")(RESULT) == 1.0
    assert Measurement.peak_to_peak("v(out)")(RESULT) == 2.0
    assert Measurement.final("v(out)")(RESULT) == 3.0
    assert Measurement.at_time("v(out)", 0.25)(RESULT) == pytest.approx(1.5)


def test_sine_with_offset():
    kwargs = {"analysis": 3}
    rms = (1.5**2 + 0.5) ** 0.5
    assert Measurement.mean("v(out)", **kwargs)(RESULT) == pytest.approx(1.5, abs=1e-9)
    assert Measurement.rms("v(out)", **kwargs)(RESULT) == pytest.approx(rms, rel=1e-7)
    assert Measurement.variance("v(out)", **kwargs)(RESULT) == pytest.approx(0.5, rel=1e-6)
    assert Measurement.peak_to_peak("v(out)", **kwargs)(RESULT) == pytest.approx(2.0, abs=1e-6)


def test_window_is_cut_at_interpolated_ends():
    assert Measurement.mean("v(out)", window=(0.2, 0.6))(RESULT) == pytest.approx(1.8, rel=1e-12)
    assert Measurement.peak("v(out)", window=(0.2, 0.6))(RESULT) == pytest.approx(2.2)
    assert Measurement.minimum("v(out)", window=(0.2, 5.0))(RESULT) == pytest.approx(1.4)
    with pytest.raises(ValueError, match="outside the simulated time"):
        Measurement.mean("v(out)", window=(2.0, 3.0))(RESULT)
    with pytest.raises(ValueError, match="outside the simulated time"):
        Measurement.at_time("v(out)", 1.5)(RESULT)


def test_operating_point_and_frequency_response():
    assert Measurement.value("v(out)")(RESULT) == 0.5 and Measurement.value("v(in)")(RESULT) == 1.0
    assert Measurement.magnitude("v(out)", 1e3)(RESULT) == pytest.approx(10 / 2**0.5, rel=1e-4)
    assert Measurement.magnitude("v(out)", 1e3, db=True)(RESULT) == pytest.approx(16.99, abs=0.01)
    assert Measurement.phase("v(out)", 1e3)(RESULT) == pytest.approx(-45.0, abs=0.01)
    second_sweep = Measurement.magnitude("v(out)", 10.0, analysis=4)
    assert second_sweep(RESULT) == pytest.approx(20.0, rel=1e-3)
    with pytest.raises(ValueError, match="outside the sweep"):
        Measurement.magnitude("v(out)", 1e6)(RESULT)


def test_custom_measurement_and_missing_data():
    gain = Measurement.custom("corner", lambda plot: abs(plot["v(out)"][300]), analysis="ac")
    assert gain(RESULT) == pytest.approx(10 / 2**0.5, rel=1e-6)
    with pytest.raises(KeyError):
        Measurement.value("v(none)")(RESULT)
    with pytest.raises(KeyError):
        Measurement.value("v(out)", analysis="dc")(RESULT)


def test_custom_result_measurement_reads_multiple_plots():
    ratio = Measurement.custom_result(
        "sweep_ratio",
        lambda result: abs(result.plot(1)["v(out)"][300])
        / abs(result.plot(4)["v(out)"][300]),
    )
    assert ratio(RESULT) == pytest.approx(0.5)


def test_names_and_metadata():
    assert Measurement.rms("V(out)").name == "rms_v(out)"
    m = Measurement.magnitude("v(out)", 50.0, analysis=4, name="gain_50", db=True)
    record = json.loads(json.dumps(m.metadata()))
    assert record == {
        "name": "gain_50", "kind": "magnitude", "vector": "v(out)", "analysis": 4,
        "parameters": {"frequency": 50.0, "db": True},
    }
    assert Measurement.mean("v(out)", window=(0.1, 0.2)).metadata()["parameters"] == {
        "window": [0.1, 0.2]
    }
    assert Measurement.custom("c", len).metadata()["function"] == "len"


def test_waveform_is_resampled_on_a_uniform_grid():
    waveform = Waveform("v(out)", fs=100.0, duration=0.5)
    samples = waveform(RESULT)
    assert samples.dtype == np.float32 and samples.shape == (50,) == (waveform.n_points,)
    assert np.allclose(samples, 2.0 * np.arange(50) / 100 + 1.0, atol=1e-6)
    assert waveform.metadata()["n_points"] == 50


# --- several values from one function -------------------------------------------------


def test_group_gives_several_named_values_from_one_call():
    calls = []

    def specifications(result):
        calls.append(result)
        dc = float(result.plot("Operating Point")["v(out)"][0].real)
        return {"dc": dc, "ratio": dc / float(result.plot("Operating Point")["v(in)"][0].real),
                "unused": 0.0}  # fmt: skip

    group = Measurement.group(("dc", "ratio"), specifications)
    assert group.columns == ("dc", "ratio")
    assert Measurement.value("v(out)").columns == ("value_v(out)",)
    assert group.values(RESULT) == {"dc": 0.5, "ratio": 0.5} and len(calls) == 1
    assert Measurement.value("v(out)", name="dc").values(RESULT) == {"dc": 0.5}
    record = json.loads(json.dumps(group.metadata()))
    assert record["kind"] == "group" and record["names"] == ["dc", "ratio"]
    assert record["name"] == "dc+ratio" and record["function"].endswith("specifications")
    with pytest.raises(NotImplementedError, match="holds a function"):
        Measurement.from_metadata(record)
    with pytest.raises(TypeError, match="use values"):
        group(RESULT)
    with pytest.raises(KeyError, match="did not return \\['gain'\\]"):
        Measurement.group(("dc", "gain"), specifications).values(RESULT)
    with pytest.raises(TypeError, match="must return a dictionary"):
        Measurement.group(("dc",), lambda result: 0.5).values(RESULT)
    with pytest.raises(ValueError, match="unique, non-empty names"):
        Measurement.group(("dc", "dc"), specifications)


# --- commands between analyses --------------------------------------------------------

DECK = "rc\nV1 in 0 dc 1 ac 0\nR1 in out 10k\nC1 out 0 100n\n.end\n"
STEPS = SimulationConfig(
    analyses=("op", "alter @v1[acmag]=1", "ac dec 10 1 1e4", "alter @v1[acmag]=2",
              "ac dec 10 1 1e4"),
    outputs=("v(out)",),
)  # fmt: skip


def test_only_analyses_are_followed_by_a_write():
    assert STEPS.plots() == ["op", "ac dec 10 1 1e4", "ac dec 10 1 1e4"]
    control = build_deck(DECK, STEPS).split(".control\n")[1].split("\n")
    assert control[2:10] == [
        "op", "write out.raw v(out)", "alter @v1[acmag]=1", "ac dec 10 1 1e4",
        "write out.raw v(out)", "alter @v1[acmag]=2", "ac dec 10 1 1e4", "write out.raw v(out)",
    ]


@pytest.mark.ngspice
def test_source_altered_between_two_sweeps():
    result = Simulator().run(DECK, STEPS)
    assert result.ok and len(result.plots) == 3
    low = Measurement.magnitude("v(out)", 1.0, analysis=1)(result)
    assert low == pytest.approx(1.0, rel=1e-4)
    assert Measurement.magnitude("v(out)", 1.0, analysis=2)(result) == pytest.approx(2 * low)
