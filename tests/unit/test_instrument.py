import json

import numpy as np
import pandas as pd
import pytest
from campaign_backend import Divider

from spicefault import (
    Campaign,
    Circuit,
    Instrument,
    Measurement,
    Reading,
    Simulator,
    Specification,
)
from spicefault.faults import ParametricFault
from spicefault.variation import tolerances

CIRCUIT = Circuit("divider\nV1 in 0 dc 1\nR1 in out 10k\nR2 out 0 10k\n.end\n", "divider")
N = 40000


def table(n=N, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "sample_id": np.arange(n), "gain": rng.uniform(10.0, 20.0, n),
        "offset": rng.normal(0.0, 1.0, n), "other": 1.0,
    })  # fmt: skip


def test_noise_resolution_and_range_match_their_closed_forms():
    samples = table()
    noisy = Instrument({"gain": Reading(noise=0.5)}).observe(samples, seed=3)
    error = noisy["gain"] - samples["gain"]
    assert error.std() == pytest.approx(0.5, rel=0.02)
    assert error.mean() == pytest.approx(0, abs=0.01)
    assert noisy["offset"].equals(samples["offset"]) and noisy["other"].equals(samples["other"])
    # rounding to a step q leaves an error that is uniform within it: q / sqrt(12)
    stepped = Instrument({"gain": Reading(resolution=0.25)}).observe(samples)
    error = stepped["gain"] - samples["gain"]
    assert np.allclose(stepped["gain"] / 0.25, np.round(stepped["gain"] / 0.25))
    assert error.abs().max() <= 0.125 and error.std() == pytest.approx(0.25 / 12**0.5, rel=0.02)
    # a range clips what is beyond it: a normal read between -1 and 1 sigma
    clipped = Instrument({"offset": Reading(limits=(-1.0, 1.0))}).observe(samples)
    assert clipped["offset"].min() == -1.0 and clipped["offset"].max() == 1.0
    assert (clipped["offset"] == 1.0).mean() == pytest.approx(0.1587, abs=0.006)
    # the three together add their variances inside the range
    reading = Reading(noise=0.3, resolution=0.4, limits=(0.0, 30.0))
    assert reading.uncertainty == pytest.approx((0.3**2 + 0.4**2 / 12) ** 0.5)
    both = Instrument({"gain": reading}).observe(samples)
    assert (both["gain"] - samples["gain"]).std() == pytest.approx(reading.uncertainty, rel=0.02)
    assert Instrument({"gain": reading}).noise_floor() == {"gain": reading.uncertainty}


def test_a_reading_depends_only_on_the_seed_the_measurement_and_the_sample():
    samples = table(2000)
    instrument = Instrument({"gain": Reading(noise=0.5), "offset": Reading(noise=0.1)})
    full = instrument.observe(samples, seed=1)
    # any order and any subset of the rows
    some = samples.sample(300, random_state=4)
    assert instrument.observe(some, seed=1).equals(full.loc[some.index])
    # whatever other measurements the instrument reads
    alone = Instrument({"offset": Reading(noise=0.1)}).observe(samples, seed=1)
    assert alone["offset"].equals(full["offset"])
    # another seed is another reading; the two measurements are read independently
    again = instrument.observe(samples, seed=2)
    assert not again["gain"].equals(full["gain"])
    errors = (full[["gain", "offset"]] - samples[["gain", "offset"]]).to_numpy()
    assert abs(np.corrcoef(errors.T)[0, 1]) < 0.07
    # a failed simulation has nothing to read
    samples.loc[5, "gain"] = np.nan
    assert np.isnan(instrument.observe(samples).loc[5, "gain"])

    record = json.loads(json.dumps(instrument.metadata()))
    assert Instrument.from_metadata(record) == instrument
    assert Instrument({"gain": {"noise": 0.5}}) == Instrument({"gain": Reading(noise=0.5)})
    with pytest.raises(KeyError, match="no measurement \\['power'\\]"):
        Instrument({"power": Reading(noise=1.0)}).observe(samples)
    with pytest.raises(ValueError, match="low < high"):
        Reading(limits=(1.0, 0.0))
    with pytest.raises(ValueError, match="resolution must be positive"):
        Reading(resolution=0.0)


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    faults = [ParametricFault("R1", deviation=d) for d in (0.01, 0.02, 0.05)]
    campaign = Campaign(
        CIRCUIT, faults, out_dir=tmp_path_factory.mktemp("instrument") / "data", samples=120,
        healthy_samples=600, variations=tolerances(CIRCUIT, {"R": 0.002}),
        simulator=Simulator(Divider()), seed=5,
        measurements=[Measurement.value("v(out)", name="vout")],
    )  # fmt: skip
    campaign.run(progress=False)
    return campaign.dataset()


def test_one_dataset_gives_the_detectability_against_the_noise_of_the_instrument(dataset):
    stored = dataset.samples.copy()
    curve = {}
    for noise in (0.0, 0.002, 0.01):
        instrument = Instrument({"vout": Reading(noise=noise)}) if noise else None
        analysis = dataset.analysis(instrument=instrument, seed=1, healthy_split=0.5)
        curve[noise] = analysis.detectability(alpha=0.02).table["p_detect"]
        assert (analysis.instrument is None) == (noise == 0.0)
    # a 2 % deviation moves the output by 0.5 %: plain without noise, lost in 10 mV of it
    fault = "R1:parametric:+0.02"
    assert curve[0.0][fault] > 0.95 > curve[0.002][fault] > curve[0.01][fault]
    assert curve[0.01][fault] < 0.2 and curve[0.01]["R1:parametric:+0.05"] < curve[0.0][fault]
    analysis = dataset.analysis(instrument=Instrument({"vout": Reading(noise=0.01)}))
    assert analysis.noise_floor["vout"] == 0.01
    assert analysis.instrument == {
        "readings": {"vout": {"noise": 0.01, "resolution": None, "limits": None}}, "seed": 0,
    }

    # the yield that a test with the instrument would find, and what it adds to the spread
    limits = [Specification("vout", 0.499, 0.501)]
    true = dataset.yield_report(limits).loc["all", "yield"]
    read = dataset.yield_report(limits, instrument=Instrument({"vout": Reading(noise=0.002)}))
    assert read.loc["all", "yield"] < true - 0.2 and read.attrs["instrument"]["seed"] == 0
    wide = dataset.statistics(instrument=Instrument({"vout": Reading(noise=0.002)}), seed=7)
    plain = dataset.statistics()
    healthy = ("healthy", "nominal", "vout")
    added = (wide.loc[healthy, "std"] ** 2 - plain.loc[healthy, "std"] ** 2) ** 0.5
    assert added == pytest.approx(0.002, rel=0.1) and plain.attrs["instrument"] is None
    assert wide.attrs["instrument"]["seed"] == 7

    # nothing was written: the dataset is the one that was simulated
    assert dataset.samples.equals(stored) and dataset.verify() == []
    observed = dataset.observe(Instrument({"vout": Reading(resolution=0.001)}))
    assert not observed["vout"].equals(stored["vout"]) and dataset.samples.equals(stored)
    with pytest.raises(KeyError, match="not measurements of this dataset"):
        dataset.observe(Instrument({"sample_id": Reading(noise=1.0)}))
