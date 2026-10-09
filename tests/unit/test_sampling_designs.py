import math

import numpy as np
import pytest
from campaign_backend import Divider

from spicefault import Campaign, Circuit, Experiment, Measurement, Simulator, Specification
from spicefault.experiments import sample_stream
from spicefault.experiments.sampling import latin_hypercube, uniforms
from spicefault.statistics import replicated_interval, student_t_quantile
from spicefault.variation import (
    CatalogueVariation,
    LogNormalVariation,
    ToleranceVariation,
    tolerances,
)

CIRCUIT = Circuit("divider\nV1 in 0 dc 1\nR1 in out 10k\nR2 out 0 10k\n.end\n", "divider")
BAND = tolerances(CIRCUIT, {"R": 0.003})


def test_latin_hypercube_has_one_point_in_each_stratum_of_each_parameter():
    points = latin_hypercube(sample_stream(0), 50, 4)
    assert points.shape == (50, 4) and (points > 0).all() and (points < 1).all()
    for column in points.T:
        assert sorted(np.floor(column * 50).astype(int)) == list(range(50))
    assert abs(np.corrcoef(points.T)[0, 1]) < 0.4  # paired at random, not in order
    # the design of a population depends on the seed, its key and its size, and on nothing else
    assert np.array_equal(uniforms("lhs", 3, (0,), 20, 2), uniforms("lhs", 3, (0,), 20, 2))
    assert not np.array_equal(uniforms("lhs", 3, (0,), 20, 2), uniforms("lhs", 3, (1,), 20, 2))
    assert not np.array_equal(uniforms("lhs", 3, (0,), 20, 2), uniforms("lhs", 4, (0,), 20, 2))
    with pytest.raises(ValueError, match="unknown sampling method"):
        uniforms("grid", 0, (0,), 4, 2)


def sampled(method, samples=64, seed=1, **kwargs):
    return Experiment(CIRCUIT, variations=BAND, samples=samples, seed=seed, sampling=method,
                      **kwargs)  # fmt: skip


def test_every_tolerance_is_stratified_in_a_latin_hypercube_experiment():
    experiment = sampled("lhs")
    values = np.array([list(experiment.realise(s).parameters.values()) for s in experiment.plan()])
    for column in values.T:  # 64 values, one in each 64th of the band of 0.3 %
        strata = np.floor((column / 1e4 - 0.997) / 0.006 * 64).astype(int)
        assert sorted(strata) == list(range(64))
    # the same samples whatever the order they are realised in, and others for another seed
    backwards = [experiment.realise(s).netlist for s in reversed(experiment.plan())]
    assert backwards[::-1] == [sampled("lhs").realise(s).netlist for s in experiment.plan()]
    assert sampled("lhs", seed=2).realise(experiment.plan()[0]).netlist != backwards[-1]
    assert experiment.metadata()["sampling"] == {"method": "lhs"}
    assert "sampling" not in sampled("random").metadata()
    rebuilt = Experiment.from_metadata(experiment.metadata(), CIRCUIT.to_netlist())
    assert rebuilt.sampling == "lhs" and rebuilt.realise(rebuilt.plan()[5]).netlist == backwards[-6]

    # each fault has its own population, and its own design
    from spicefault.faults import ParametricFault

    faulty = sampled("lhs", samples=8, healthy_samples=16,
                     faults=[ParametricFault("R1", deviation=0.2)])  # fmt: skip
    drawn = [faulty.realise(s).parameters["R2", "value"] for s in faulty.plan()]
    assert sorted(np.floor((np.array(drawn[:16]) / 1e4 - 0.997) / 0.006 * 16)) == list(range(16))
    assert sorted(np.floor((np.array(drawn[16:]) / 1e4 - 0.997) / 0.006 * 8)) == list(range(8))

    with pytest.raises(ValueError, match="the CatalogueVariation of R1 has none"):
        Experiment(CIRCUIT, sampling="lhs",
                   variations=[CatalogueVariation("R1", {"a": {"value": 1.0}})])  # fmt: skip
    with pytest.raises(ValueError, match="unknown sampling method 'grid'"):
        sampled("grid")


def test_sobol_points_cover_the_band_and_are_recorded_with_their_scipy():
    scipy = pytest.importorskip("scipy")
    experiment = sampled("sobol", samples=128)
    values = np.array([list(experiment.realise(s).parameters.values()) for s in experiment.plan()])
    assert (values >= 9970).all() and (values <= 10030).all()
    # a balanced sequence: exactly half of the 128 points in each half of each band
    assert ((values < 1e4).sum(axis=0) == 64).all()
    assert experiment.metadata()["sampling"] == {"method": "sobol", "scipy": scipy.__version__}
    again = sampled("sobol", samples=128)
    sample = experiment.plan()[17]
    assert again.realise(sample).netlist == experiment.realise(sample).netlist


@pytest.mark.parametrize("method", ["lhs", "sobol"])
def test_stratified_campaign_estimates_better_and_claims_no_interval(method, tmp_path):
    if method == "sobol":
        pytest.importorskip("scipy")

    def run(name, sampling, seed, workers=1):
        campaign = Campaign(
            CIRCUIT, out_dir=tmp_path / name, samples=64, variations=BAND, sampling=sampling,
            simulator=Simulator(Divider()), seed=seed,
            measurements=[Measurement.value("v(out)", name="vout")],
        )  # fmt: skip
        campaign.run(workers=workers, chunk=20, progress=False)
        return campaign.dataset()

    dataset = run("data", method, seed=0, workers=2)
    assert dataset.verify() == [] and dataset.metadata["sampling"]["method"] == method
    other = run("other", method, seed=0, workers=1)
    assert other.samples.drop(columns="elapsed_s").equals(dataset.samples.drop(columns="elapsed_s"))
    assert dataset.reproduce(n=10, experiment=dataset.experiment(simulator=Simulator(Divider())))[
        "max_abs_diff"
    ].max() == 0.0

    # the estimates are given, the intervals that assume independence are not
    table = dataset.statistics(quantiles=(0.5,))
    row = table.iloc[0]
    assert row["n"] == 64 and np.isfinite(row[["mean", "std", "q0.5"]].astype(float)).all()
    assert np.isnan(row[["mean_low", "mean_high", "q0.5_low", "q0.5_high"]].astype(float)).all()
    assert "replicated_interval" in table.attrs["intervals"]
    report = dataset.yield_report([Specification("vout", 0.4995, 0.5005)])
    assert 0 < report.loc["all", "yield"] < 1 and np.isnan(report.loc["all", "ci_low"])
    with pytest.raises(ValueError, match=f"come from a {method} design"):
        dataset.analysis()

    # the mean of the ratio is exactly 0.5: over 12 seeds a stratified design is closer to
    # it than random sampling, and the replications give its interval
    means = {"random": [], method: []}
    for seed in range(12):
        for sampling in means:
            measured = [Measurement.value("v(out)", name="vout")]
            experiment = sampled(sampling, seed=seed, simulator=Simulator(Divider()),
                                 measurements=measured)  # fmt: skip
            means[sampling].append(experiment.run().to_frame()["vout"].mean())
    error = {
        name: float(np.sqrt(np.mean((np.array(v) - 0.5) ** 2))) for name, v in means.items()
    }
    assert error[method] < 0.5 * error["random"]
    mean, low, high = replicated_interval(means[method])
    assert low < 0.5 < high and high - low < 4 * error["random"]


def test_replicated_interval_and_the_student_t_quantile():
    # tabulated values of the t distribution
    for df, expected in ((1, 12.7062), (4, 2.7764), (9, 2.2622), (30, 2.0423)):
        assert student_t_quantile(0.975, df) == pytest.approx(expected, abs=2e-4)
    assert student_t_quantile(0.5, 5) == pytest.approx(0.0, abs=1e-9)
    assert student_t_quantile(0.05, 9) == pytest.approx(-1.8331, abs=2e-4)
    mean, low, high = replicated_interval([1.0, 2.0, 3.0, 4.0, 5.0])
    assert mean == 3.0 and high - mean == pytest.approx(2.7764 * math.sqrt(2.5 / 5), abs=1e-3)
    assert mean - low == pytest.approx(high - mean)
    # it covers a known mean as often as it says
    rng = np.random.default_rng(0)
    covered = 0
    for _ in range(200):
        _, low, high = replicated_interval(rng.normal(7.0, 2.0, 6))
        covered += low <= 7.0 <= high
    assert 0.91 <= covered / 200 <= 0.985
    with pytest.raises(ValueError, match="at least two"):
        replicated_interval([1.0])
    assert LogNormalVariation("R1", None, 0.1).quantile(0.5, 3.0) == pytest.approx(3.0)
    assert ToleranceVariation("R1", 0.1).quantile(0.5, 3.0) == pytest.approx(3.0)
