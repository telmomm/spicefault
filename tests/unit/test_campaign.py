import json

import numpy as np
import pandas as pd
import pytest
from campaign_backend import Divider

from spicefault import (
    Circuit,
    Experiment,
    FaultCampaign,
    Measurement,
    OperatingCondition,
    SimulationConfig,
    Simulator,
    Waveform,
)
from spicefault.dataset import load_metadata
from spicefault.faults import OpenCircuit, ParametricFault, ShortCircuit
from spicefault.variation import tolerances

CIRCUIT = Circuit("divider\nV1 in 0 dc 1\nR1 in out 10k\nR2 out 0 10k\n.end\n", "divider")
FAULTS = [OpenCircuit("R2"), ShortCircuit("R2"), ParametricFault("R1", deviation=0.2)]
N = (10 + 3 * 6) * 2  # healthy and three faults, under two conditions
LOW = {("V1", "dc"): 0.5}


def campaign(out_dir, **kwargs) -> FaultCampaign:
    defaults = {
        "samples_per_fault": 6,
        "healthy_samples": 10,
        "variations": tolerances(CIRCUIT, {"R": 0.01}),
        "conditions": [OperatingCondition(), OperatingCondition("low", settings=LOW)],
        "simulator": Simulator(Divider()),
        "measurements": [Measurement.value("v(out)", name="vout"), Measurement.final("v(out)")],
        "waveform": Waveform("v(out)", fs=10.0, duration=1.0),
        "seed": 3,
    }
    return FaultCampaign(CIRCUIT, FAULTS, out_dir=out_dir, **{**defaults, **kwargs})


def reproducible(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns="elapsed_s")


def test_campaign_writes_every_sample_with_its_status(tmp_path):
    c = campaign(tmp_path / "data")
    assert c.status() == {"total": N, "completed": 0, "failed": 0, "pending": N}
    assert c.run(workers=2, chunk=20, progress=False) == tmp_path / "data"

    df, waveforms, manifest = c.load(drop_failed=False)
    assert len(df) == len(waveforms) == N and list(df["sample_id"]) == list(range(N))
    assert list(df.columns[:14]) == [
        "sample_id", "fault_index", "fault_id", "fault_type", "fault_location",
        "fault_magnitude", "fault_severity", "replica", "condition", "seed_key", "status",
        "message", "sim_ok", "elapsed_s",
    ]
    assert list(df.columns[14:]) == ["p_R1_value", "p_R2_value", "vout", "final_v(out)"]

    # the open does not converge; some draws have R2 too large for the fake backend
    by_status = df.groupby("status")["fault_id"].agg(lambda s: sorted(set(s))).to_dict()
    assert by_status["CONVERGENCE_ERROR"] == ["R2:open"]
    assert "healthy" in by_status["INVALID_OUTPUT"] and "R2:short" in by_status["SUCCESS"]
    invalid = df[df["status"] == "INVALID_OUTPUT"]
    assert invalid["message"].str.startswith("measurement failed: KeyError").all()
    assert (invalid["p_R2_value"] > 10040).all()  # still traceable to the drawn circuit

    ok = df["sim_ok"].to_numpy()
    assert (df["status"] == "SUCCESS").equals(df["sim_ok"])
    assert df.loc[~ok, ["vout", "final_v(out)"]].isna().all().all()
    assert np.isnan(waveforms[~ok]).all() and not np.isnan(waveforms[ok]).any()
    good = df[ok]
    source = np.where(good["condition"] == "low", 0.5, 1.0)
    r1 = good["p_R1_value"] * np.where(good["fault_type"] == "parametric", 1.2, 1.0)
    shorted = good["fault_type"] == "short_circuit"
    r2 = np.where(shorted, 1 / (1 / good["p_R2_value"] + 1), good["p_R2_value"])
    assert np.allclose(good["vout"], source * r2 / (r1 + r2), rtol=1e-12)
    assert np.allclose(waveforms[ok][:, -1], 0.9 * good["final_v(out)"], rtol=1e-6)

    # the manifest and the summaries agree with the table
    n_ok = int(ok.sum())
    assert manifest["n_samples"] == N and manifest["n_completed"] == n_ok
    assert manifest["n_failed"] == N - n_ok and manifest["status_counts"]["CONVERGENCE_ERROR"] == 12
    assert load_metadata(c.out_dir) == json.loads(json.dumps(c.experiment.metadata()))
    assert (c.out_dir / "circuit.cir").read_text() == CIRCUIT.to_netlist()
    assert manifest["workers"] == 2 and manifest["chunk"] == 20
    assert c.status() == {"total": N, "completed": n_ok, "failed": N - n_ok, "pending": 0}
    kept, kept_waveforms, _ = c.load()
    assert len(kept) == len(kept_waveforms) == n_ok

    summary = c.summary()
    assert list(summary.index) == ["healthy", "R2:open", "R2:short", "R1:parametric:+0.2"]
    assert summary.loc["R2:open"].to_dict() == {
        "samples": 12, "success_rate": 0.0, "CONVERGENCE_ERROR": 12, "INVALID_OUTPUT": 0,
        "SUCCESS": 0,
    }
    assert summary["samples"].tolist() == [20, 12, 12, 12]
    failures = c.failures()
    assert len(failures) == N - n_ok and set(failures["status"]) == {
        "CONVERGENCE_ERROR", "INVALID_OUTPUT",
    }


def test_dataset_does_not_depend_on_workers_or_chunking(tmp_path):
    one = campaign(tmp_path / "a")
    many = campaign(tmp_path / "b")
    one.run(workers=1, chunk=2000, progress=False)
    many.run(workers=3, chunk=7, progress=False)
    a, wa, _ = one.load(drop_failed=False)
    b, wb, _ = many.load(drop_failed=False)
    pd.testing.assert_frame_equal(reproducible(a), reproducible(b), check_exact=True)
    assert np.array_equal(wa, wb, equal_nan=True)


def test_interrupted_campaign_resumes_where_it_stopped(tmp_path):
    reference = campaign(tmp_path / "ref")
    reference.run(chunk=10, progress=False)

    c = campaign(tmp_path / "data")
    c.run(chunk=10, progress=False)
    # put the campaign back in the state of an interrupted run: chunks, no dataset
    c2 = campaign(tmp_path / "again")
    with pytest.raises(RuntimeError, match="stop"):
        c2.experiment.simulator = Simulator(Stops(after=25))
        c2.run(chunk=10, progress=False, validate=False)
    assert c2.status() == {"total": N, "completed": c2.status()["completed"], "failed":
                           c2.status()["failed"], "pending": N - 20}  # fmt: skip
    assert not (tmp_path / "again" / "manifest.json").exists()

    c2.experiment.simulator = Simulator(Divider())
    c2.run(chunk=10, progress=False)
    resumed, waveforms, manifest = c2.load(drop_failed=False)
    expected, expected_waveforms, _ = reference.load(drop_failed=False)
    pd.testing.assert_frame_equal(reproducible(resumed), reproducible(expected), check_exact=True)
    assert np.array_equal(waveforms, expected_waveforms, equal_nan=True)
    assert manifest["resumed"] is True and c2.status()["pending"] == 0


class Stops(Divider):
    """Breaks down after a number of simulations, like an interrupted run."""

    def __init__(self, after):
        self.left = after

    def run(self, netlist, config):
        self.left -= 1
        if self.left < 0:
            raise RuntimeError("stop")
        return super().run(netlist, config)


def test_restart_and_finished_campaigns(tmp_path):
    c = campaign(tmp_path / "data")
    c.experiment.simulator = Simulator(Stops(after=25))
    with pytest.raises(RuntimeError):
        c.run(chunk=10, progress=False, validate=False)
    assert len(list((tmp_path / "data" / "parts").glob("part_*.parquet"))) == 2

    # the definitions changed: the partial run belongs to another campaign
    other = campaign(tmp_path / "data", seed=4)
    with pytest.raises(ValueError, match="another configuration"):
        other.run(chunk=10, progress=False)
    other.run(chunk=10, progress=False, resume=False)  # start again from nothing
    _, _, manifest = other.load()
    assert manifest["resumed"] is False and load_metadata(other.out_dir)["seed"] == 4

    # a finished campaign is returned as it is, and never mixed with another one
    created = manifest["created"]
    other.run(progress=False)
    assert other.load()[2]["created"] == created
    with pytest.raises(FileExistsError, match="another campaign"):
        campaign(tmp_path / "data").run(progress=False)


def test_validation_finds_definitions_that_do_not_fit(tmp_path):
    assert campaign(tmp_path / "ok").validate().ok
    bad = campaign(
        tmp_path / "bad",
        conditions=[OperatingCondition("x", settings={("V9", "dc"): 1.0})],
        measurements=[Measurement.value("v(none)")],
    )
    bad.experiment.faults = (*FAULTS, OpenCircuit("R7"))
    report = bad.validate()
    assert not report.ok
    assert any(e.startswith("fault R7:open: KeyError") for e in report.errors)
    assert any(e.startswith("condition x: KeyError") for e in report.errors)
    with pytest.raises(ValueError, match="the campaign is not valid"):
        bad.run(progress=False)
    assert not (tmp_path / "bad").exists()

    unmeasurable = campaign(tmp_path / "m", measurements=[Measurement.value("v(none)")])
    report = unmeasurable.validate()
    assert report.errors == [
        "measurement value_v(none), condition nominal: KeyError: 'v(none)'",
        "measurement value_v(none), condition low: KeyError: 'v(none)'",
    ]
    assert unmeasurable.validate(simulate=False).ok

    silent = campaign(tmp_path / "s", measurements=[], waveform=None)
    assert silent.validate().warnings == [
        "no measurement and no waveform: only the status is recorded"
    ]
    overlapping = campaign(tmp_path / "o")
    overlapping.experiment.faults = (ParametricFault("R1", deviation=0.01),)
    assert "1 parametric fault conditions lie partly inside" in str(overlapping.validate())


def test_a_fault_that_cannot_be_injected_does_not_stop_the_run(tmp_path):
    c = campaign(tmp_path / "data", conditions=[OperatingCondition()])
    c.experiment.faults = (OpenCircuit("R7"), FAULTS[2])
    c.run(progress=False, validate=False)
    df, _, _ = c.load(drop_failed=False)
    broken = df[df["fault_id"] == "R7:open"]
    assert (broken["status"] == "FAILED").all() and len(broken) == 6
    assert broken["message"].str.startswith("could not build the netlist: KeyError").all()
    assert df[df["fault_id"] != "R7:open"]["status"].isin(["SUCCESS", "INVALID_OUTPUT"]).all()


def test_campaign_from_an_experiment(tmp_path):
    experiment = Experiment(
        CIRCUIT, faults=FAULTS[1:], samples=2, simulator=Simulator(Divider()),
        measurements=[Measurement.value("v(out)")],
    )
    c = FaultCampaign.from_experiment(experiment, tmp_path / "data")
    c.run(progress=False)
    df, waveforms, manifest = c.load(drop_failed=False)
    assert len(df) == 6 and waveforms is None and load_metadata(c.out_dir)["waveform"] is None


def test_condition_can_override_config_and_measurements(tmp_path):
    bench = OperatingCondition(
        "bench",
        config=SimulationConfig(analyses=("tran 1u 1m",), outputs=("v(out)",)),
        measurements=(Measurement.value("v(out)", name="bench_out"),),
    )
    c = campaign(
        tmp_path / "data",
        conditions=[OperatingCondition(), bench],
        waveform=None,
    )
    c.run(progress=False)
    dataset = c.dataset()
    assert dataset.features == ["vout", "final_v(out)", "bench_out"]
    service = dataset.samples[dataset.samples["condition"] == "nominal"]
    testbench = dataset.samples[dataset.samples["condition"] == "bench"]
    assert service["bench_out"].isna().all()
    assert testbench[["vout", "final_v(out)"]].isna().all().all()
    assert testbench.loc[testbench["sim_ok"], "bench_out"].notna().all()
    assert testbench.loc[~testbench["sim_ok"], "bench_out"].isna().all()
    rebuilt = dataset.experiment(simulator=Simulator(Divider()))
    assert rebuilt.conditions[1] == bench


def divider_values(result):
    """Two values from one reading of the output; the second is undefined below 0.497 V."""
    out = float(result.plot("Operating Point")["v(out)"][0].real)
    return {"vout": out, "margin": out - 0.497 if out >= 0.497 else float("nan")}


def test_group_measurement_fills_one_column_per_name(tmp_path):
    c = campaign(
        tmp_path / "data",
        conditions=[OperatingCondition()],
        measurements=[Measurement.group(("vout", "margin"), divider_values, name="divider"),
                      Measurement.final("v(out)")],
        waveform=None,
    )  # fmt: skip
    assert c.experiment.measurement_columns == ("vout", "margin", "final_v(out)")
    c.run(progress=False)
    dataset = c.dataset()
    assert dataset.features == ["vout", "margin", "final_v(out)"] and dataset.verify() == []
    df = dataset.samples
    ok = df[df["sim_ok"]]
    assert np.allclose(ok["margin"], ok["vout"] - 0.497) and len(ok) >= 5
    # a value of the group that is not finite fails the sample, and none of them is kept:
    # every divider with R1 20 % high gives 0.4545 V
    low = df[df["fault_id"] == "R1:parametric:+0.2"]
    assert set(low["status"]) == {"INVALID_OUTPUT"} and low["vout"].isna().all()
    assert low["message"].str.contains("a measurement is not finite").sum() >= 4
    assert load_metadata(c.out_dir)["measurements"][0] == {
        "name": "divider", "kind": "group", "vector": "", "analysis": 0, "parameters": {},
        "function": "divider_values", "names": ["vout", "margin"],
    }
    with pytest.raises(ValueError, match="repeated measurement identifiers: \\['vout'\\]"):
        campaign(tmp_path / "twice", measurements=[
            Measurement.group(("vout", "margin"), divider_values),
            Measurement.value("v(out)", name="vout"),
        ])  # fmt: skip


def test_nominal_circuit_is_simulated_and_measured_under_each_condition(tmp_path):
    e = campaign(tmp_path / "data").experiment
    nominal = e.nominal()
    assert list(nominal) == ["nominal", "low"]
    assert nominal["nominal"].measurements == {"vout": 0.5, "final_v(out)": 0.5}
    assert nominal["low"].measurements["vout"] == 0.25  # no variation: exactly half of 0.5 V
    reference = nominal["nominal"]
    assert reference.fault_id == "healthy" and reference.parameters == {} and reference.result.ok
    assert reference.sample.sample_id == -1 and reference.sample.condition_index == 0
    assert reference.result.plot("Transient Analysis")["v(out)"][-1] == 0.5

    # with a fault: 10k in series with 12k, and a divider shorted by 1 ohm
    high = e.nominal("R1:parametric:+0.2")["nominal"]
    assert high.measurements["vout"] == pytest.approx(10 / 22) and high.sample.fault_index == 3
    shorted = e.nominal(ShortCircuit("R2", r_short=1.0))["low"]
    assert shorted.measurements["vout"] == pytest.approx(0.5 / 10001, rel=1e-3)
    other = e.nominal(ParametricFault("R2", deviation=-0.5))["nominal"]
    assert other.sample.fault_index == -1 and other.measurements["vout"] == pytest.approx(1 / 3)
    # each condition gives its own measurements, not the empty columns of the others
    bench = OperatingCondition("bench", measurements=(Measurement.final("v(out)", name="end"),))
    mixed = campaign(tmp_path / "mixed", conditions=[OperatingCondition(), bench]).experiment
    assert {name: list(r.measurements) for name, r in mixed.nominal().items()} == {
        "nominal": ["vout", "final_v(out)"], "bench": ["end"],
    }
    # a simulation that fails is returned with its status
    failed = e.nominal("R2:open")["nominal"]
    assert failed.result.status.value == "CONVERGENCE_ERROR" and failed.measurements == {}
    with pytest.raises(KeyError):
        e.nominal("R9:open")


def test_fault_tags_become_label_columns_and_user_metadata_is_recorded(tmp_path):
    faults = [
        OpenCircuit("R2", tags={"origin": "electrode", "part": "lead"}),
        ShortCircuit("R2"),
    ]
    c = FaultCampaign(
        CIRCUIT,
        faults,
        out_dir=tmp_path / "data",
        samples_per_fault=2,
        healthy_samples=2,
        simulator=Simulator(Divider()),
        measurements=[Measurement.value("v(out)", name="vout")],
        conditions=[OperatingCondition()],
        tag_columns=("origin", "part"),
        metadata={"git_commit": "abc123", "limits": {"vout_max": 1.2}},
    )
    c.run(progress=False)
    dataset = c.dataset()
    assert dataset.labels == ["origin", "part"]
    healthy = dataset.samples[dataset.samples["fault_id"] == "healthy"]
    tagged = dataset.samples[dataset.samples["fault_id"] == "R2:open"]
    untagged = dataset.samples[dataset.samples["fault_id"] == "R2:short"]
    assert (healthy[["origin", "part"]] == "").all().all()
    assert set(tagged["origin"]) == {"electrode"} and set(tagged["part"]) == {"lead"}
    assert (untagged[["origin", "part"]] == "").all().all()
    assert dataset.manifest.user == {"git_commit": "abc123", "limits": {"vout_max": 1.2}}
    assert dataset.verify() == []


@pytest.mark.ngspice
def test_campaign_with_ngspice(tmp_path):
    circuit = Circuit(
        "rc\nV1 in 0 dc 1 ac 1 pulse(0 1 1m 1u 1u 5m 20m)\nR1 in out 10k\nC1 out 0 100n\n"
        "R2 out 0 10k\n.end\n",
        "rc",
    )
    c = FaultCampaign(
        circuit,
        [OpenCircuit("R2"), ParametricFault("C1", deviation=0.5)],
        out_dir=tmp_path / "data",
        samples_per_fault=3,
        variations=tolerances(circuit, {"R": 0.01, "C": 0.05}),
        config=SimulationConfig(("op", "ac dec 20 1 1e5", "tran 10u 10m"), outputs=("v(out)",)),
        measurements=[
            Measurement.value("v(out)", name="dc"),
            Measurement.magnitude("v(out)", 1e3, name="gain_1k"),
            Measurement.peak("v(out)", name="peak"),
        ],
        waveform=Waveform("v(out)", 10e3, 10e-3),
        seed=1,
    )
    assert c.validate().ok
    c.run(workers=2, progress=False)
    df, waveforms, manifest = c.load()
    assert len(df) == 9 and manifest["status_counts"] == {"SUCCESS": 9}
    assert manifest["simulator"].startswith("ngspice-") and waveforms.shape == (9, 100)
    r1, r2, c1 = df["p_R1_value"], df["p_R2_value"], df["p_C1_value"]
    r2 = r2 + np.where(df["fault_type"] == "open_circuit", 1e9, 0.0)
    c1 = c1 * np.where(df["fault_type"] == "parametric", 1.5, 1.0)
    assert np.allclose(df["dc"], r2 / (r1 + r2), rtol=1e-6)
    thevenin = r1 * r2 / (r1 + r2)
    gain = (r2 / (r1 + r2)) / np.sqrt(1 + (2 * np.pi * 1e3 * thevenin * c1) ** 2)
    assert np.allclose(df["gain_1k"], gain, rtol=1e-3)
    assert np.allclose(waveforms.max(axis=1), df["peak"], rtol=1e-3)


def test_definitions_that_workers_cannot_receive_are_reported(tmp_path):
    local = campaign(tmp_path / "data", measurements=[Measurement.custom("x", lambda plot: 1.0)])
    with pytest.raises(TypeError, match="must be defined at module level"):
        local.run(progress=False)
    with pytest.raises(TypeError, match="must be defined at module level"):
        local.experiment.run(workers=2)
    assert len(local.experiment.run(workers=1)) == N  # in this process it is fine
    assert not (tmp_path / "data").exists()


def test_in_memory_run_also_records_a_measurement_that_fails(tmp_path):
    """`Experiment.run` treats an unusable output as the campaign does: a status, not an error."""
    c = campaign(tmp_path / "data", waveform=None)
    c.run(progress=False)
    stored = c.load(drop_failed=False)[0]
    result = c.experiment.run(workers=1)
    assert [s.result.status.value for s in result] == list(stored["status"])
    invalid = [s for s in result if s.result.status.value == "INVALID_OUTPUT"]
    assert invalid and all(s.result.message.startswith("measurement failed") for s in invalid)
    assert all(s.measurements == {} and not s.result.ok for s in invalid)
