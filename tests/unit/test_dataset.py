import json

import numpy as np
import pandas as pd
import pytest
from campaign_backend import Divider

from spicefault import (
    Circuit,
    Dataset,
    Experiment,
    FaultCampaign,
    Measurement,
    OperatingCondition,
    SimulationConfig,
    Simulator,
    VariationSet,
    Waveform,
    split_by_magnitude,
    split_by_replica,
)
from spicefault.dataset import Manifest, git_source
from spicefault.experiments import run_chunks, simulate_sample
from spicefault.faults import LeakageFault, OpenCircuit, ParametricFault, ShortCircuit
from spicefault.variation import (
    FixedVariation,
    JointVariation,
    LogNormalVariation,
    LogUniformVariation,
    NormalVariation,
    ToleranceVariation,
    UniformVariation,
    tolerances,
    variation_from_metadata,
)

CIRCUIT = Circuit("divider\nV1 in 0 dc 1\nR1 in out 10k\nR2 out 0 10k\n.end\n", "divider")
FAULTS = [
    OpenCircuit("R2"),
    ShortCircuit("R2"),
    ParametricFault("R1", deviation=0.2),
    LeakageFault("R2", 1e5, r_min=1e3, r_max=1e7),
]
SIMULATOR = Simulator(Divider())


def campaign(out_dir, **kwargs) -> FaultCampaign:
    defaults = {
        "samples_per_fault": 5,
        "healthy_samples": 20,
        "variations": tolerances(CIRCUIT, {"R": 0.01}),
        "conditions": [OperatingCondition(), OperatingCondition("low", 85, {("V1", "dc"): 0.5})],
        "simulator": SIMULATOR,
        "measurements": [Measurement.value("v(out)", name="vout"), Measurement.final("v(out)")],
        "waveform": Waveform("v(out)", fs=10.0, duration=1.0),
        "seed": 5,
        "seeding": "content",
    }
    return FaultCampaign(CIRCUIT, FAULTS, out_dir=out_dir, **{**defaults, **kwargs})


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    c = campaign(tmp_path_factory.mktemp("dataset") / "data")
    c.run(workers=2, chunk=30, progress=False)
    return c


def half_supply(result):
    return float(result.plot("Operating Point")["v(out)"][0].real) - 0.5


def two_setups(out_dir, **kwargs) -> FaultCampaign:
    """`service` and `bench` simulate the same drawn circuits and measure different things."""
    conditions = [
        OperatingCondition("service", measurements=(Measurement.value("v(out)", name="vout"),)),
        OperatingCondition(
            "bench",
            settings={("V1", "dc"): 2.0},
            measurements=(Measurement.custom_result("offset", half_supply),),
        ),
    ]
    defaults = {
        "samples_per_fault": 3, "healthy_samples": 4, "conditions": conditions,
        "measurements": (), "waveform": None,
    }  # fmt: skip
    return campaign(out_dir, **{**defaults, **kwargs})


@pytest.fixture(scope="module")
def setups(tmp_path_factory):
    c = two_setups(tmp_path_factory.mktemp("setups") / "data")
    c.run(progress=False)
    return c


def test_dataset_knows_the_role_of_its_columns(run):
    dataset = run.dataset()
    assert len(dataset) == (20 + 4 * 5) * 2
    assert "80 samples, 4 faults, 2 measurements" in repr(dataset)
    assert dataset.features == ["vout", "final_v(out)"] and dataset.labels == []
    assert dataset.parameters == {"p_R1_value": ("R1", "value"), "p_R2_value": ("R2", "value")}
    assert dataset.waveforms.shape == (80, 10)
    assert dataset.ok.sum() == dataset.manifest.summary["n_completed"] < 80  # the open fails
    assert dataset.circuit.to_netlist() == CIRCUIT.to_netlist()
    assert dataset.faults.ids() == [f.fault_id for f in FAULTS]
    assert sorted(p.name for p in dataset.path.iterdir()) == [
        "circuit.cir", "manifest.json", "metadata.json", "samples.parquet", "waveforms.npy",
    ]


def test_samples_carry_what_was_injected(run):
    df = run.dataset().samples
    one = df.groupby("fault_id", sort=False).first()
    assert one["fault_location"].to_dict() == {
        "healthy": "", "R2:open": "R2", "R2:short": "R2", "R1:parametric:+0.2": "R1",
        "R2:leakage:100000": "R2",
    }
    size = ["fault_magnitude", "fault_severity"]
    assert one.loc["R1:parametric:+0.2", size].tolist() == [0.2, 0.2]
    assert one.loc["R2:leakage:100000", "fault_magnitude"] == 1e5
    assert one.loc["R2:leakage:100000", "fault_severity"] == pytest.approx(0.5)
    assert one.loc[["healthy", "R2:open"], size].isna().all().all()


# --- manifest and integrity -------------------------------------------------------------


def test_manifest_is_a_typed_record(run):
    manifest = run.dataset().manifest
    assert manifest.schema_version == 1 and manifest.n_samples == 80 and manifest.workers == 2
    assert manifest.spicefault_version and manifest.python and manifest.platform
    assert manifest.elapsed_total_s >= 0 and manifest.resumed is False
    assert set(manifest.files) == {
        "samples.parquet", "waveforms.npy", "metadata.json", "circuit.cir",
    }
    assert all(len(f["sha256"]) == 64 and f["bytes"] > 0 for f in manifest.files.values())
    assert manifest.summary["status_counts"]["CONVERGENCE_ERROR"] == 10
    record = json.loads((run.out_dir / "manifest.json").read_text())
    assert Manifest.from_dict(record) == manifest and manifest.to_dict() == record
    assert record["n_completed"] == manifest.summary["n_completed"]  # written flat
    with pytest.raises(ValueError, match="unsupported manifest schema version"):
        Manifest.from_dict({**record, "schema_version": 99})


def test_manifest_records_the_commit_of_the_project_that_ran_the_campaign(run, tmp_path):
    import shutil
    import subprocess

    # the tests run from the repository of the library, or from a copy without one
    here = git_source()
    assert run.dataset().manifest.source == here
    record = json.loads((run.out_dir / "manifest.json").read_text())
    assert record.get("source", {}) == here and Manifest.from_dict(record).source == here

    assert git_source(tmp_path) == {}  # not a repository
    if shutil.which("git") is None:
        pytest.skip("git not installed")

    def git(*arguments):
        identity = ["-c", "user.name=test", "-c", "user.email=test@example.org"]
        subprocess.run(["git", *identity, *arguments], cwd=tmp_path, check=True,
                       capture_output=True)  # fmt: skip

    git("init")
    (tmp_path / "study.py").write_text("seed = 1\n")
    git("add", "study.py")
    git("commit", "-m", "study")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=tmp_path, capture_output=True,
                            text=True, check=True).stdout.strip()  # fmt: skip
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "samples.parquet").write_text("")  # what a campaign writes is untracked
    assert git_source(tmp_path / "data") == {"commit": commit, "dirty": False}
    (tmp_path / "study.py").write_text("seed = 2\n")
    assert git_source(tmp_path) == {"commit": commit, "dirty": True}


def test_verify_detects_altered_files(run, tmp_path):
    import shutil

    assert run.dataset().verify() == []
    copy = tmp_path / "copy"
    shutil.copytree(run.out_dir, copy)

    df = pd.read_parquet(copy / "samples.parquet")
    df.loc[3, "vout"] = 0.123
    df.to_parquet(copy / "samples.parquet", index=False)
    assert Dataset(copy).verify() == ["samples.parquet: content differs from the manifest"]

    (copy / "circuit.cir").write_text(CIRCUIT.to_netlist().replace("10k", "11k", 1))
    waveforms = np.load(copy / "waveforms.npy")
    np.save(copy / "waveforms.npy", waveforms[:-1])
    problems = Dataset(copy).verify()
    assert "circuit.cir: content differs from the manifest" in problems
    assert "circuit.cir is not the netlist of the experiment" in problems
    assert "waveforms and samples have different lengths" in problems
    (copy / "waveforms.npy").unlink()
    assert "waveforms.npy: missing" in Dataset(copy).verify()


def test_verify_expects_only_the_measurements_of_each_condition(setups, tmp_path):
    import shutil

    dataset = setups.dataset()
    service = (dataset.samples["condition"] == "service").to_numpy()
    assert dataset.ok[service].any() and dataset.ok[~service].any()
    assert dataset.samples.loc[dataset.ok & service, "offset"].isna().all()
    assert dataset.samples.loc[dataset.ok & ~service, "offset"].notna().all()
    assert dataset.verify() == []

    def altered(name, column, row, value):
        copy = tmp_path / name
        shutil.copytree(setups.out_dir, copy)
        df = pd.read_parquet(copy / "samples.parquet")
        df.loc[row, column] = value
        df.to_parquet(copy / "samples.parquet", index=False)
        return Dataset(copy).verify()[1:]  # after the fingerprint of the table

    first_service, first_bench = (int(np.flatnonzero(dataset.ok & rows)[0])
                                  for rows in (service, ~service))  # fmt: skip
    assert altered("lost", "offset", first_bench, np.nan) == [
        "a successful sample has a measurement that is not finite"
    ]
    assert altered("extra", "offset", first_service, 1.0) == [
        "a sample has a measurement that its operating condition does not declare"
    ]


def test_waveform_is_stored_for_the_conditions_that_declare_one(tmp_path):
    service, bench = two_setups(tmp_path / "unused").experiment.conditions
    stored = Waveform("v(out)", fs=10.0, duration=1.0)
    conditions = [
        OperatingCondition("service", measurements=service.measurements, waveform=stored),
        bench,
    ]
    c = two_setups(tmp_path / "data", conditions=conditions)
    assert c.experiment.waveform_for(0) == stored and c.experiment.waveform_for(1) is None
    c.run(progress=False)
    dataset = c.dataset()
    in_service = (dataset.samples["condition"] == "service").to_numpy()
    assert dataset.waveforms.shape == (len(dataset), 10) and dataset.verify() == []
    assert np.isfinite(dataset.waveforms[dataset.ok & in_service]).all()
    assert np.isnan(dataset.waveforms[~in_service]).all() and dataset.ok[~in_service].any()
    x, y = dataset.to_ml(waveforms=True)
    assert len(x) == len(y) == (dataset.ok & in_service).sum() and np.isfinite(x).all()
    record = dataset.metadata["conditions"]
    assert record[0]["waveform"]["n_points"] == 10 and "waveform" not in record[1]
    rebuilt = dataset.experiment(simulator=SIMULATOR, conditions=conditions)
    assert dataset.reproduce(n=12, experiment=rebuilt)["waveform_abs_diff"].dropna().max() == 0.0

    # the waveform of the experiment is the default, and a condition can decline it
    silent = OperatingCondition("bench", settings={("V1", "dc"): 2.0}, waveform=False)
    e = Experiment(CIRCUIT, conditions=[OperatingCondition("service"), silent], waveform=stored)
    assert e.waveform_for(0) == stored and e.waveform_for(1) is None and e.waveform_points == 10
    assert OperatingCondition.from_metadata(json.loads(json.dumps(silent.metadata()))) == silent
    with_one = conditions[0]
    assert OperatingCondition.from_metadata(json.loads(json.dumps(with_one.metadata()))) == with_one
    other = OperatingCondition("bench", waveform=Waveform("v(out)", fs=10.0, duration=2.0))
    with pytest.raises(ValueError, match="same number of points, not \\[10, 20\\]"):
        Experiment(CIRCUIT, conditions=[with_one, other])
    with pytest.raises(ValueError, match="False for none"):
        OperatingCondition("x", waveform=True)


def test_update_derived_columns_keeps_dataset_verifiable(run, tmp_path):
    import shutil

    copy = tmp_path / "derived"
    shutil.copytree(run.out_dir, copy)
    dataset = Dataset(copy)
    derived = pd.DataFrame(
        {"pass_limit": dataset.samples["vout"].fillna(0.0) < 0.6},
        index=dataset.samples.index,
    )
    dataset.update_columns(derived, note="limits/default.yaml")
    assert dataset.verify() == []
    assert dataset.labels == ["pass_limit"]
    assert dataset.manifest.summary["history"][-1]["columns"] == ["pass_limit"]
    with pytest.raises(ValueError, match="measurement columns"):
        dataset.update_columns(dataset.samples[["vout"]])


def test_splits_keep_fault_replicas_and_magnitudes_together():
    samples = pd.DataFrame(
        {
            "fault_id": ["a", "a", "a", "a", "b", "b", "b", "b"],
            "replica": [0, 0, 1, 1, 0, 0, 1, 1],
            "condition": ["service", "bench"] * 4,
            "fault_magnitude": [1, 1, 2, 2, 1, 1, 2, 2],
        }
    )
    train, test = split_by_replica(samples, 0.5, seed=5)
    assert set(train).isdisjoint(test)
    for fault_id in samples["fault_id"].unique():
        fault_train = train[samples.iloc[train]["fault_id"].to_numpy() == fault_id]
        fault_test = test[samples.iloc[test]["fault_id"].to_numpy() == fault_id]
        assert len(fault_train) == len(fault_test) == 2
        assert samples.iloc[fault_train]["replica"].nunique() == 1
        assert samples.iloc[fault_test]["replica"].nunique() == 1
    train, test = split_by_magnitude(samples, [2.0])
    assert set(samples.iloc[test]["fault_magnitude"]) == {2.0}
    assert set(samples.iloc[train]["fault_magnitude"]) == {1.0}
    assert len(test) == 4


def test_elapsed_time_adds_up_over_interrupted_runs(tmp_path):
    c = campaign(tmp_path / "data")
    e = c.experiment
    # a first run that completed one chunk in 100 s and was then interrupted
    run_chunks(e.plan()[:30], simulate_sample, e, c.out_dir / "parts", 10, chunk=30, progress=False)
    (c.out_dir / "parts" / "elapsed_s").write_text("100.0")
    c.run(chunk=30, progress=False, validate=False)
    manifest = c.dataset().manifest
    assert manifest.resumed is True
    assert manifest.elapsed_total_s >= 100.0 > manifest.elapsed_last_run_s


# --- traceability -----------------------------------------------------------------------


def test_provenance_of_a_sample(run):
    dataset = run.dataset()
    df = dataset.samples
    wanted = (df["fault_id"] == "R1:parametric:+0.2") & (df["condition"] == "low") & df["sim_ok"]
    sample_id = int(df.index[wanted][0])
    provenance = dataset.provenance(sample_id)
    record = json.loads(json.dumps(provenance.to_dict()))
    assert record == {
        "sample_id": sample_id,
        "seed": 5,
        "seeding": "content",
        "seed_key": df.loc[sample_id, "seed_key"],
        "circuit": "divider",
        "netlist_sha256": CIRCUIT.metadata()["netlist_sha256"],
        "fault_id": "R1:parametric:+0.2",
        "fault_type": "parametric",
        "fault_location": "R1",
        "fault_magnitude": 0.2,
        "fault_severity": 0.2,
        "condition": "low",
        "status": "SUCCESS",
        "message": "",
        "simulator": "divider",
        "simulator_version": "1",
        "spicefault_version": dataset.manifest.spicefault_version,
        "parameters": {
            "p_R1_value": df.loc[sample_id, "p_R1_value"],
            "p_R2_value": df.loc[sample_id, "p_R2_value"],
        },
    }
    healthy = dataset.provenance(0)
    assert (healthy.fault_id, healthy.fault_location) == ("healthy", "")
    assert healthy.fault_magnitude is None and healthy.fault_severity is None
    failed = dataset.provenance(int(df.index[df["fault_id"] == "R2:open"][0]))
    assert failed.status == "CONVERGENCE_ERROR" and failed.message == "no convergence"
    assert set(failed.parameters) == {"p_R1_value", "p_R2_value"}  # still traceable


def test_netlist_of_every_sample_is_rebuilt_from_the_stored_values(run):
    dataset, experiment = run.dataset(), run.experiment
    for sample in experiment.plan():
        assert dataset.netlist(sample.sample_id) == experiment.realise(sample).netlist
    shorted = int(np.flatnonzero(dataset.samples["fault_id"] == "R2:short")[0])
    assert "Rpar_R2" in dataset.netlist(shorted)
    assert ".options temp=85.0" in dataset.netlist(1)  # the second condition of the first draw


def test_netlist_of_a_sample_does_not_need_the_measurements_of_its_condition(setups):
    dataset, experiment = setups.dataset(), setups.experiment
    assert dataset.samples.loc[1, "condition"] == "bench"  # measured by a function
    for sample in experiment.plan():
        assert dataset.netlist(sample.sample_id) == experiment.realise(sample).netlist
    assert "dc 2" in dataset.netlist(1) and "dc 1" in dataset.netlist(0)


# --- reproducibility --------------------------------------------------------------------


def test_experiment_is_rebuilt_from_the_dataset(run):
    dataset = run.dataset()
    with pytest.raises(ValueError, match="unknown backend 'divider'"):
        dataset.experiment()  # the backend of these tests is not a built-in one
    rebuilt = dataset.experiment(simulator=SIMULATOR)
    assert rebuilt.metadata() == run.experiment.metadata()
    assert [rebuilt.realise(s).netlist for s in rebuilt.plan()[:10]] == [
        run.experiment.realise(s).netlist for s in run.experiment.plan()[:10]
    ]


def test_reproduce_compares_new_simulations_with_the_stored_ones(run, tmp_path):
    import shutil

    dataset = run.dataset()
    report = dataset.reproduce(n=25, experiment=dataset.experiment(simulator=SIMULATOR))
    assert len(report) == 25 and report["sample_id"].is_monotonic_increasing
    assert report[["definition", "parameters", "status"]].all().all()
    assert report["max_abs_diff"].max() == 0.0 and report["max_rel_diff"].max() == 0.0
    assert report["waveform_abs_diff"].dropna().max() == 0.0
    assert report["waveform_abs_diff"].isna().sum() == (~dataset.ok[report["sample_id"]]).sum()

    # a stored value that the experiment does not give is found
    copy = tmp_path / "copy"
    shutil.copytree(run.out_dir, copy)
    df = pd.read_parquet(copy / "samples.parquet")
    a, b, c = (int(i) for i in np.flatnonzero(dataset.ok)[:3])
    df.loc[a, "vout"] *= 1.01
    df.loc[b, "p_R1_value"] += 1.0
    df.to_parquet(copy / "samples.parquet", index=False)
    altered = Dataset(copy)
    report = altered.reproduce([a, b, c], experiment=altered.experiment(simulator=SIMULATOR))
    assert report["max_rel_diff"].tolist() == pytest.approx([0.01 / 1.01, 0.0, 0.0])
    assert report["parameters"].tolist() == [True, False, True]

    other = campaign(tmp_path / "other", samples_per_fault=6).experiment
    with pytest.raises(ValueError, match="does not have the samples of this dataset"):
        dataset.reproduce(experiment=other)


def test_functions_cannot_be_stored_and_are_passed_again(tmp_path):
    joint = JointVariation("pair", (("R1", "value"), ("R2", "value")), ratio_pair)
    c = campaign(tmp_path / "data", variations=[joint], measurements=[
        Measurement.value("v(out)", name="vout"), Measurement.custom("twice", twice),
    ])  # fmt: skip
    c.run(progress=False)
    dataset = c.dataset()
    assert dataset.labels == ["ratio"] and "ratio" in dataset.samples
    with pytest.raises(NotImplementedError, match="joint variation 'pair' holds a function"):
        dataset.experiment(simulator=SIMULATOR)
    with pytest.raises(NotImplementedError, match="custom measurement 'twice' holds a function"):
        dataset.experiment(simulator=SIMULATOR, variations=[joint])
    rebuilt = dataset.experiment(
        simulator=SIMULATOR, variations=[joint], measurements=c.experiment.measurements
    )
    assert dataset.reproduce(n=10, experiment=rebuilt)["max_abs_diff"].max() == 0.0
    # the netlist of a sample needs no function: the drawn values are in the table
    assert dataset.netlist(7) == c.experiment.realise(c.experiment.plan()[7]).netlist


def ratio_pair(rng, netlist):
    from spicefault.variation import Draw

    ratio = float(rng.uniform(0.9, 1.1))
    return Draw({("R1", "value"): 1e4, ("R2", "value"): 1e4 * ratio}, {"ratio": ratio})


def twice(plot):
    return 2.0 * plot["v(out)"][0]


def test_records_rebuild_variations_measurements_and_conditions():
    variations = VariationSet(
        [
            FixedVariation("R9", 1.0),
            ToleranceVariation("R1", 0.01, "truncnorm"),
            ToleranceVariation("XU1", 1e-3, parameter="vos", relative=False),
            UniformVariation("V1", 0.9, 1.1, "dc"),
            NormalVariation("R2", None, 50.0, truncate=3.0),
            LogNormalVariation("C1", 1e-6, 0.1),
            LogUniformVariation("C2", 2.0),
        ]
    )
    records = json.loads(json.dumps(variations.metadata()))
    assert VariationSet.from_metadata(records).variations == variations.variations
    assert variation_from_metadata(records[4]) == variations.variations[4]

    measurements = [
        Measurement.value("v(out)"),
        Measurement.rms("v(out)", window=(1e-3, 2e-3), analysis=2, name="rms_late"),
        Measurement.magnitude("v(out)", 50.0, db=True),
        Measurement.at_time("v(out)", 1e-3),
    ]
    for m in measurements:
        assert Measurement.from_metadata(json.loads(json.dumps(m.metadata()))) == m
    waveform = Waveform("v(out)", 1e3, 0.5, analysis=1)
    assert Waveform.from_metadata(json.loads(json.dumps(waveform.metadata()))) == waveform
    condition = OperatingCondition("hot", 85.0, {("V1", "dc"): 3.0, ("R1", "value"): 2e3})
    record = json.loads(json.dumps(condition.metadata()))
    assert OperatingCondition.from_metadata(record) == condition
    config = SimulationConfig(("op", "tran 1u 1m"), ("v(out)",), timeout=30.0)
    assert SimulationConfig.from_metadata(json.loads(json.dumps(config.metadata()))) == config

    netlist = CIRCUIT.to_netlist()
    with pytest.raises(ValueError, match="not the one this experiment was defined with"):
        Experiment.from_metadata(Experiment(CIRCUIT).metadata(), netlist.replace("10k", "12k"))


# --- uses -------------------------------------------------------------------------------


def test_to_ml(run):
    dataset = run.dataset()
    n_ok = int(dataset.ok.sum())
    x, y = dataset.to_ml()
    assert x.shape == (n_ok, 2) and y.shape == (n_ok,) and np.isfinite(x).all()
    assert set(y) == {"healthy", "R2:short", "R1:parametric:+0.2", "R2:leakage:100000"}
    x, y = dataset.to_ml(target="fault_location", features=["vout"])
    assert x.shape == (n_ok, 1) and set(y) == {"", "R1", "R2"}
    x, y = dataset.to_ml(target=lambda df: (df["fault_id"] != "healthy").to_numpy(), waveforms=True)
    assert x.shape == (n_ok, 10) and x.dtype == np.float32 and y.dtype == bool
    x, y = dataset.to_ml(drop_failed=False)
    assert len(x) == len(y) == 80 and np.isnan(x).any() and "R2:open" in set(y)
    with pytest.raises(ValueError, match="no waveforms"):
        campaign_without_waveforms = Dataset.__new__(Dataset)
        campaign_without_waveforms.waveforms = None
        campaign_without_waveforms.samples = dataset.samples
        campaign_without_waveforms.to_ml(waveforms=True)


def test_analysis_of_the_dataset(run):
    analysis = run.dataset().analysis(features=["vout"], healthy_split=None)
    assert analysis.features == ["vout"] and len(analysis.fault_ids) == 4
    counts = analysis.counts()
    assert counts.loc["R2:open"].to_dict() == {"n": 10, "n_ok": 0, "n_failed": 10}
    assert analysis.response("R2:short").record["fault_type"] == "short_circuit"


@pytest.mark.ngspice
def test_dataset_of_a_real_campaign_reproduces_itself(tmp_path):
    """With ngspice and standard definitions, the folder alone is enough to simulate again."""
    circuit = Circuit(
        "rc\nV1 in 0 dc 1 ac 1 pulse(0 1 1m 1u 1u 5m 20m)\nR1 in out 10k\nC1 out 0 100n\n"
        "R2 out 0 10k\n.end\n",
        "rc",
    )
    FaultCampaign(
        circuit,
        [OpenCircuit("R2"), ParametricFault("C1", deviation=0.5), LeakageFault("C1", 1e5)],
        out_dir=tmp_path / "data",
        samples_per_fault=4,
        variations=tolerances(circuit, {"R": 0.01, "C": 0.05}, "truncnorm"),
        conditions=[OperatingCondition(), OperatingCondition("low", settings={("V1", "dc"): 0.5})],
        config=SimulationConfig(("op", "ac dec 20 1 1e5", "tran 10u 10m"), outputs=("v(out)",)),
        measurements=[
            Measurement.value("v(out)", name="dc"),
            Measurement.magnitude("v(out)", 1e3, name="gain_1k"),
            Measurement.rms("v(out)", window=(2e-3, 6e-3), name="rms"),
        ],
        waveform=Waveform("v(out)", 10e3, 10e-3),
        seed=9,
    ).run(workers=2, progress=False)

    dataset = Dataset(tmp_path / "data")  # nothing but the folder from here on
    assert dataset.verify() == [] and dataset.ok.all() and len(dataset) == 32
    assert dataset.provenance(5).simulator_version.startswith("ngspice-")
    report = dataset.reproduce(n=12)
    assert report[["definition", "parameters", "status"]].all().all()
    assert report["max_abs_diff"].max() == 0.0 and report["waveform_abs_diff"].max() == 0.0

    # the rebuilt netlist of a sample gives its stored measurements
    result = Simulator().run(dataset.netlist(17), dataset.experiment().config)
    stored = dataset.samples.loc[17, dataset.features]
    assert [m(result) for m in dataset.experiment().measurements] == stored.tolist()
