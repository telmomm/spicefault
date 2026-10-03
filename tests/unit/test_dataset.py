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
)
from spicefault.dataset import Manifest
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
