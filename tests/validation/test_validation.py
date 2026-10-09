"""The validation studies: their circuits behave as designed and their campaigns run."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repository root

import validation  # noqa: E402
from spicefault import Simulator  # noqa: E402

pytestmark = pytest.mark.ngspice

N_FAULTS = {"sallen_key": 58, "biquad": 104, "regulator": 54}


@pytest.fixture(scope="module", params=sorted(validation.STUDIES))
def study(request):
    return validation.get(request.param)


@pytest.fixture(scope="module")
def nominal(study):
    result = Simulator().run(study.circuit.to_netlist(), study.config)
    assert result.ok, result.message
    return result, {m.name: m(result) for m in study.measurements}


def test_definitions_fit_the_circuit(study):
    universe = study.universe()
    assert len(universe) == N_FAULTS[study.name] and universe.coverage() == 1.0
    campaign = study.campaign("unused", samples_per_fault=1, healthy_samples=1)
    report = campaign.validate()
    assert report.ok, str(report)
    assert set(study.specifications) <= set(study.features) == set(study.noise)
    campaign.experiment.check_picklable()


def test_nominal_circuit_meets_its_specification(study, nominal):
    _, values = nominal
    for name, (low, high) in study.specifications.items():
        assert low < values[name] < high, name


def test_sallen_key_matches_the_ideal_transfer_function():
    study = validation.get("sallen_key")
    result = Simulator().run(study.circuit.to_netlist(), study.config)
    f = result.plot("ac")["frequency"].real
    r1, r2, r3, r4, r5, c1, c2 = 5.18e3, 1e3, 2e3, 4e3, 4e3, 5e-9, 5e-9
    s = 2j * np.pi * f
    gain = 1 + r5 / r4
    # node a: (vin - va) / r1 + (vo - va) / r2 = va s c2 + (va - vb) s c1
    # node b: (va - vb) s c1 = vb / r3, and vo = gain * vb
    va_over_vb = 1 + 1 / (s * c1 * r3)
    admittance = 1 / r1 + 1 / r2 + s * c1 + s * c2
    ideal = gain / r1 / (va_over_vb * admittance - s * c1 - gain / r2)
    simulated = result.plot("ac")["v(out)"]
    band = f < 100e3  # above, the finite bandwidth of the amplifier shows
    assert np.allclose(np.abs(simulated[band]), np.abs(ideal[band]), rtol=0.03)
    centre = np.sqrt((r1 + r2) / (r1 * r2 * r3 * c1 * c2)) / (2 * np.pi)
    assert {m.name: m(result) for m in study.measurements}["centre_frequency"] == pytest.approx(
        centre, rel=0.01
    )


def test_biquad_matches_the_ideal_transfer_function():
    study = validation.get("biquad")
    result = Simulator().run(study.circuit.to_netlist(), study.config)
    f = result.plot("ac")["frequency"].real
    r1, r2, r4, c1, c2 = 6.2e3, 6.2e3, 1.6e3, 5e-9, 5e-9  # with r3 = r2 and r5 = r6
    tau = np.sqrt(r2 * r4 * c1 * c2)  # a cut-off of 10.1 kHz
    q = r1 * c1 / tau  # 1.97
    s = 2j * np.pi * f
    ideal = (s * tau) ** 2 / ((s * tau) ** 2 + s * tau / q + 1)
    band = f < 50e3
    assert np.allclose(np.abs(result.plot("ac")["v(out)"][band]), np.abs(ideal[band]), rtol=0.02)


def test_regulator_follows_its_reference_and_temperature():
    study = validation.get("regulator")
    by_condition = {}
    for condition in study.conditions["one_factor"]:
        netlist = study.circuit.netlist()
        condition.apply(netlist)
        result = Simulator().run(str(netlist), study.config)
        assert result.ok, condition.name
        by_condition[condition.name] = {m.name: m(result) for m in study.measurements}
    nominal = by_condition["nominal"]
    # output = (reference + base-emitter voltage) * (1 + R3 / R4)
    base_emitter = nominal["divider_voltage"] - nominal["reference_voltage"]
    assert 0.6 < base_emitter < 0.75
    expected = nominal["divider_voltage"] * (1 + 1 / 1.2)
    assert nominal["output_voltage"] == pytest.approx(expected, rel=0.01)
    out = {name: values["output_voltage"] for name, values in by_condition.items()}
    assert out["cold"] > out["nominal"] > out["hot"] and out["cold"] - out["hot"] > 0.2
    # the reference drifts within the range of the data sheet, -3.5 to +0.2 mV/K
    reference = {name: values["reference_voltage"] for name, values in by_condition.items()}
    assert -3.5e-3 < (reference["hot"] - reference["cold"]) / 105.0 < 0.2e-3
    assert out["low_line"] < out["nominal"] < out["high_line"]
    assert out["heavy_load"] < out["nominal"] < out["light_load"]
    assert len(study.conditions["corners"]) == 9


def test_a_small_campaign_runs_and_can_be_analysed(study, tmp_path):
    campaign = study.campaign(tmp_path / "data", samples_per_fault=1, healthy_samples=6)
    campaign.run(workers=2, progress=False)
    dataset = campaign.dataset()
    assert dataset.verify() == [] and len(dataset) == N_FAULTS[study.name] + 6
    samples = dataset.samples
    healthy = samples[samples["fault_id"] == "healthy"]
    assert (healthy["status"] == "SUCCESS").all()
    assert set(samples["status"]) <= {"SUCCESS", "CONVERGENCE_ERROR", "INVALID_OUTPUT", "FAILED"}
    assert samples["sim_ok"].mean() > 0.7

    ok = samples[samples["sim_ok"]]
    compliant = study.compliant(ok)
    assert compliant[ok["fault_id"] == "healthy"].mean() >= 0.5
    assert not compliant[ok["fault_type"] == "short_circuit"].all()  # some shorts break it

    noisy = study.observed(ok, seed=1)
    assert noisy[study.features].notna().all().all()
    shuffled = study.observed(ok.iloc[::-1], seed=1).iloc[::-1]
    assert np.array_equal(noisy[study.features], shuffled[study.features])  # by sample, not by row
    difference = (noisy[study.features] - ok[study.features]).abs()
    assert (difference > 0).all().all()
    scale = ok.loc[ok["fault_id"] == "healthy", study.features].abs().median()
    assert (difference.median() / scale).max() < 0.05  # small against the measurements


def test_tolerance_scale_shrinks_the_same_circuits(study):
    full = study.experiment(samples_per_fault=1, healthy_samples=2)
    none = study.experiment(samples_per_fault=1, healthy_samples=2, tolerance_scale=0.0)
    sample = full.plan()[0]
    nominal = study.circuit.parameters()
    a, b = full.realise(sample).parameters, none.realise(sample).parameters
    assert all(b[key] == pytest.approx(nominal[key], rel=1e-12) for key in b)
    assert any(a[key] != nominal[key] for key in a)
    assert full.seed_key(sample) == none.seed_key(sample)
