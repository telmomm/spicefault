import json

import numpy as np
import pytest

from spicefault import Circuit, Experiment, VariationSet
from spicefault.experiments import sample_stream
from spicefault.netlist import Netlist
from spicefault.variation import (
    CustomVariation,
    Draw,
    FixedVariation,
    JointVariation,
    LogNormalVariation,
    LogUniformVariation,
    NormalVariation,
    ToleranceVariation,
    UniformVariation,
    tolerances,
)

TEXT = (
    "t\nV1 in 0 dc 1\nR1 in out 10k\nR2 out 0 20k\nC1 out 0 1e-06\n"
    "XU1 out o opamp vos=0.0 aol=200000.0 gain={g}\n.end\n"
)
N = 20000


def draws(variation, n=N, seed=0) -> np.ndarray:
    rng, net = sample_stream(seed), Netlist(TEXT)
    return np.array([variation.draw(rng, net).values[variation.targets()[0]] for _ in range(n)])


# --- each distribution --------------------------------------------------------------


def test_fixed():
    assert set(draws(FixedVariation("R1", 12e3), 10)) == {12e3}


def test_tolerance_relative_and_absolute():
    relative = draws(ToleranceVariation("R1", 0.05))
    assert relative.min() >= 9500 and relative.max() <= 10500
    assert relative.mean() == pytest.approx(1e4, rel=2e-3)
    assert relative.std() == pytest.approx(500 / 3**0.5, rel=0.02)  # uniform
    normal = draws(ToleranceVariation("R1", 0.05, "truncnorm"))
    assert np.abs(normal - 1e4).max() <= 500 and normal.std() == pytest.approx(500 / 3, rel=0.03)
    # an offset is nominally zero: its tolerance is in volts, not relative
    offset = draws(ToleranceVariation("XU1", 0.5e-3, parameter="vos", relative=False))
    assert np.abs(offset).max() <= 0.5e-3 and np.abs(offset).max() > 0.49e-3
    assert abs(offset.mean()) < 1e-5


def test_uniform():
    values = draws(UniformVariation("R1", 9e3, 12e3))
    assert values.min() >= 9e3 and values.max() <= 12e3
    assert values.mean() == pytest.approx(10.5e3, rel=2e-3)
    with pytest.raises(ValueError, match="low <= high"):
        UniformVariation("R1", 2.0, 1.0)


def test_normal():
    values = draws(NormalVariation("R1", 10000, 500))
    assert values.mean() == pytest.approx(1e4, abs=15)
    assert values.std() == pytest.approx(500, rel=0.02)
    assert (np.abs(values - 1e4) > 1500).mean() == pytest.approx(0.0027, abs=0.0015)
    # mean taken from the netlist, and truncation at 2 sigma
    r2 = draws(NormalVariation("R2", None, 1000, truncate=2.0))
    assert r2.mean() == pytest.approx(2e4, abs=30) and np.abs(r2 - 2e4).max() <= 2000
    assert r2.std() == pytest.approx(1000 * 0.8796, rel=0.02)  # sd of a normal cut at 2 sigma
    with pytest.raises(ValueError):
        NormalVariation("R1", 1e4, -1.0)


def test_lognormal_is_positive_with_a_multiplicative_spread():
    values = draws(LogNormalVariation("R1", None, 0.5))
    assert values.min() > 0
    assert np.median(values) == pytest.approx(1e4, rel=0.02)
    assert np.log(values).std() == pytest.approx(0.5, rel=0.02)
    assert np.median(draws(LogNormalVariation("R1", 3e3, 0.1))) == pytest.approx(3e3, rel=0.01)


def test_loguniform():
    values = draws(LogUniformVariation("C1", 2.0))
    assert values.min() >= 0.5e-6 and values.max() <= 2e-6
    assert np.exp(np.log(values).mean()) == pytest.approx(1e-6, rel=0.01)
    with pytest.raises(ValueError, match="at least 1"):
        LogUniformVariation("C1", 0.5)


def test_custom():
    either = CustomVariation("R1", lambda rng, nominal: nominal * rng.choice([0.5, 2.0]))
    values = draws(either, 200)
    assert set(values) == {5e3, 2e4}
    # the nominal value of a parameter given as an expression is not a number
    seen = draws(CustomVariation("XU1", lambda rng, nominal: float(np.isnan(nominal)), "gain"), 1)
    assert seen[0] == 1.0


# --- joint variations ---------------------------------------------------------------


def part(rng, netlist):
    """A part of a type drawn first, whose parameters depend on the type."""
    kind = str(rng.choice(["film", "ceramic"]))
    spread = {"film": 0.01, "ceramic": 0.2}[kind]
    c1 = netlist.value("C1") * (1.0 + spread * rng.uniform(-1, 1))
    return Draw({("C1", "value"): c1, ("R2", "value"): 2e4}, {"c1_kind": kind})


JOINT = JointVariation("capacitor", (("C1", "value"), ("R2", "value")), part, "type, then value")


def test_joint_variation_draws_dependent_parameters_and_labels():
    drawn = [JOINT.draw(sample_stream(0, r), Netlist(TEXT)) for r in range(400)]
    spread = {"film": 0.0, "ceramic": 0.0}
    for d in drawn:
        kind = d.labels["c1_kind"]
        spread[kind] = max(spread[kind], abs(d.values["C1", "value"] / 1e-6 - 1))
    assert spread["film"] <= 0.01 < 0.15 < spread["ceramic"] <= 0.2
    plain = JointVariation("pair", (("R1", "value"),), lambda rng, net: {("R1", "value"): 1.0})
    assert plain.draw(sample_stream(0), Netlist(TEXT)) == Draw({("R1", "value"): 1.0})
    wrong = JointVariation("pair", (("R1", "value"),), lambda rng, net: {("R2", "value"): 1.0})
    with pytest.raises(ValueError, match="must return its declared parameters"):
        wrong.draw(sample_stream(0), Netlist(TEXT))


def test_labels_reach_the_experiment_record():
    circuit = Circuit(TEXT.replace(" gain={g}", ""), "rc")
    experiment = Experiment(circuit, variations=[ToleranceVariation("R1", 0.01), JOINT], samples=3)
    realised = [experiment.realise(s) for s in experiment.plan()]
    assert {r.labels["c1_kind"] for r in realised} <= {"film", "ceramic"}
    assert set(realised[0].parameters) == {("R1", "value"), ("C1", "value"), ("R2", "value")}
    assert f"C1 out 0 {realised[0].parameters['C1', 'value']!r}" in realised[0].netlist


# --- sets ---------------------------------------------------------------------------

EVERY_TYPE = VariationSet(
    [
        ToleranceVariation("R1", 0.05),
        NormalVariation("R2", None, 1000, truncate=3.0),
        LogNormalVariation("C1", None, 0.1),
        UniformVariation("V1", 0.9, 1.1, parameter="dc"),
        ToleranceVariation("XU1", 0.5e-3, "truncnorm", "vos", relative=False),
        LogUniformVariation("XU1", 2.0, parameter="aol"),
    ]
)


def test_a_parameter_has_at_most_one_variation():
    with pytest.raises(ValueError, match="C1.value has more than one variation"):
        VariationSet([ToleranceVariation("C1", 0.05), JOINT])
    assert len(EVERY_TYPE + [FixedVariation("XU1", 1.0, "gain")]) == 7


def test_metadata_is_json_and_names_every_distribution():
    other = JointVariation("capacitor", (("C9", "value"),), part, "type, then value")
    record = json.loads(json.dumps((EVERY_TYPE + [other, FixedVariation("R9", 1.0)]).metadata()))
    assert [r["type"] for r in record] == [
        "tolerance", "normal", "lognormal", "uniform", "tolerance", "loguniform", "joint", "fixed",
    ]
    assert record[1] == {
        "type": "normal", "component": "R2", "parameter": "value", "mean": None, "sigma": 1000,
        "truncate": 3.0,
    }
    assert record[6]["sampler"] == "part" and record[6]["description"] == "type, then value"
    assert record[6]["parameters"] == [{"component": "C9", "parameter": "value"}]


def test_scaling_the_spread_keeps_the_draws_aligned():
    """Scaled sets use the same random numbers: each deviation is multiplied, not redrawn."""
    net = Netlist(TEXT)
    nominal = {("R1", "value"): 1e4, ("R2", "value"): 2e4, ("V1", "dc"): 1.0, ("XU1", "vos"): 0.0}
    for seed in range(20):
        full = EVERY_TYPE.sample(sample_stream(seed), net).values
        half = EVERY_TYPE.scaled(0.5).sample(sample_stream(seed), net).values
        none = EVERY_TYPE.scaled(0.0).sample(sample_stream(seed), net).values
        for key, centre in nominal.items():
            tiny = 1e-9 * abs(centre) + 1e-15
            assert half[key] - centre == pytest.approx(0.5 * (full[key] - centre), abs=tiny)
            assert none[key] == pytest.approx(centre, abs=tiny)
        for key, centre in ((("C1", "value"), 1e-6), (("XU1", "aol"), 2e5)):  # multiplicative
            assert np.log(half[key] / centre) == pytest.approx(0.5 * np.log(full[key] / centre))
            assert none[key] == pytest.approx(centre)


def test_custom_variations_cannot_be_scaled():
    mixed = VariationSet([ToleranceVariation("R1", 0.05), JOINT])
    with pytest.raises(NotImplementedError, match="joint variation 'capacitor'"):
        mixed.scaled(0.5)
    kept = mixed.scaled(0.5, strict=False)
    assert kept.variations[-1] is JOINT and kept.variations[0].tolerance == 0.025
    with pytest.raises(NotImplementedError):
        CustomVariation("R1", lambda rng, nominal: nominal).scaled(2.0)


def test_tolerances_by_component_kind():
    circuit = Circuit(TEXT.replace(" gain={g}", ""), "rc")
    built = tolerances(circuit, {"R": 0.01, "C": 0.05}, "truncnorm", overrides={"r2": 0.001})
    assert [(v.component, v.tolerance, v.distribution) for v in built] == [
        ("R1", 0.01, "truncnorm"), ("R2", 0.001, "truncnorm"), ("C1", 0.05, "truncnorm"),
    ]
    assert [v.component for v in tolerances(circuit, {"R": 0.01}, components=["R2"])] == ["R2"]
    assert len(tolerances(circuit, {"X": 0.1})) == 0  # an instance has no value
