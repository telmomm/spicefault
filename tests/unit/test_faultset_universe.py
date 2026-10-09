import json

import numpy as np
import pytest

from spicefault import Circuit, Experiment
from spicefault.faults import (
    FaultRule,
    FaultSet,
    FaultUniverse,
    LeakageFault,
    OpenCircuit,
    ParametricFault,
    ShortCircuit,
    leakage_rule,
    open_rule,
    parametric_rule,
    series_rule,
    short_rule,
)
from spicefault.variation import ToleranceVariation, VariationSet

CIRCUIT = Circuit(
    "filter\nV1 in 0 dc 1 ac 1\nR1 in out 10k\nC1 out 0 100n\nR2 out 0 10k\n"
    "XU1 out o o opamp vos=0.0 aol=200000.0\n.end\n",
    "filter",
)
FAULTS = FaultSet(
    [
        OpenCircuit("R1", tags={"stage": "input"}),
        ShortCircuit("C1"),
        ParametricFault("R1", deviation=0.05, tags={"stage": "input"}),
        ParametricFault("C1", deviation=0.05),
        LeakageFault("C1", 1e6),
    ]
)


def test_fault_set_is_an_ordered_sequence_with_unique_identifiers():
    assert len(FAULTS) == 5 and FAULTS[1] == ShortCircuit("C1")
    assert FAULTS.ids() == [
        "R1:open", "C1:short", "R1:parametric:+0.05", "C1:parametric:+0.05", "C1:leakage:1e+06",
    ]
    assert FAULTS["C1:short"] is FAULTS[1] and FAULTS[:2] == FaultSet(list(FAULTS)[:2])
    assert "R1:open" in FAULTS and ShortCircuit("C1") in FAULTS and "R9:open" not in FAULTS
    assert len(FAULTS + [OpenCircuit("R2")]) == 6
    with pytest.raises(ValueError, match="repeated fault identifiers: \\['R1:open'\\]"):
        FAULTS + [OpenCircuit("R1")]
    with pytest.raises(KeyError):
        FAULTS["R9:open"]


def test_filter_and_select():
    assert FAULTS.filter(component="c1").ids() == [
        "C1:short", "C1:parametric:+0.05", "C1:leakage:1e+06",
    ]
    assert FAULTS.filter(fault_type="parametric", component="R1").ids() == ["R1:parametric:+0.05"]
    assert FAULTS.filter(tags={"stage": "input"}).ids() == ["R1:open", "R1:parametric:+0.05"]
    assert FAULTS.filter(where=lambda f: f.magnitude is None).ids() == ["R1:open", "C1:short"]
    assert FAULTS.select(["C1:short", "R1:open"]).ids() == ["C1:short", "R1:open"]


def test_fault_set_round_trip(tmp_path):
    assert FaultSet.from_metadata(json.loads(json.dumps(FAULTS.metadata()))) == FAULTS
    FAULTS.to_json(tmp_path / "faults.json")
    loaded = FaultSet.from_json(tmp_path / "faults.json")
    assert loaded == FAULTS and [type(f) for f in loaded] == [type(f) for f in FAULTS]


def test_an_experiment_takes_a_fault_set():
    experiment = Experiment(CIRCUIT, faults=FAULTS, samples=2)
    assert len(experiment.plan()) == 2 * 6
    assert [f["fault_id"] for f in experiment.metadata()["faults"]] == FAULTS.ids()


# --- fault and normal variation -----------------------------------------------------


def test_tolerance_overlap_shows_faults_inside_the_healthy_band():
    variations = VariationSet(
        [ToleranceVariation("R1", 0.01), ToleranceVariation("C1", 0.05),
         ToleranceVariation("XU1", 0.5, parameter="aol")]
    )  # fmt: skip
    faults = FAULTS + [
        ParametricFault("C1", deviation=-0.2),
        ParametricFault("XU1", "aol", factor=0.01),
        ParametricFault("R2", deviation=0.01),  # no tolerance declared: not reported
        ParametricFault("R1", fault_value=10_050.0),
    ]
    report = faults.tolerance_overlap(variations, CIRCUIT).set_index("fault_id")
    assert list(report.index) == [
        "R1:parametric:+0.05", "C1:parametric:+0.05", "C1:parametric:-0.2",
        "XU1.aol:parametric:x0.01", "R1:parametric:=10050",
    ]
    # a 5 % fault on a 5 % capacitor: half of the faulty population is within tolerance
    row = report.loc["C1:parametric:+0.05"]
    assert (row["deviation_min"], row["deviation_max"]) == pytest.approx((-0.0025, 0.1025))
    assert row["inside_fraction"] == pytest.approx(0.5)
    # the same deviation on a 1 % resistor is always outside
    assert report.loc["R1:parametric:+0.05", "inside_fraction"] == 0.0
    assert report.loc["C1:parametric:-0.2", "inside_fraction"] == 0.0
    assert report.loc["XU1.aol:parametric:x0.01", "inside_fraction"] == 0.0
    assert report.loc["R1:parametric:=10050", "inside_fraction"] == 1.0  # +0.5 %, within 1 %
    # absolute faults need the circuit for their nominal value
    assert "R1:parametric:=10050" not in set(faults.tolerance_overlap(variations)["fault_id"])


@pytest.mark.parametrize("distribution", ["uniform", "truncnorm"])
@pytest.mark.parametrize("deviation", [0.02, 0.05, -0.08])
def test_inside_fraction_matches_the_drawn_population(distribution, deviation):
    from spicefault.experiments import sample_stream

    variation = ToleranceVariation("C1", 0.05, distribution)
    fault = ParametricFault("C1", deviation=deviation)
    predicted = FaultSet([fault]).tolerance_overlap(VariationSet([variation]))["inside_fraction"][0]
    rng = sample_stream(0)
    drawn = np.array([variation.sample(rng, 1.0) for _ in range(20000)]) * (1.0 + deviation)
    assert np.mean(np.abs(drawn - 1.0) <= 0.05) == pytest.approx(predicted, abs=0.01)


# --- universe and coverage ----------------------------------------------------------

RULES = [
    open_rule(),
    short_rule(),
    parametric_rule([-0.2, 0.2]),
    leakage_rule([1e5, 1e7]),
    FaultRule(
        "offset",
        lambda c: [ParametricFault(c.name, "vos", fault_value=v) for v in (0.02, 0.05)],
        kinds="X",
    ),
]


def test_universe_is_generated_from_the_rules():
    universe = FaultUniverse(CIRCUIT, RULES)
    assert len(universe) == 3 * 4 + 2 + 2 and universe.coverage() == 1.0
    assert universe.faults.ids()[:6] == [
        "R1:open", "R1:short", "R1:parametric:-0.2", "R1:parametric:+0.2",
        "C1:open", "C1:short",
    ]
    assert universe.selected() == universe.faults
    matrix = universe.coverage_matrix()
    assert list(matrix.index) == ["R1", "C1", "R2", "XU1"]  # the source has no fault
    assert list(matrix.columns) == ["open", "short", "parametric", "leakage", "offset"]
    assert matrix.loc["C1"].tolist() == [1, 1, 2, 2] + [pytest.approx(float("nan"), nan_ok=True)]
    assert np.isnan(matrix.loc["R1", "leakage"]) and matrix.loc["XU1", "offset"] == 2
    assert universe.faults["C1:leakage:100000"].severity.value == 1.0
    assert universe.metadata()["components_without_faults"] == ["V1"]
    with pytest.raises(ValueError, match="repeated rule names"):
        FaultUniverse(CIRCUIT, [open_rule(), open_rule("C")])


def test_rules_can_be_restricted_to_named_components():
    universe = FaultUniverse(CIRCUIT, [open_rule(components=("r2", "C1")), short_rule("C")])
    assert universe.faults.ids() == ["C1:open", "C1:short", "R2:open"]


AMPLIFIERS = Circuit(
    "front end\nV1 in 0 dc 1\nR1 in a 10k\n"
    "XU1 a b out1 opamp vos=0.0 aol=200000.0\nXU2 out1 c out opamp vos=0.0 aol=200000.0\n"
    "XB1 out 0 buffer gain=1.0\nD1 out 0 1N4148\nQ1 c b 0 2N2222 2\nQ2 c b 0 sub BC547\n"
    "M1 d g 0 0 nch W=1u L=1u\n.end\n",
    "front_end",
)


def test_components_know_their_subcircuit_or_model():
    models = {c.name: c.model for c in AMPLIFIERS.components()}
    assert models == {
        "V1": "", "R1": "", "XU1": "opamp", "XU2": "opamp", "XB1": "buffer", "D1": "1N4148",
        "Q1": "2N2222", "Q2": "BC547", "M1": "nch",
    }  # fmt: skip


def test_rules_select_by_model_tag_their_faults_and_take_absolute_values_or_factors():
    rules = [
        parametric_rule(
            kinds="X", parameter="vos", models=("OpAmp",), fault_values=(0.02, -0.02),
            fault_type="offset", tags=lambda c: {"part": c.name, "origin": "amplifier"},
        ),
        parametric_rule(kinds="X", parameter="aol", models=("opamp",), factors=(0.01,),
                        fault_type="gain_loss", tags={"origin": "amplifier"}),
        parametric_rule([0.2], "R", fault_values=(47e3,)),
        open_rule("R", tags={"origin": "assembly"}),
    ]  # fmt: skip
    universe = FaultUniverse(AMPLIFIERS, rules)
    assert universe.faults.ids() == [
        "R1:parametric:+0.2", "R1:parametric:=47000", "R1:open",
        "XU1.vos:parametric:=0.02", "XU1.vos:parametric:=-0.02", "XU1.aol:parametric:x0.01",
        "XU2.vos:parametric:=0.02", "XU2.vos:parametric:=-0.02", "XU2.aol:parametric:x0.01",
    ]  # the buffer is an X instance of another subcircuit
    matrix = universe.coverage_matrix()
    assert list(matrix.columns) == ["offset", "gain_loss", "parametric", "open"]
    assert matrix.loc["XU2"].tolist()[:2] == [2, 1] and "XB1" not in matrix.index

    offset = universe.faults["XU2.vos:parametric:=0.02"]
    assert offset.fault_type == "offset" and offset.tags == {"part": "XU2", "origin": "amplifier"}
    assert offset.severity is None  # nominally zero: no reference to measure it against
    gain = universe.faults["XU1.aol:parametric:x0.01"]
    assert gain.fault_type == "gain_loss" and gain.tags == {"origin": "amplifier"}
    assert gain.severity.value == pytest.approx(0.99)
    assert universe.faults["R1:open"].tags == {"origin": "assembly"}
    assert universe.faults["R1:parametric:+0.2"].tags == {}
    # an absolute value is measured against the nominal one: |47k - 10k| / 10k
    assert universe.faults["R1:parametric:=47000"].severity.value == pytest.approx(3.7)

    netlist = AMPLIFIERS.netlist()
    offset.apply(netlist)
    gain.apply(netlist)
    assert netlist.value("XU2", "vos") == 0.02 and netlist.value("XU1", "aol") == 2000.0
    # a tagged fault is the same fault, and is rebuilt from its record with its tags
    rebuilt = FaultSet.from_metadata(json.loads(json.dumps(universe.faults.metadata())))
    assert rebuilt == universe.faults and isinstance(rebuilt[3], ParametricFault)
    plain = ParametricFault("R1", deviation=0.2)
    assert plain.with_tags({"a": "b"}).tags == {"a": "b"} and plain.tags == {}
    with pytest.raises(ValueError, match="needs deviations, factors or fault_values"):
        parametric_rule()


def test_series_resistance_rule():
    universe = FaultUniverse(CIRCUIT, [series_rule([1.0, 100.0], kinds="C")])
    assert [fault.fault_id for fault in universe.faults] == [
        "C1:series_resistance:1", "C1:series_resistance:100",
    ]
    assert [fault.severity.value for fault in universe.faults] == [0.0, 1.0]


def test_every_excluded_fault_carries_a_reason():
    universe = FaultUniverse(CIRCUIT, RULES)
    assert universe.exclude("R2:short", "shorts the output to ground: same as C1:short") == 1
    assert universe.exclude(lambda f: f.fault_type == "leakage", "no leakage model validated") == 2
    assert universe.exclude(["R2:short"], "repeated") == 0  # the first reason is kept
    with pytest.raises(KeyError):
        universe.exclude("R9:open", "does not exist")
    with pytest.raises(ValueError, match="needs a reason"):
        universe.exclude("R1:open", "")

    assert len(universe.selected()) == 13 and "R2:short" not in universe.selected()
    assert universe.coverage() == pytest.approx(13 / 16)
    assert universe.exclusions().to_dict("records") == [
        {"fault_id": "R2:short", "reason": "shorts the output to ground: same as C1:short"},
        {"fault_id": "C1:leakage:100000", "reason": "no leakage model validated"},
        {"fault_id": "C1:leakage:1e+07", "reason": "no leakage model validated"},
    ]
    assert universe.coverage_matrix().loc["C1", "leakage"] == 0
    assert universe.coverage_matrix("universe").loc["C1", "leakage"] == 2
    assert universe.coverage_matrix("fraction").loc["R2"].tolist()[:3] == [1.0, 0.0, 1.0]
    with pytest.raises(ValueError, match="unknown counts"):
        universe.coverage_matrix("ratio")

    record = json.loads(json.dumps(universe.metadata()))
    assert (record["n_universe"], record["n_selected"]) == (16, 13)
    assert len(record["exclusions"]) == 3 and len(record["cells"]) == 3 + 4 + 3 + 1
    listed = {i for cell in record["cells"] for i in cell["fault_ids"]}
    assert listed == set(universe.faults.ids())  # every fault is in exactly one cell
