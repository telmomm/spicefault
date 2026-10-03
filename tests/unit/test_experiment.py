import json

import numpy as np
import pytest

from spicefault import (
    Circuit,
    Experiment,
    Fault,
    OperatingCondition,
    SimulationConfig,
    SimulationResult,
    SimulationStatus,
    Simulator,
)
from spicefault.faults import InsertSeries, SetParameter
from spicefault.variation import ToleranceVariation

# R1 has a temperature coefficient of 1 %/K: its value doubles 100 K above 27 degC
CIRCUIT = Circuit(
    "divider\nV1 in 0 dc 1\nR1 in out 10k tc1=0.01\nR2 out 0 10k\n.end\n", "divider"
)
OPEN = Fault("open", [InsertSeries("R2", 2, 1e9)], model_parameters={"r_open": 1e9})
DRIFT = Fault("parametric", [SetParameter("R1", "value", "relative", 0.2)], magnitude=0.2)
CONDITIONS = [
    OperatingCondition(),
    OperatingCondition("hot", temperature=127.0),
    OperatingCondition("low", settings={("V1", "dc"): 0.5}),
]


def experiment(**kwargs):
    defaults = {
        "circuit": CIRCUIT,
        "config": SimulationConfig(analyses=("op",), outputs=("v(out)",)),
        "faults": [OPEN, DRIFT],
        "variations": [ToleranceVariation("R1", 0.01), ToleranceVariation("R2", 0.01)],
        "conditions": CONDITIONS,
        "samples": 2,
        "healthy_samples": 3,
        "seed": 42,
    }
    return Experiment(**{**defaults, **kwargs})


def vout(sample_result) -> float:
    return float(sample_result.result.plot("op")["v(out)"][0])


def test_plan_is_fixed_and_healthy_comes_first():
    plan = experiment().plan()
    assert len(plan) == (3 + 2 + 2) * 3
    assert [s.sample_id for s in plan] == list(range(len(plan)))
    assert [(s.fault_index, s.replica, s.condition_index) for s in plan[:4]] == [
        (0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0),
    ]
    assert plan[-1].fault_index == 2 and plan == experiment().plan()
    assert len(experiment(healthy_samples=None).plan()) == (2 + 2 + 2) * 3


def test_realisation_draws_then_injects_then_sets_the_condition():
    e = experiment()
    sample = next(s for s in e.plan() if (s.fault_index, s.replica, s.condition_index) == (2, 1, 1))
    realised = e.realise(sample)
    r1, r2 = realised.parameters[("R1", "value")], realised.parameters[("R2", "value")]
    assert abs(r1 / 1e4 - 1) <= 0.01 and abs(r2 / 1e4 - 1) <= 0.01
    # the fault compounds with the drawn value, and the other component keeps its spread
    assert realised.netlist == (
        f"divider\nV1 in 0 dc 1\nR1 in out {r1 * (1.0 + 0.2)!r} tc1=0.01\n"
        f"R2 out 0 {r2!r}\n.options temp=127.0\n.end\n"
    )
    assert e.realise(sample) == realised
    assert CIRCUIT.to_netlist().count("10k") == 2


def test_one_drawn_circuit_is_seen_under_every_condition():
    e = experiment()
    by_key = {(s.fault_index, s.replica, s.condition_index): e.realise(s) for s in e.plan()}
    assert by_key[1, 0, 0].parameters == by_key[1, 0, 1].parameters == by_key[1, 0, 2].parameters
    assert by_key[1, 0, 0].parameters != by_key[1, 1, 0].parameters  # another replica
    assert by_key[1, 0, 0].parameters != by_key[2, 0, 0].parameters  # another fault
    assert experiment(seed=43).realise(e.plan()[0]).parameters != by_key[0, 0, 0].parameters


def test_definitions_are_checked():
    with pytest.raises(ValueError, match="repeated fault"):
        experiment(faults=[OPEN, OPEN])
    with pytest.raises(ValueError, match="repeated operating condition"):
        experiment(conditions=[OperatingCondition(), OperatingCondition()])
    with pytest.raises(ValueError, match="at least one operating condition"):
        experiment(conditions=[])
    assert len(experiment(conditions=OperatingCondition()).conditions) == 1


def test_metadata_describes_the_whole_experiment():
    record = json.loads(json.dumps(experiment().metadata()))
    assert record["circuit"] == "divider" and len(record["netlist_sha256"]) == 64
    assert record["seed"] == 42 and record["samples"] == 2 and record["healthy_samples"] == 3
    assert record["simulator"] == "ngspice" and record["simulation"]["analyses"] == ["op"]
    assert [f["fault_id"] for f in record["faults"]] == ["R2:open", "R1:parametric:+0.2"]
    assert [v["component"] for v in record["variations"]] == ["R1", "R2"]
    assert [c["name"] for c in record["conditions"]] == ["nominal", "hot", "low"]
    assert Fault.from_metadata(record["faults"][0]) == OPEN


class NoOpens:
    """Backend that cannot solve a circuit with an open: stands for a convergence failure."""

    name = "fake"

    def version(self):
        return "0"

    def run(self, netlist, config):
        if "Rser_" in netlist:
            return SimulationResult(SimulationStatus.CONVERGENCE_ERROR, message="no convergence")
        return SimulationResult(SimulationStatus.SUCCESS)


@pytest.mark.parametrize("workers", [1, 2])
def test_failed_simulations_stay_in_the_record(workers):
    run = experiment(simulator=Simulator(NoOpens()), conditions=OperatingCondition()).run(workers)
    frame = run.to_frame()
    assert len(frame) == 7 and run.status_counts() == {"SUCCESS": 5, "CONVERGENCE_ERROR": 2}
    failed = frame[frame["status"] != "SUCCESS"]
    assert list(failed["fault_id"]) == ["R2:open", "R2:open"]
    assert list(failed["message"]) == ["no convergence"] * 2
    assert failed[["p_R1_value", "p_R2_value"]].notna().all().all()  # still traceable
    assert not run.samples[3].result.ok and run.samples[3].result.plots == ()


@pytest.mark.ngspice
class TestRun:
    def test_responses_match_the_divider_formula(self):
        result = experiment().run()
        assert len(result) == 21 and result.status_counts() == {"SUCCESS": 21}
        for s in result:
            r1, r2 = s.parameters[("R1", "value")], s.parameters[("R2", "value")]
            if s.fault_id == "R1:parametric:+0.2":
                r1 *= 1.2
            if s.fault_id == "R2:open":
                r2 += 1e9
            if s.condition == "hot":
                r1 *= 2.0
            source = 0.5 if s.condition == "low" else 1.0
            assert vout(s) == pytest.approx(source * r2 / (r1 + r2), rel=1e-6), s

    def test_result_does_not_depend_on_the_number_of_workers(self):
        one, three = experiment().run(workers=1), experiment().run(workers=3)
        assert one.metadata == three.metadata
        assert [s.sample for s in one] == [s.sample for s in three]
        assert [s.parameters for s in one] == [s.parameters for s in three]
        assert [vout(s) for s in one] == [vout(s) for s in three]

    def test_frame_has_definitions_status_and_parameters(self):
        frame = experiment().run().to_frame()
        assert list(frame.columns) == [
            "sample_id", "fault_index", "fault_id", "replica", "condition", "status", "message",
            "p_R1_value", "p_R2_value",
        ]
        assert frame["sample_id"].is_unique and (frame["status"] == "SUCCESS").all()
        assert np.isfinite(frame[["p_R1_value", "p_R2_value"]].to_numpy()).all()
