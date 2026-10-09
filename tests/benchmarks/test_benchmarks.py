"""The benchmarks run and write well-formed results. Sizes are tiny: timings mean nothing here."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repository root

from benchmarks import common  # noqa: E402
from benchmarks.correctness import run as correctness  # noqa: E402
from benchmarks.fault_coverage import run as fault_coverage  # noqa: E402
from benchmarks.reproducibility import run as reproducibility  # noqa: E402
from benchmarks.sampling import run as sampling  # noqa: E402
from benchmarks.scalability import run as scalability  # noqa: E402


@pytest.fixture
def results(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "RESULTS", tmp_path)
    return tmp_path


def test_environment_record():
    record = common.environment()
    assert {"date", "host", "platform", "cpu", "python", "numpy", "pandas", "spicefault_version",
            "spicefault_commit", "uncommitted_changes", "simulator"} <= set(record)  # fmt: skip
    assert record["cpu"]["logical_cores"] >= 1 and record["cpu"]["model"]
    json.dumps(record)
    assert common.spread([3.0, 1.0, 2.0]) == {"median": 2.0, "min": 1.0, "max": 3.0, "n": 3}
    assert common.peak_memory_mb()["parent_mb"] > 1


def test_fault_coverage(results):
    result = fault_coverage.run()
    rc = result["circuits"]["rc"]
    assert (rc["n_components"], rc["n_components_with_faults"]) == (4, 3)
    assert rc["components_without_faults"] == ["V1"]
    assert rc["n_fault_conditions"] == rc["n_selected"] == 30 and rc["structural_coverage"] == 1.0
    assert rc["conditions_per_fault_type"] == {"open": 3, "short": 3, "parametric": 24}
    assert rc["coverage_matrix_universe"]["C1"] == {"open": 1, "short": 1, "parametric": 8}
    assert rc["inside_tolerance"]["faults"] == {
        "C1:parametric:-0.05": 0.5, "C1:parametric:+0.05": 0.5, "C1:parametric:+0.1": 0.0455,
    }
    path = common.save("fault_coverage", result, "test")
    assert path.parent == results / "fault_coverage" and path.name.endswith("_test.json")
    assert json.loads(path.read_text())["benchmark"] == "fault_coverage"
    assert "30 fault conditions" in fault_coverage.report(result)
    counts = {name: c["n_fault_conditions"] for name, c in result["circuits"].items()}
    assert counts == {"rc": 30, "biquad": 104, "regulator": 54, "sallen_key": 58}
    regulator = result["circuits"]["regulator"]
    assert {"Vin", "Iout", "RL"} <= set(regulator["components_without_faults"])
    assert regulator["coverage_matrix_universe"]["XQ1"] == {"transistor": 6}


@pytest.mark.ngspice
def test_correctness_against_ngspice_run_directly(results):
    result = correctness.run(("rc",))
    rc = result["circuits"]["rc"]
    assert (rc["n_components"], rc["n_faults"], rc["total"]["decks"]) == (4, 30, 31)
    assert set(rc["by_fault_type"]) == {"healthy", "open_circuit", "short_circuit", "parametric"}
    assert rc["by_fault_type"]["parametric"]["status"] == {"SUCCESS": 24}
    assert rc["total"]["values"] == rc["total"]["values_bit_identical"] > 1000
    assert result["verdict"] == {
        "decks": 31, "values": rc["total"]["values"], "bit_identical": True, "max_abs_diff": 0.0,
    }
    assert "31 decks" in correctness.report(result) and "NOT" not in correctness.report(result)
    saved = json.loads(common.save("correctness", result, "test").read_text())
    assert saved["benchmark"] == "correctness" and saved["protocol"]["seed"] == 42


def test_correctness_counts_values_by_their_bits():
    a = np.array([1 + 2j, 3 + 4j, 5 + 6j])
    assert correctness.identical_values(a, a.copy()) == 3
    assert correctness.identical_values(a, np.array([1 + 2j, 3 + 5j, 5 + 6j])) == 2
    assert correctness.identical_values(np.array([0.1 + 0.2]), np.array([0.3])) == 0
    assert correctness.identical_values(np.array([np.nan]), np.array([np.nan])) == 1


@pytest.mark.ngspice
def test_scalability(results):
    result = scalability.run("rc", workers=(1, 2), repetitions=2, n_samples=60, chunk=30, settle=0)
    assert result["protocol"]["n_samples"] == 60 and len(result["runs"]) == 4
    table = result["summary"]["spicefault"]
    assert table["1"]["speedup"] == 1.0 and table["1"]["efficiency"] == 1.0
    assert table["2"]["wall_s"]["n"] == 2 and table["2"]["sims_per_s"]["median"] > 0
    assert table["2"]["efficiency"] == pytest.approx(table["2"]["speedup"] / 2)
    phases = result["phase_breakdown"]
    assert set(phases["ms_per_sample"]) == {
        "netlist_generation", "simulator_process", "output_parsing", "measurement", "storage",
    }
    assert sum(phases["fraction"].values()) == pytest.approx(1.0)
    assert phases["fraction"]["simulator_process"] > 0.5  # the simulator dominates
    assert result["resume_overhead"]["simulations_repeated"] == 0
    assert result["bytes_per_sample"] > 0 and result["peak_memory"]["largest_child_mb"] > 0
    saved = json.loads(common.save("scalability", result, "test").read_text())
    assert saved["summary"]["spicefault"]["2"]["speedup"] == table["2"]["speedup"]
    assert "speed-up" in scalability.report(result)
    assert set(result["machine_before_start"]) == {
        "waited_s", "load_average_1min", "quiet_below", "quiet",
    }
    assert all("load_average_before" in r for r in result["runs"])


def test_waiting_for_a_quiet_machine_gives_up_after_the_timeout(monkeypatch):
    monkeypatch.setattr(common.os, "getloadavg", lambda: (99.0, 99.0, 99.0))
    busy = common.wait_until_quiet(timeout=0.0)
    assert busy["quiet"] is False and busy["load_average_1min"] == 99.0
    monkeypatch.setattr(common.os, "getloadavg", lambda: (0.1, 0.1, 0.1))
    assert common.wait_until_quiet(timeout=0.0) == {
        "waited_s": 0.0, "load_average_1min": 0.1, "quiet_below": busy["quiet_below"],
        "quiet": True,
    }


@pytest.mark.ngspice
def test_reproducibility(results):
    result = reproducibility.run("rc", workers=2, n_samples=120, chunk=30)
    assert result["verdict"] == {"L1_exact": True, "L2_identical": True, "L2_max_rel_diff": 0.0}
    assert set(result["comparisons_with_one_worker"]) == {
        "several_workers", "several_workers_again", "other_chunk_size", "resumed",
    }
    assert result["resumed_run"] == {
        "samples_before_interruption": 30,
        "manifest_says_resumed": True,
    }
    assert result["integrity_problems"] == []
    again = result["simulated_again_from_the_folder"]
    assert again["drawn_values_identical"] and again["max_abs_diff"] == 0.0
    json.loads(common.save("reproducibility", result, "test").read_text())


@pytest.mark.ngspice
def test_scalability_against_the_direct_script(results):
    result = scalability.run(
        "sallen_key", workers=(2,), repetitions=1, n_samples=120, chunk=60, settle=0
    )
    assert result["protocol"]["n_samples"] == 120
    assert set(result["summary"]) == {"spicefault", "direct_script"}
    order = [(r["implementation"], r["workers"]) for r in result["runs"]]
    assert order == [("spicefault", 2), ("direct_script", 2)]  # one after the other
    assert 0.2 < result["relative_throughput"]["2"] < 5.0
    assert 0.2 < result["relative_throughput_of_fastest_runs"]["2"] < 5.0
    assert "direct_script" in scalability.report(result)


@pytest.mark.ngspice
def test_sampling_benchmark_compares_the_methods_at_equal_cost(results):
    pytest.importorskip("scipy")  # for the Sobol sequence
    assert sampling.exact_mean_gain() == pytest.approx(0.7046, abs=2e-3)
    result = sampling.run(sizes=(32,), repetitions=4)
    rows = {row["method"]: row for row in result["rows"]}
    assert set(rows) == {"random", "lhs", "sobol"} and rows["random"]["mean_error_vs_random"] == 1
    assert all(row["simulations"] == 32 and row["rmse_mean"] > 0 for row in rows.values())
    assert rows["lhs"]["rmse_mean"] < rows["random"]["rmse_mean"]
    path = common.save("sampling", result, "test")
    assert json.loads(path.read_text())["exact"]["yield"] == 0.9
    assert "rmse(mean)" in sampling.report(result)
