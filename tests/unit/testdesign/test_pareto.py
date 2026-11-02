import pandas as pd
import pytest

from spicefault.testdesign.select import pareto as pr

# each test covers a set of equally weighted faults and sits at a node
COVERS = {"a": {1, 2, 3}, "b": {3, 4}, "c": {5}, "d": {1, 2, 3, 4, 5, 6}}
NODE = {"a": "n1", "b": "n1", "c": "n2", "d": "n3"}
NODE_EUROS = {"n1": 0.0, "n2": 0.0, "n3": 1.0}


def scorer(tests):
    return len(set().union(*(COVERS[t] for t in tests))) / 6


def coster(tests):
    nodes = {NODE[t] for t in tests}
    return sum(NODE_EUROS[n] for n in nodes), float(len(tests))


def test_free_tests_come_first_and_the_path_stops_when_nothing_helps():
    path = pr.greedy(list(COVERS), scorer, coster)
    # a covers 3 faults for free; then b and c add one each for free; d adds the last at 1 euro
    assert [s.tests[-1] for s in path] == ["a", "b", "c", "d"]
    assert [s.euros for s in path] == [0, 0, 0, 1]
    assert path[-1].score == pytest.approx(1.0)


def test_target_test_count_and_time_limit():
    path = pr.greedy(list(COVERS), scorer, coster, target=0.8)
    assert path[-1].tests == ("a", "b", "c") and path[-1].score >= 0.8
    assert len(pr.greedy(list(COVERS), scorer, coster, max_time=2.0)) == 2
    assert len(pr.greedy(list(COVERS), scorer, coster, max_tests=1)) == 1


def test_expensive_test_wins_when_it_is_worth_it():
    # with a large epsilon cost hardly matters, so d (all six faults) is taken first
    path = pr.greedy(list(COVERS), scorer, coster, epsilon=100.0)
    assert path[0].tests == ("d",) and len(path) == 1


def test_a_forbidden_plan_is_skipped():
    def no_d(tests):
        return float("-inf") if "d" in tests else scorer(tests)

    path = pr.greedy(list(COVERS), no_d, coster)
    assert "d" not in path[-1].tests and path[-1].score == pytest.approx(5 / 6)


def test_front_and_cheapest():
    plans = pd.DataFrame({
        "euros": [0, 0, 1, 1, 2], "coverage": [0.5, 0.7, 0.6, 0.9, 0.9],
        "lower": [0.4, 0.6, 0.5, 0.85, 0.88], "tests": [1, 2, 2, 3, 2], "time": [1, 2, 2, 3, 2],
    })  # fmt: skip
    f = pr.front(plans)
    assert list(zip(f.euros, f.coverage, strict=True)) == [(0, 0.7), (1, 0.9)]
    assert pr.cheapest(plans, 0.8).euros == 1
    assert pr.cheapest(plans, 0.95) is None
