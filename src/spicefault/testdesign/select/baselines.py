"""Published test selection methods, reimplemented as reference implementations.

Every method reads the test matrix of `candidates.matrix` (one row per drawn circuit,
one column per test) and returns the tests it selects, in the order it selects them,
so that all of them can be evaluated afterwards with `testdesign.metrics` on the same
yardstick.

- Fault-pair Boolean table (`pair_table`, `select_pairs`, `isolation`): which tests
  isolate which pairs of conditions, by a distance threshold on the nominal responses
  (the classic table), by the Bhattacharyya distance under a normal model, or by the
  Bhattacharyya distance between kernel density estimates with the bandwidth chosen by
  grey wolf optimisation (Wang et al., Intell. Robot. 5(2), 2025). The selection takes
  the essential tests first and then the one isolating most remaining pairs.
- Integer-coded dictionary (`integer_codes`, `codes_from_values`): ambiguity sets of a
  fixed width per test (Hochwald and Bastian), on which Starzyk's entropy
  (`starzyk_entropy`, Starzyk et al., IEEE TIM 53(3), 2004, Eq. 2 and Section III), the
  genetic algorithm (`genetic`, Golonek and Rutkowski, IEEE TCAS-II 54(2), 2007,
  Section III, with their parameters) and the exhaustive search (`minimum_sets`) work.
- Greedy by the hardest fault (`detection_table`, `zilch_greedy`, Zilch et al.,
  AICSP 112, 2022).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from ..metrics.detection import HEALTHY, angles_of, split_healthy, values

_SQRT2PI = math.sqrt(2 * math.pi)
# NumPy 2 renamed trapz
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def _conditions(matrix: pd.DataFrame) -> list[str]:
    """The fault-free state first, then the faults, in the order of the matrix."""
    ids = list(dict.fromkeys(matrix.fault_id))
    return [HEALTHY, *[i for i in ids if i != HEALTHY]] if HEALTHY in ids else ids


# --- kernel density estimation and the Bhattacharyya coefficient ----------------------


def lscv(x: np.ndarray, h: float) -> float:
    """Least-squares cross-validation score of a Gaussian KDE with bandwidth h: an
    estimate of its integrated squared error, up to a constant. Smaller is better.
    """
    n = len(x)
    d = (x[:, None] - x[None, :]) / h
    integral = np.exp(-d * d / 4).sum() / (n * n * h * 2 * math.sqrt(math.pi))
    off = np.exp(-d * d / 2)
    np.fill_diagonal(off, 0.0)
    leave_one_out = off.sum() / (n * (n - 1) * h * _SQRT2PI)
    return float(integral - 2 * leave_one_out)


def grey_wolf(
    objective,
    low: float,
    high: float,
    wolves: int = 10,
    iterations: int = 30,
    rng: np.random.Generator | None = None,
) -> float:
    """Minimise a function of one variable on [low, high] by grey wolf optimisation
    (Mirjalili et al., 2014): each wolf moves to the mean of three steps towards the best
    three, with a coefficient that decreases linearly from 2 to 0.
    """
    rng = rng or np.random.default_rng(0)
    position = rng.uniform(low, high, wolves)
    fitness = np.array([objective(p) for p in position])
    for t in range(iterations):
        order = np.argsort(fitness)
        leaders = position[order[:3]]
        a = 2 - 2 * t / iterations
        steps = []
        for leader in leaders:
            A = 2 * a * rng.random(wolves) - a
            C = 2 * rng.random(wolves)
            steps.append(leader - A * np.abs(C * leader - position))
        position = np.clip(np.mean(steps, axis=0), low, high)
        fitness = np.array([objective(p) for p in position])
    return float(position[np.argmin(fitness)])


def bandwidth(x: np.ndarray, method: str = "gwo", rng: np.random.Generator | None = None) -> float:
    """Bandwidth of a Gaussian KDE: `silverman` (rule of thumb) or `gwo` (LSCV minimised
    by grey wolf optimisation between a tenth and three times the rule of thumb).
    """
    spread = np.std(x, ddof=1) if len(x) > 1 else 0.0
    iqr = np.subtract(*np.percentile(x, [75, 25])) / 1.349
    scale = min(spread, iqr) if iqr > 0 else spread
    if scale <= 0:
        return 0.0
    rule = 0.9 * scale * len(x) ** -0.2
    if method == "silverman":
        return float(rule)
    if method == "gwo":
        return grey_wolf(lambda h: lscv(x, h), rule / 10, rule * 3, rng=rng)
    raise ValueError(f"unknown bandwidth method {method!r}")


def _kde(x: np.ndarray, h: float, grid: np.ndarray) -> np.ndarray:
    z = (grid[:, None] - x[None, :]) / h
    return np.exp(-z * z / 2).sum(axis=1) / (len(x) * h * _SQRT2PI)


def bhattacharyya_kde(
    x: np.ndarray, y: np.ndarray, hx: float, hy: float, points: int = 512
) -> float:
    """BC = ∫ √(p_x p_y), with each density a Gaussian KDE, on a common grid.

    A sample with no spread is a point mass: BC is 1 if both are the same point, 0 if
    the point is far from the other sample, and NaN otherwise.
    """
    if hx <= 0 or hy <= 0:
        if hx <= 0 and hy <= 0:
            return float(np.isclose(x[0], y[0]))
        point, other, h = (x[0], y, hy) if hx <= 0 else (y[0], x, hx)
        far = point < other.min() - 6 * h or point > other.max() + 6 * h
        return 0.0 if far else float("nan")
    if x.max() + 5 * hx < y.min() - 5 * hy or y.max() + 5 * hy < x.min() - 5 * hx:
        return 0.0  # the supports do not overlap
    lo = min(x.min() - 5 * hx, y.min() - 5 * hy)
    hi = max(x.max() + 5 * hx, y.max() + 5 * hy)
    grid = np.linspace(lo, hi, points)
    value = _trapezoid(np.sqrt(_kde(x, hx, grid) * _kde(y, hy, grid)), grid)
    return float(min(value, 1.0))


def bhattacharyya_normal(x: np.ndarray, y: np.ndarray) -> float:
    """BC between two normal distributions with the mean and spread of the samples."""
    mx, my, sx, sy = x.mean(), y.mean(), x.std(ddof=1), y.std(ddof=1)
    if sx <= 0 or sy <= 0:
        return float(np.isclose(mx, my)) if sx <= 0 and sy <= 0 else 0.0
    total = sx * sx + sy * sy
    return float(math.sqrt(2 * sx * sy / total) * math.exp(-((mx - my) ** 2) / (4 * total)))


# --- fault-pair Boolean table ---------------------------------------------------------


@dataclass(frozen=True)
class PairTable:
    """Which test isolates which pair of conditions. `table` has one row per pair (u, v)
    and one Boolean column per test; `distance` the distances it was cut from.
    """

    conditions: tuple[str, ...]
    table: pd.DataFrame
    distance: pd.DataFrame
    method: str
    threshold: float | pd.Series


def pair_table(
    matrix: pd.DataFrame,
    tests: Sequence[str],
    method: str = "kde",
    threshold: float | pd.Series = 0.95,
    bandwidth_method: str = "gwo",
    seed: int = 0,
) -> PairTable:
    """The fault-pair Boolean table of `tests` over all conditions, the fault-free one
    included.

    - `kde`: d = 1 − BC of the KDE of each condition (Wang et al. 2025); isolable if
      d > threshold.
    - `normal`: d = 1 − BC of normal models; isolable if d > threshold.
    - `nominal`: d = |median_u − median_v|, in the units of the test; isolable if
      d > threshold.

    `threshold` is a number or one value per test (a Series indexed by test).
    """
    tests = list(tests)
    conditions = _conditions(matrix)
    groups = {c: matrix.loc[matrix.fault_id == c, tests].to_numpy(float) for c in conditions}
    pairs = list(combinations(range(len(conditions)), 2))
    index = pd.MultiIndex.from_tuples(
        [(conditions[u], conditions[v]) for u, v in pairs], names=["u", "v"]
    )
    distance = np.zeros((len(pairs), len(tests)))
    rng = np.random.default_rng(seed)
    for j in range(len(tests)):
        samples = [groups[c][:, j] for c in conditions]
        if method == "kde":
            h = [bandwidth(s, bandwidth_method, rng) for s in samples]
            for p, (u, v) in enumerate(pairs):
                bc = bhattacharyya_kde(samples[u], samples[v], h[u], h[v])
                if math.isnan(bc):
                    bc = bhattacharyya_normal(samples[u], samples[v])
                distance[p, j] = 1 - bc
        elif method == "normal":
            for p, (u, v) in enumerate(pairs):
                distance[p, j] = 1 - bhattacharyya_normal(samples[u], samples[v])
        elif method == "nominal":
            medians = [np.median(s) for s in samples]
            for p, (u, v) in enumerate(pairs):
                distance[p, j] = abs(medians[u] - medians[v])
        else:
            raise ValueError(f"unknown pair table method {method!r}")
    distance = pd.DataFrame(distance, index=index, columns=tests)
    return PairTable(tuple(conditions), distance.gt(threshold), distance, method, threshold)


def rethreshold(table: PairTable, threshold: float | pd.Series) -> PairTable:
    """The same table cut at another threshold (a number, or one per test)."""
    return PairTable(
        table.conditions, table.distance.gt(threshold), table.distance, table.method, threshold
    )


def select_pairs(table: PairTable) -> list[str]:
    """Wang et al. (2025), Section 4.1: first every test that is the only one to isolate
    some pair; then, repeatedly, the test that isolates most of the remaining pairs,
    until every pair that some test can isolate is isolated.
    """
    t = table.table
    remaining = t[t.any(axis=1)]
    chosen = list(remaining[remaining.sum(axis=1) == 1].idxmax(axis=1).unique())
    if chosen:
        remaining = remaining[~remaining[chosen].any(axis=1)]
    while len(remaining):
        counts = remaining.drop(columns=chosen).sum()
        if counts.max() == 0:
            break
        best = counts.idxmax()
        chosen.append(best)
        remaining = remaining[~remaining[best]]
    return chosen


def isolation(table: PairTable, tests: Sequence[str]) -> dict:
    """With `tests`, the pairs left unisolated and the conditions isolated from every other
    one (the "fault isolation degree" of Wang et al., which counts the fault-free state).
    """
    if tests:
        covered = table.table[list(tests)].any(axis=1)
    else:
        covered = pd.Series(False, index=table.table.index)
    left = covered[~covered].index
    ambiguous = {c for pair in left for c in pair}
    return {
        "remaining_pairs": len(left),
        "isolated": len(table.conditions) - len(ambiguous),
        "conditions": len(table.conditions),
    }


# --- integer-coded dictionary, entropy and genetic algorithm -------------------------


def integer_codes(
    matrix: pd.DataFrame, tests: Sequence[str], half_width: float | dict = 0.7
) -> pd.DataFrame:
    """Integer code of each condition at each test (Hochwald and Bastian; Golonek and
    Rutkowski 2007, Eq. 2): the nominal (median) responses are sorted and grouped into
    ambiguity sets whose members all lie within ± `half_width` of the mean of the set
    (0.7 V in the sources, for node voltages). `half_width` may be one value per test.
    """
    conditions = _conditions(matrix)
    nominal = matrix.groupby("fault_id")[list(tests)].median().loc[conditions]
    return codes_from_values(nominal, half_width)


def codes_from_values(nominal: pd.DataFrame, half_width: float | dict = 0.7) -> pd.DataFrame:
    """Integer codes from one nominal value per condition (rows) and test (columns)."""
    codes = pd.DataFrame(0, index=nominal.index, columns=nominal.columns)
    for test in nominal.columns:
        w = half_width[test] if isinstance(half_width, dict) else half_width
        code, members = 0, []
        for condition, value in nominal[test].sort_values().items():
            trial = [*members, value]
            mean = float(np.mean(trial))
            if members and (value - mean > w or mean - trial[0] > w):
                code, members = code + 1, [value]
            else:
                members = trial
            codes.loc[condition, test] = code
    return codes


def _array(codes) -> np.ndarray:
    if isinstance(codes, pd.DataFrame):
        return codes.to_numpy(dtype=np.int64)
    return np.asarray(codes, dtype=np.int64)


def _set_sizes(arr: np.ndarray, columns) -> np.ndarray:
    """Size of the ambiguity set of each row with the joint codes of `columns`."""
    columns = list(columns)
    if not columns:
        return np.full(len(arr), len(arr))
    _, inverse, counts = np.unique(
        arr[:, columns], axis=0, return_inverse=True, return_counts=True
    )
    return counts[inverse.ravel()]


def _entropy(arr: np.ndarray, columns) -> float:
    columns = list(columns)
    if not columns:
        n = np.array([len(arr)], dtype=float)
    else:
        n = np.unique(arr[:, columns], axis=0, return_counts=True)[1].astype(float)
    return float((n * np.log(n)).sum())


def entropy_index(codes: pd.DataFrame, tests: Sequence[str]) -> float:
    """Σ over ambiguity sets of n ln n, for the joint codes of `tests`: smaller means the
    conditions are spread over more, smaller sets.
    """
    return _entropy(_array(codes), [codes.columns.get_loc(t) for t in tests])


def starzyk_trace(codes: pd.DataFrame, base: float = 10.0) -> list[dict]:
    """The entropy index of every candidate test at each iteration of `starzyk_entropy`, in
    logarithms of `base` (Starzyk et al. report base 10), as in their Table VII.
    """
    arr, names = _array(codes), list(codes.columns)
    chosen: list[int] = []
    trace = []
    current = _entropy(arr, chosen)
    while True:
        found = {j: _entropy(arr, [*chosen, j]) for j in range(len(names)) if j not in chosen}
        if not found:
            break
        trace.append({names[j]: v / math.log(base) for j, v in found.items()})
        best = min(found, key=found.get)
        if found[best] >= current - 1e-12:
            break
        chosen.append(best)
        current = found[best]
        if current == 0:
            break
    return trace


def starzyk_entropy(codes: pd.DataFrame) -> list[str]:
    """Starzyk et al. (2004), Section III: add, one at a time, the test with the smallest
    entropy index E = Σ F log F over the ambiguity sets of the joint codes; stop when the
    index reaches zero or no test lowers it. Isolated conditions add 1 log 1 = 0, which is
    the same as removing them from the dictionary as in Step 6. Ties go to the first test.
    """
    arr, names = _array(codes), list(codes.columns)
    chosen: list[int] = []
    current = _entropy(arr, chosen)
    while current > 0:
        found = {j: _entropy(arr, [*chosen, j]) for j in range(len(names)) if j not in chosen}
        if not found:
            break
        best = min(found, key=found.get)
        if found[best] >= current - 1e-12:
            break
        chosen.append(best)
        current = found[best]
    return [names[j] for j in chosen]


def isolated_by_codes(codes: pd.DataFrame, tests: Sequence[str]) -> int:
    """Conditions alone in their ambiguity set with the joint codes of `tests`."""
    columns = [codes.columns.get_loc(t) for t in tests]
    return int((_set_sizes(_array(codes), columns) == 1).sum())


def minimum_sets(codes: pd.DataFrame, max_size: int) -> tuple[int | None, int]:
    """Smallest number of tests that isolate every condition, by exhaustive search up to
    `max_size` tests, and how many sets of that size do so; (None, 0) if none up to
    `max_size`. Each candidate set is checked by sorting the joint codes of all conditions.
    """
    arr = _array(codes)
    n, k = arr.shape
    base = int(arr.max()) + 1
    for size in range(1, max_size + 1):
        found = 0
        sets = np.array(list(combinations(range(k), size)), dtype=np.int64)
        for chunk in np.array_split(sets, max(1, len(sets) // 20000)):
            keys = np.zeros((len(chunk), n), dtype=np.int64)
            for position in range(size):
                keys = keys * base + arr[:, chunk[:, position]].T
            keys.sort(axis=1)
            found += int((np.diff(keys, axis=1) != 0).all(axis=1).sum())
        if found:
            return size, found
    return None, 0


def genetic(
    codes: pd.DataFrame,
    population: int = 40,
    generations: int = 30,
    crossover: float = 0.8,
    mutation: float = 0.08,
    scaling: float = 150.0,
    cut_points: int = 4,
    code_modification: bool = True,
    seed: int = 0,
    report: dict | None = None,
) -> list[str]:
    """Genetic algorithm of Golonek and Rutkowski (2007), with their parameters.

    A genotype is one bit per test. Fitness is (S + R) / (F + K + 1), with S the
    conditions isolated, R the tests left out, F + 1 the conditions and K the tests
    (Eq. 8); it is scaled as Fit^C before selection (Eq. 9). Each generation keeps the
    best genotype (elitism), draws the mating pool by roulette on the scaled fitness,
    crosses random pairs at `cut_points` points with probability `crossover` and flips
    each bit with probability `mutation`. With `code_modification`, genes are reordered
    before the genetic operations so that the ones of the best genotype are contiguous
    (Section III-C); this changes only how crossover cuts, not what a genotype means.

    Since a test left out counts as much as a condition isolated, a test that isolates a
    single condition is worth nothing, and when few conditions can be isolated the empty
    set can win. The source applies it where every condition can be isolated.

    If `report` is a dictionary, it receives the generation at which the returned
    genotype was first found (0 for the initial population).
    """
    rng = np.random.default_rng(seed)
    tests = list(codes.columns)
    arr = _array(codes)
    K, states = len(tests), len(codes)
    order = np.arange(K)  # gene position -> test index
    cache: dict[bytes, float] = {}

    def fitness(bits: np.ndarray) -> float:
        chosen_bits = np.zeros(K, dtype=bool)
        chosen_bits[order] = bits
        key = chosen_bits.tobytes()
        if key not in cache:
            chosen = np.flatnonzero(chosen_bits)
            isolated = int((_set_sizes(arr, chosen) == 1).sum())
            cache[key] = (isolated + (K - len(chosen))) / (states + K)
        return cache[key]

    pop = rng.random((population, K)) < 0.5
    score = np.array([fitness(g) for g in pop])
    best, best_score = pop[np.argmax(score)].copy(), score.max()
    best_order, found_at = order.copy(), 0
    for generation in range(1, generations + 1):
        if code_modification:
            leader = pop[np.argmax(score)]
            perm = np.concatenate([np.flatnonzero(leader), np.flatnonzero(~leader)])
            pop, order = pop[:, perm], order[perm]
        scaled = score**scaling
        p = scaled / scaled.sum() if scaled.sum() > 0 else np.full(population, 1 / population)
        pool = pop[rng.choice(population, population, p=p)]
        children = [pop[np.argmax(score)].copy()]
        for i in range(0, population - 1, 2):
            a, b = pool[i].copy(), pool[i + 1].copy()
            if rng.random() < crossover:
                cuts = np.sort(rng.choice(np.arange(1, K), min(cut_points, K - 1), replace=False))
                swap = np.zeros(K, dtype=bool)
                for c, cut in enumerate(cuts):
                    if c % 2 == 0:
                        swap[cut : cuts[c + 1] if c + 1 < len(cuts) else K] = True
                a[swap], b[swap] = pool[i + 1][swap], pool[i][swap]
            children += [a, b]
        pop = np.array(children[:population])
        pop[1:] ^= rng.random((population - 1, K)) < mutation
        score = np.array([fitness(g) for g in pop])
        if score.max() > best_score:
            best, best_score, best_order = pop[np.argmax(score)].copy(), score.max(), order.copy()
            found_at = generation
    if report is not None:
        report["generation"] = found_at
    chosen_bits = np.zeros(K, dtype=bool)
    chosen_bits[best_order] = best
    return [t for t, b in zip(tests, chosen_bits, strict=True) if b]


# --- greedy by the hardest fault ------------------------------------------------------


def detection_table(
    matrix: pd.DataFrame,
    tests: Sequence[str],
    detector_factory,
    share: float = 0.5,
    fraction: float = 0.5,
) -> pd.DataFrame:
    """Binary fault dictionary: a test detects a fault when, with its own limits set on the
    calibration fault-free circuits (see `metrics.split_healthy`), it raises an alarm in
    at least `share` of the circuits with that fault. Phases are read around their
    fault-free direction, as in `metrics.detect`. Each test gets its own detector, made
    by `detector_factory()` and fitted on that test alone.
    """
    tests = list(tests)
    calibration, _ = split_healthy(matrix, fraction)
    angles = angles_of(matrix, calibration, tests)
    x = values(matrix, tests, angles)
    calibration_rows = matrix.index.get_indexer(calibration)
    alarms = np.zeros_like(x, dtype=bool)
    for j in range(len(tests)):
        detector = detector_factory().fit(x[calibration_rows, j : j + 1])
        alarms[:, j] = np.asarray(detector.flag(x[:, j : j + 1]), dtype=bool)
    faulty = (matrix.fault_id != HEALTHY).to_numpy()
    by_fault = matrix.fault_id.to_numpy()[faulty]
    rates = pd.DataFrame(alarms[faulty], columns=tests).groupby(by_fault, sort=False).mean()
    return rates >= share


def zilch_greedy(detects: pd.DataFrame) -> list[str]:
    """Zilch et al. (2022), Algorithm 1: repeatedly take, among the tests that detect the
    hardest remaining fault (the one detected by fewest tests), the one detecting most
    remaining faults, then most faults in all; stop when no remaining fault is detectable.
    """
    remaining = detects.index[detects.any(axis=1)]
    chosen: list[str] = []
    total = detects.sum()
    while len(remaining):
        hardest = detects.loc[remaining].sum(axis=1).idxmin()
        options = [t for t in detects.columns if detects.loc[hardest, t] and t not in chosen]
        new = detects.loc[remaining, options].sum()
        best = max(options, key=lambda t: (new[t], total[t]))  # noqa: B023
        chosen.append(best)
        remaining = remaining[~detects.loc[remaining, best].to_numpy()]
    return chosen
