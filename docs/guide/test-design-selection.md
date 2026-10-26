# Test design: test selection

Published test selection methods are rarely reproducible from their papers: in CERTA,
6 of 19 published figures could be reproduced, and with 50 runs per fault the rankings
and selections of the methods changed from one sample to another. Comparisons between
methods are hard to repeat because the methods are in no library.
`testdesign.select.baselines` holds reference implementations of them, checked
against their sources. Every method reads the test matrix of
[`candidates.matrix`](test-design-candidates.md) and returns the tests it selects, in
the order it selects them, so that all can be evaluated with
[`testdesign.metrics`](test-design-metrics.md).

## Fault-pair Boolean table (Wang et al., 2025)

```python
from spicefault.testdesign.select import baselines

table = baselines.pair_table(matrix, test_ids, method="kde", threshold=0.95)
chosen = baselines.select_pairs(table)
baselines.isolation(table, chosen)   # pairs left, conditions isolated from all others
```

A test isolates a pair of conditions, the fault-free one included, when their
distance exceeds a threshold: `nominal`, the difference of the medians; `normal`,
1 − the Bhattacharyya coefficient of normal models; `kde`, 1 − the Bhattacharyya
coefficient of Gaussian kernel density estimates whose bandwidth minimises the
least-squares cross-validation score by grey wolf optimisation. `select_pairs` takes
the essential tests first, then the one isolating most remaining pairs. `rethreshold`
cuts the same distances again.

## Integer-coded dictionary

```python
codes = baselines.integer_codes(matrix, test_ids, half_width=0.7)
baselines.starzyk_entropy(codes)        # Starzyk et al. (2004)
baselines.genetic(codes, seed=0)        # Golonek and Rutkowski (2007)
baselines.minimum_sets(codes, max_size=4)
```

`integer_codes` groups the nominal responses of each test into ambiguity sets of a
fixed half-width (Hochwald and Bastian); `codes_from_values` does the same from a table
of nominal values, such as a published dictionary. `starzyk_entropy` adds the test
with the smallest entropy index Σ n ln n until it is zero or no test lowers it
(`starzyk_trace` gives the index of every test at every iteration, as in their tables);
`genetic` is the genetic algorithm of Golonek and Rutkowski with their parameters;
`minimum_sets` finds the smallest sets that isolate every condition by exhaustive
search.

On the published dictionary of Prasad and Babu, the entropy method selects n11, n9, n5
and n1 in that order and the genetic algorithm V1, V5, V9 and V11, as their sources
report; both are tests of the library.

## Greedy by the hardest fault (Zilch et al., 2022)

```python
dictionary = baselines.detection_table(matrix, test_ids, lambda: GaussianLimits(0.01))
baselines.zilch_greedy(dictionary)
```

`detection_table` says which test detects which fault: with its own limits, in at
least `share` of the circuits with that fault. `zilch_greedy` repeatedly takes, among
the tests that detect the hardest remaining fault, the one detecting most remaining
faults.

## What comes next

`select` is also where selection under a hardware cost, by a greedy rule and by an
integer programme, will go.
