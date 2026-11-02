# Test design: theoretical bounds

A plan chosen to detect faults says nothing about which component failed. In CERTA,
with every candidate test, the Sallen-Key and the biquad kept one ambiguity group each,
and the optimal detection plan of the biquad identified no component.
`testdesign.bounds` gives what any plan of the candidate tests could identify at best,
and what a plan leaves out of it.

## The sensitivity matrix

```python
from spicefault import Experiment
from spicefault.testdesign import bounds, candidates

components = ["R1", "R2", "R3", "C1", "C2"]
run = Experiment(circuit, faults=bounds.perturbations(components),
                 conditions=candidates.conditions(tests), samples=1, healthy_samples=1)
S = bounds.sensitivity(candidates.matrix(run.run().to_frame(), tests), test_ids, components)
```

S[t, c] = dy_t / d ln p_c, the change of each reading for a relative change of each
component, by central differences of ±1 % (`perturbations`), with phase differences
wrapped to ±180°.

## Ranks and ambiguity groups

```python
noise = read[read.fault_id == "healthy"][test_ids].std()
full = bounds.bound(S, noise)
full.rank, full.resolvable, full.groups, full.identifiable
plan = bounds.bound(S.loc[chosen], noise)
bounds.gap(full, plan)    # ranks and identifiable components the plan does not reach
```

Following Fedi, Manetti and Piccirilli, the testability of a set of tests is the rank
of its rows, and a canonical ambiguity group is a minimal set of components whose
columns are linearly dependent; the rank and the null space come from the SVD (Manetti
and Piccirilli, 2003). The algebraic rank is taken on rows scaled to unit norm, so
units do not matter; the resolvable rank divides each row by the fault-free spread of
its test and counts the directions along which a `deviation` of 10 % moves the readings
by more than `k` = 3 spreads.

`rtol` is 1e-2 by default. On S estimated by central differences with op-amp
macromodels, a smaller tolerance counts numerical noise as rank. `ambiguity_groups`
enumerates groups up to `max_size` components within the support of the null space.
