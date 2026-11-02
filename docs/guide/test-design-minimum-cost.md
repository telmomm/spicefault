# Test design: minimum-cost selection

The problem is to reach the coverage target set by the safety level, checked against
the lower bound, at the lowest hardware cost. `select.ilp` solves it exactly and
`select.pareto` by a greedy rule. In CERTA the integer programme solved every benchmark
problem (80 to 276 tests, up to 1 360 faulty circuits) to optimality in at most about
12 s with HiGHS; the greedy matched it on two circuits and was 4.1 points behind on the
third. The greedy is therefore the baseline to report a gap against, not a replacement.

## A problem on a fixed detection matrix

```python
from spicefault.testdesign.select import ilp

def duration(t):
    return costmodel.cost([t], board, access, price).time

inst = ilp.instance(formulation.part(read, "selection"), pool, fault_table, "dc_w",
                    access, board, price, alpha_per_test=0.01 / 4, duration_of=duration)
inst.coverage(plan), inst.false_alarm_rate(plan), inst.euros(plan), inst.time(plan)
```

To make coverage linear, each test gets limits of its own, at `alpha_per_test`, and a
plan alarms when any of its tests does. The coverage is the metric (DC_w, DC, SFF or
SPFM) as a weighted mean over the faulty circuits; the euros and time are those of
`costmodel.cost`. The union bound keeps the false alarm rate of a plan of k tests
within k · `alpha_per_test` in theory, but limits leak when tails are heavy, so the
false alarm rate of the plan on held-out fault-free circuits can be a constraint too.
`inst.subset(tests)` restricts the problem to some tests.

## The exact plan

```python
best = ilp.solve(inst, target=0.97, max_tests=4, max_false_alarm=0.01)
best.tests, best.euros, best.time, best.coverage, best.false_alarm
top = ilp.solve(inst, maximise=True, max_tests=2, max_euros=0.5)
```

`solve` minimises euros, then test time, then the number of tests, then maximises
coverage, each kept at its optimum while the next is, so that ties are not broken by
the solver. With `maximise` it returns the most covering plan of at most `max_tests`
tests within `max_euros`. `brute_force` solves the same problem by enumeration, and the
tests check the two against each other. `solve` needs `pip install "spicefault[ilp]"`
(PuLP and HiGHS); the rest of the module does not.

## The greedy and the Pareto front

```python
from spicefault.testdesign.select import pareto

path = pareto.greedy(pool_ids, scorer=inst.coverage,
                     coster=lambda p: (inst.euros(p), inst.time(p)),
                     target=0.97, max_time=0.05, max_tests=4)
pareto.front(plans)                 # plans no other beats on both cost and coverage
pareto.cheapest(plans, target=0.97) # the cheapest whose lower bound reaches it
```

The greedy adds the test with the largest gain per euro added, plus `epsilon` euros so
that tests that add no hardware are ranked by their gain. Tests that break the time
limit are skipped, and a scorer can return −inf to forbid a plan (one over a false
alarm budget). The scorer and the coster are functions of a plan, so the greedy works
with any metric and any cost model. Each prefix of its path is a plan.

The plan is chosen on the selection half; its claim is made on the evaluation half or
on new circuits ([scoring and certifying](test-design-formulation.md)).
