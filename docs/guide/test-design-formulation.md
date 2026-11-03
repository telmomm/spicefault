# Test design: scoring and certifying a plan

The plan that wins on some circuits looks better on them than it is. In CERTA, limits
set at 1 % / 12 per test gave up to 2.5 % false alarms on held-out fault-free circuits,
because the tails were heavier than the normal model. `testdesign.metrics.formulation`
keeps the data apart: limits on fault-free circuits used for nothing else, a plan
chosen on half of the faulty replicas and scored on the other half, and the claim made
on new circuits.

## Selection and evaluation halves

```python
from spicefault.testdesign.metrics import formulation

selection = formulation.part(read, "selection")    # first half of the replicas of each fault
evaluation = formulation.part(read, "evaluation")  # second half
s = formulation.score(evaluation, plan, fault_table, localise=True)
s["dc_w"], s.lower("dc_w"), s.h_prior
```

Both parts keep every fault-free row. `score` sets the limits on the calibration half
of them (`metrics.split_healthy`), and gives the detection, every metric with its
bounds (`metrics.coverage`) and, with `localise`, H(F | Y) with a uniform prior and
with the rates as prior. `bounds=False` gives the estimates only, fast enough inside a
search.

## Certifying on new circuits

```python
from spicefault.testdesign import costmodel
from spicefault.testdesign.metrics import adaptive

with formulation.ConditionSampler(
    circuit, faults, conditions, test_ids, GaussianLimits(0.01), variations=variations,
    seed=2, transform=lambda m: costmodel.observe(m, tests, board, amplitude, seed=2),
) as sampler:
    claim = adaptive.certify_sampled(sampler, fault_table, target=0.97, metric="dc_w")
```

`ConditionSampler` is `adaptive.SpiceSampler` for any operating conditions and test
names, with an optional `transform` of the test matrix, such as the ADC model, applied
before the detector. Each replica is simulated once and kept. Weighted fault sampling
certifies with few circuits: in CERTA, SPFM ≥ 97 % on a battery cell chain took 70
circuits, against 4 179 to bound each fault. See [stopping rules](test-design-adaptive.md);
the weights it draws with are `adaptive.draw_weights(faults, metric)`.

With `record=True`, `sampler.alarming` keeps the tests that alarm on each fault, the
safety mechanism of its row in the [FMEDA](test-design-fmeda.md).

## Tests from operating conditions

```python
conditions, test_ids = formulation.prefixed(case_conditions)
```

When the candidate tests are the measurements of some operating conditions rather
than `candidates` stimuli, `prefixed` names each `condition:measurement`, so that a
measurement read under two conditions gives two tests.
