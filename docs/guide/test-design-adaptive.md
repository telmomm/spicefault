# Test design: stopping rules

To certify a coverage of 99 % by bounding every fault took more than 22,400 circuits
on the Sallen-Key band-pass in CERTA; drawing the fault of each circuit with the
weights of the metric took 800, with an exact interval. `testdesign.metrics.adaptive`
has both, and rules to estimate and to check the false alarm rate.

## A fixed plan, read until the claim is decided

A rule decides how many replicas of each fault to read, never which circuits exist:
replica r of fault k is the circuit that the experiment draws from its own stream,
`sample_stream(seed, k, r)`. Each replica is therefore the same whatever the order in
which it is read, the number of workers or where a run stopped, and a run can be
repeated or continued on the same circuits. This is *sequential stopping over a fixed
plan*, which the [scope](../scope.md) admits; rare-event methods, whose samples depend
on earlier ones, stay outside.

A sampler is a function `sampler(fault_id, start, stop)` that returns the alarms of
replicas `start..stop-1`. `SpiceSampler` simulates them:

```python
from spicefault.testdesign.metrics import adaptive

with adaptive.SpiceSampler(circuit, faults, tests, GaussianLimits(0.01),
                           variations=variations, n_calibration=200, n_max=1600,
                           seed=1, workers=4) as sampler:
    claim = adaptive.certify_sampled(sampler, fault_table, target=0.99, metric="dc_w")
claim.decision    # certified, not reachable or undecided
claim.history     # per stage: circuits simulated, estimate and bounds
```

It builds one experiment with `n_max` replicas per fault and `n_calibration + n_max`
fault-free ones, fits the detector once on the first `n_calibration` fault-free
replicas, simulates a replica only when a rule asks for it, and keeps it. A failed
simulation counts as undetected for a fault and as an alarm for a fault-free circuit,
the conservative side for both claims. It runs through `Experiment.run_samples`, with
the worker processes kept for the whole run.

Any function with the same signature can stand in for it, such as a Bernoulli draw
with a known probability, which is how the rules are tested.

## The rules

| Rule | Decides | Bounds |
|---|---|---|
| `certify(sampler, faults, target, metric)` | metric ≥ target, with a claim about every fault | Per fault, one-sided Clopper–Pearson at α / (K L) for K faults and L stages |
| `certify_sampled(sampler, faults, target, metric)` | metric ≥ target, for DC_w, DC, SFF and SPFM | One exact interval on the total, at α / L |
| `estimate(sampler, faults, width)` | Nothing: estimates each fault to a given interval width | Not corrected for the looks |
| `certify_false_alarm(sampler, target)` | false alarm rate ≤ target | At α / L |

Stage sizes come from `schedule(n_start, n_max, growth)`: 25, 50, 100, ... up to
1,600 by default.

**`certify`** reads every fault at the first stage, then after each round grows only
the faults that cost most in the lower bound of the metric (at least `share` of the
largest). Dividing α by K L makes the per-fault bounds hold for every fault at every
stage at once, so looking after each stage does not inflate the chance of a wrong
claim.

**`certify_sampled`** uses that DC_w, DC, SFF and SPFM are means over the faults
weighted by their rates (and, for DC_w, severities). If the fault of each circuit is
drawn with those weights, each circuit is a success (an alarm, or a fault that cannot
lower the metric, which is not simulated) with probability equal to the metric. One
Clopper–Pearson interval on the total is then exact, with no bound per fault; it says
nothing about any one fault. LFM is not such a mean and is not offered. The draws come
from a generator seeded with `seed`.

In both, a claim is wrong at most α of the time; a test checks that the rate of wrong
claims of `certify_sampled` stays below 5 % on a population whose coverage is known.
