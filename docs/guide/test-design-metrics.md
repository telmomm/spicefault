# Test design: coverage metrics

`spicefault.testdesign.metrics` reads the test matrix of
[`candidates.matrix`](test-design-candidates.md) and says, for a set of tests, how
often each fault raises an alarm, how often a fault-free circuit does, and what
diagnostic coverage that gives, with bounds.

## Detection and false alarms

```python
from spicefault.testdesign import GaussianLimits, metrics

detection = metrics.detect(matrix, test_ids, GaussianLimits(alpha=0.01))
detection.per_fault      # per fault: n, detected, p and its Clopper–Pearson interval
detection.false_alarm    # the same on fault-free circuits not used for the limits
```

A circuit raises an alarm when any test of the set leaves its limits. `split_healthy`
divides the fault-free circuits by replica: the first half sets the limits and the
second measures the false alarm rate, so the split is the same for every set of tests.

| Detector | Limits | Use |
|---|---|---|
| `LimitTest(alpha)` (the default) | Sample quantiles at alpha / (2d) per side | Many fault-free circuits |
| `GaussianLimits(alpha, cover_sample=True)` | Median ± z · robust spread, z at alpha / (2d) per side; never inside the calibration range | Small alpha with few fault-free circuits |

With the limits set as sample quantiles on 200 fault-free circuits, the false alarm
rate of a set of tests was 4 % to 36 % against a 1 % target on the CERTA benchmark
circuits; limits extrapolated from a normal model kept it at 0 % to 2.5 %. Either way,
the rate is measured, not assumed.

Phase tests (`|phase@` in the id) are read as their deviation from the circular mean
of the calibration circuits, so that a phase near ±180° does not wrap into a large
change. `values` and `circular_mean` give the same reading for other uses.

## Coverage metrics

```python
faults = metrics.fault_table(detection.per_fault.index, rate=fit, severity=1.0,
                             iec=iec_class, iso=iso_class)
metrics.coverage(detection, faults)                    # estimate, lower, upper
metrics.coverage(detection, faults, method="jeffreys")
```

`fault_table` gives each fault a failure rate λ (any unit, such as FIT), a severity
w, an IEC class (safe or dangerous) and an ISO class (safe, single-point or
multiple-point). Each is one value for all faults or a Series indexed by fault. With
$P_k$ the detection probability of fault $k$:

| Metric | Definition | Source |
|---|---|---|
| DC_w | $\sum_k \lambda_k w_k P_k / \sum_k \lambda_k w_k$ | Risk-weighted coverage |
| DC | $\lambda_{DD} / \lambda_D$, with $\lambda_{DD} = \sum_{\text{dangerous}} \lambda_k P_k$ | IEC 61508-2 Annex C |
| SFF | $(\lambda_S + \lambda_{DD}) / (\lambda_S + \lambda_D)$ | IEC 61508-2 Annex C |
| SPFM | $1 - \sum_{\text{single-point}} \lambda_k (1 - P_k) / \sum_k \lambda_k$ | ISO 26262-5 Annex C |
| LFM | $1 - \sum_{\text{multiple-point}} \lambda_k (1 - P_k) / (\sum_k \lambda_k - \sum_{\text{single-point}} \lambda_k (1 - P_k))$ | ISO 26262-5 Annex C |

A metric with no fault of its kind is NaN. Each is non-decreasing in every $P_k$, so
per-fault bounds that hold for all faults at once give bounds on the metric:

- `bonferroni` (the default): one-sided Clopper–Pearson bounds at (1 − confidence) / K
  for each of the K faults. The bounds of the metric hold at least at `confidence`;
  they are conservative.
- `jeffreys`: quantiles of the metric over draws of each $P_k$ from its Jeffreys
  posterior, Beta(x + ½, n − x + ½). Less conservative, and Bayesian.

These are not the diagnostic coverage of `ReliabilityAnalysis`, which counts the
detected share of the circuits that fail a specification; the difference is set out in
[Reliability metrics](../RELIABILITY_METRICS.md#m6-and-the-coverage-metrics-of-testdesign).

## Targets

```python
metrics.asil_reached(spfm=bounds.loc["spfm", "lower"], lfm=bounds.loc["lfm", "lower"])
metrics.sil_reached(sff=bounds.loc["sff", "lower"], hft=0, element_type="B")
```

!!! warning "Target tables from drafts"
    `ASIL_TARGETS` (ISO 26262-5 Table E.1) and `SIL_ROUTE_1H` (IEC 61508-2 Tables 2
    and 3) are taken from the drafts ISO/DIS 26262-5 and prEN IEC 61508-2. They are to
    be checked against the editions in force before a claim is made with them.

Compare the lower bound, not the estimate, with a target.

## Localisation

```python
metrics.signatures(matrix, detection)      # per circuit: -, 0 or + for each test
metrics.ambiguity_sets(matrix, detection)  # faults grouped by their usual signature
metrics.localisation(matrix, detection)    # H(F) and H(F | Y), in bits
```

The ambiguity sets list first the faults whose usual signature is the fault-free one,
those the tests mostly miss. `localisation` estimates the uncertainty about the fault
left once the signature is read, H(F | Y), as a cross-entropy cross-validated between
even and odd replicas. It is an upper bound: a better model of P(f | y) could only
lower it, and fitting and scoring on the same circuits would make it look close to 0.
