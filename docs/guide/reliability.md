# Reliability analysis

`ReliabilityAnalysis` reads the dataset of a campaign and computes what the fault
responses say about the circuit:

```python
from spicefault.reliability import ReliabilityAnalysis, robustness

analysis = ReliabilityAnalysis.from_dataset("data/rc_lowpass")

detection = analysis.detectability(alpha=0.01)   # limit test at 1 % false alarms
detection.table                                  # per fault: P(detect), interval, bounds
detection.false_alarm                            # measured on held-out healthy samples

analysis.standardised_shift()     # how far each fault moves the best feature
analysis.auc()                    # threshold-free view
analysis.minimum_detectable()     # smallest deviation detected 90 % of the time
analysis.ambiguity()              # faults and components that cannot be told apart
```

| Question | Method |
|---|---|
| Is the fault detected? | `detectability`, `standardised_shift`, `auc` |
| Does the circuit still meet its specifications? | `failure_probability` (needs a `compliant` column from the application) |
| What fraction of the failures is caught? | `diagnostic_coverage`: coverage, escape rate, false-reject rate |
| How small a deviation is visible? | `severity_response`, `minimum_detectable`, `local_sensitivity` |
| How does tolerance erode detection? | `robustness`, over campaigns at scaled tolerances |
| Does it depend on the operating condition? | `analysis.by("condition")`, `detectability_across` |
| Which faults look alike? | `separation`, `ambiguity` |

Every proportion comes with a confidence interval. When simulations failed, detection
is also given as bounds, counting the failed ones first as undetected and then as
detected. Thresholds are set on one half of the healthy samples and the false-alarm
rate is measured on the other.

[examples/filter/reliability.py](examples/filter/reliability.py) runs every metric on
the RC filter at three tolerance scales. Among its results: a ±5 % fault of the 5 %
capacitor is detected about half of the time, which is what the tolerance overlap
predicts (half of that fault population is inside the tolerance band); and doubling
the tolerances takes the detection of the ±5 % resistor faults from 100 % to between
72 and 83 %.

## What the numbers depend on

Every figure is conditional on four things, which should be reported with it: the
measurements used, the instrument noise applied to them, the tolerances of the
campaign and its operating condition. Detection also depends on the decision rule; the
default is a limit test, which flags a sample when any measurement is outside the
range of the healthy circuits.

Three rules are applied throughout:

- **Intervals.** Every proportion comes with a Wilson (or Clopper–Pearson) interval.
- **Failed simulations.** Proportions are computed over the successful simulations,
  and also given as bounds that count every failed one first as undetected and then
  as detected.
- **Healthy samples are split.** One half sets the thresholds and the other measures
  the false-alarm rate. Measured on the samples that set the thresholds, the rate
  looks right whatever it really is.

## Specifications

Failure probability and diagnostic coverage need to know whether a circuit meets its
specification. That is the application's knowledge: give the limits to the dataset,
which labels every sample and keeps them (see [Datasets](datasets.md)), and name the
column.

```python
from spicefault import Specification

dataset.label([Specification("gain_1k", minimum=0.14, maximum=0.17)])
analysis = dataset.analysis(compliant="compliant")
analysis.failure_probability()      # per fault, and the yield of the healthy circuits
analysis.diagnostic_coverage()      # of the circuits that fail, the fraction detected
```

The definitions, estimators and their limits are in
[reliability metrics](../RELIABILITY_METRICS.md).
