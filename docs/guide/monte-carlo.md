# Monte Carlo and yield

A campaign does not need faults. Without them it is a Monte Carlo study of the healthy
population: circuits drawn from the variations, simulated, and measured. The engine, the
dataset and its provenance are the same as in a fault study.

```python
from spicefault import Campaign, Circuit, Measurement, SimulationConfig, Specification
from spicefault.variation import tolerances

circuit = Circuit.from_netlist("examples/filter/rc_lowpass.cir")
campaign = Campaign(
    circuit,
    out_dir="data/rc_monte_carlo",
    samples=10_000,
    variations=tolerances(circuit, {"R": 0.01, "C": 0.05}),
    config=SimulationConfig(("op", "ac dec 20 1 1e5"), outputs=("v(out)",)),
    measurements=[
        Measurement.value("v(out)", name="dc"),
        Measurement.magnitude("v(out)", 1e3, name="gain_1k"),
    ],
    seed=42,
)

if __name__ == "__main__":
    campaign.run(workers=8)
    dataset = campaign.dataset()
```

`Campaign` is the class that `FaultCampaign` names; `samples` is the argument that a
fault study calls `samples_per_fault`. Every row of the dataset has `fault_id ==
"healthy"`. See [Normal variation](variation.md) for the distributions.

## The distribution of a measurement

```python
dataset.statistics(quantiles=(0.001, 0.01, 0.5, 0.99, 0.999))
```

One row per measurement (and per fault and operating condition, when there are
several), with:

- `n` and `n_failed`: the simulations used and those that failed. They are counted and
  shown, never dropped silently;
- `mean`, `std`, `minimum`, `maximum`, and the interval of the mean;
- each quantile with a distribution-free interval. A quantile that the sample cannot
  support is left empty: the smallest of 100 values is not an estimate of the 0.1 %
  quantile.

`spicefault.statistics.ecdf(dataset.samples["gain_1k"])` gives the empirical
distribution as two arrays, to plot with any tool.

## Yield

```python
limits = [
    Specification("dc", minimum=0.4965, maximum=0.5035),
    Specification("gain_1k", minimum=0.146),
]
dataset.yield_report(limits)
```

One row per specification and one for all of them:

| Column | Meaning |
|---|---|
| `n`, `n_failed` | Circuits simulated, and simulations that failed |
| `passed`, `yield`, `ci_low`, `ci_high` | Over the simulated circuits, with a binomial interval (`interval="clopper-pearson"` for the exact one) |
| `yield_min`, `yield_max` | The yield over all circuits if every failed simulation is counted as not compliant, and as compliant |
| `only_this` | Circuits that fail this specification and no other: what relaxing it alone would recover |
| `mean`, `std`, `minimum`, `maximum` | Of the specified quantity |
| `margin_sigma` | Standard deviations from the mean to the nearest limit |

The limits are not part of the simulation. Calling `yield_report` with other limits
simulates nothing and writes nothing; `dataset.label(limits)` stores them and writes the
pass/fail columns (see [Datasets](datasets.md)), and they are then the default of
`yield_report()`. With several operating conditions the yield is that of the drawn
circuits, each quantity read in the condition that measures it.

## What a finite sample can say

A yield from a sample is an estimate with an interval, and it bounds the true yield
without proving it.

```python
from spicefault.statistics import samples_for_half_width, zero_failure_bound

zero_failure_bound(1000)                    # 0.003: no failure in 1000 circuits
samples_for_half_width(0.005, p=0.95)       # 7299 circuits for a yield near 0.95 +- 0.005
```

- No failure in $n$ circuits means a failure probability below about $3/n$, not zero.
- `margin_sigma` describes the sample. For a distribution that is not normal, such as
  the bounded one that uniform tolerances give, it is a poor guide to the fraction
  beyond a limit; the yield is the measure of that.
- The intervals assume circuits drawn independently and at random. They are refused
  for a dataset of designed samples, such as tolerance corners.

## Fewer simulations: Latin hypercube and Sobol sampling

By default every circuit is drawn from its own independent stream. `sampling="lhs"` or
`"sobol"` instead chooses the circuits of a population jointly, so that they cover the
tolerances more evenly:

```python
campaign = Campaign(circuit, out_dir="data/rc_lhs", samples=256, sampling="lhs", ...)
```

On the RC low-pass of the validation, at equal numbers of simulations and over 16
repetitions (`python -m benchmarks.sampling.run`):

| Simulations | Method | Error of the mean gain | Error of the yield |
|---|---|---|---|
| 64 | random | 4.8e-3 | 0.026 |
| 64 | Latin hypercube | 2.0e-4 | 0.019 |
| 64 | Sobol | 2.1e-4 | 0.014 |
| 256 | random | 1.8e-3 | 0.017 |
| 256 | Latin hypercube | 6.3e-5 | 0.010 |
| 256 | Sobol | 6.1e-5 | 0.003 |

The mean is estimated about 25 times better; a proportion such as the yield gains less,
between a quarter and four fifths of the random error here. The price:

- **No interval that assumes independent samples.** The circuits of such a design are
  not independent, so `statistics` and `yield_report` give the estimates and leave
  their intervals empty, and `analysis` refuses the dataset. The uncertainty comes from
  repeating the campaign with other seeds:

    ```python
    from spicefault.statistics import replicated_interval

    yields = [campaign_with(seed).dataset().yield_report(limits).loc["all", "yield"]
              for seed in range(8)]
    replicated_interval(yields)      # mean, low, high
    ```

- **Every variation needs a quantile function.** Custom, joint, catalogue, correlated
  and lot variations are refused.
- **Sobol needs SciPy** (`pip install "spicefault[sampling]"`), works best with a number
  of samples that is a power of two, and its points depend on the version of SciPy,
  which the dataset records. The Latin hypercube needs nothing.

The design of a population is a function of the seed, the population and its size, so
the dataset is still the same for any number of workers, and each fault has its own
design. The method is recorded in the metadata of the dataset.

The definitions and estimators are in
[Statistics of a population](../STATISTICS.md), and their check against circuits with a
known distribution in [Validation of the statistics](../validation.md).
