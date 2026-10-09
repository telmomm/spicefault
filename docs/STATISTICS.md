# Statistics of a population

Definitions and estimators of the quantities that describe a population of drawn
circuits. They are computed by `spicefault.statistics`, and by `Dataset.statistics` and
`Dataset.yield_report`, from the samples of a dataset; nothing is simulated again. The
quantities that involve faults are in [reliability metrics](RELIABILITY_METRICS.md).

## 1. Data and assumptions

A campaign without faults draws $n$ circuits $\theta_1, \dots, \theta_n$ independently
from the variations and measures each one. The estimators below assume that the
circuits are **independent and drawn at random**. They are refused for a dataset whose
samples were designed, such as tolerance corners: a proportion over chosen circuits is
not a probability.

A simulation that fails gives no measurement. It is counted and reported with every
statistic, never dropped silently: a statistic over the successful simulations
describes the circuits that could be simulated, and that is all it describes.

## S1. Mean and spread

$\bar m$ and $s$ (with $n - 1$) of a measurement over the $n$ successful samples. The
interval of the mean is $\bar m \pm z\, s / \sqrt{n}$, which holds when $n$ is large
enough for the mean to be normally distributed; it says nothing about where a single
circuit falls.

## S2. Quantiles

The estimate of the quantile $\xi_q$ interpolates linearly between order statistics.
Its interval is distribution-free. The number of samples below $\xi_q$ is
$\text{Binomial}(n, q)$, so the ranks $l < u$ with

$$P(X \le l - 1) \le \alpha/2, \qquad P(X \ge u) \le \alpha/2, \qquad X \sim \text{Binomial}(n, q)$$

give $P(x_{(l)} \le \xi_q \le x_{(u)}) \ge 1 - \alpha$ for any continuous distribution.
The interval is conservative, because the ranks are whole numbers.

When no such ranks exist the quantile is **not estimable** from the sample, and the
estimate and its interval are reported as missing. This happens in the tails: the
smallest of $n$ values lies below $\xi_q$ with probability $1 - (1 - q)^n$, so 100
samples cannot bracket the 1 % quantile at 95 % confidence, and the smallest of them is
not an estimate of it.

## S3. Yield

With a set of specification limits, $c_i = 1$ if circuit $i$ meets all of them:

$$Y = P(c = 1 \mid \varnothing), \qquad \hat Y = \frac{k}{n}, \quad k = \sum_i c_i.$$

The interval is binomial: Wilson by default, Clopper-Pearson (exact, conservative) on
request. The same holds for each specification alone. A value that could not be
computed in a successful simulation is a violation.

Failed simulations are not known to pass or to fail. With $n_f$ of them, the report
gives $\hat Y$ over the $n$ successful ones and the two bounds over all $n + n_f$:
$k / (n + n_f)$ if every failed one is counted as not compliant, and $(k + n_f) / (n +
n_f)$ if every one is counted as compliant.

**A finite sample bounds the yield and does not prove it.** With no failure in $n$
circuits the failure probability is below

$$1 - (1 - \gamma)^{1/n} \approx \frac{3}{n} \quad (\gamma = 0.95),$$

not zero. A half-width $h$ for a yield near $p$ needs about $z^2 p (1 - p) / h^2$
circuits.

## S4. Margin

For a specification with limits $[a, b]$ on a quantity of mean $\bar m$ and spread $s$:

$$\text{margin} = \min\!\left(\frac{\bar m - a}{s},\ \frac{b - \bar m}{s}\right),$$

with one term when a bound is open. It describes how many standard deviations separate
the centre of the sample from the nearest limit. It is not a guarantee, and for a
distribution far from normal it is a poor guide to the fraction beyond the limit, which
is what the yield measures.

## Sampling other than independent draws

The intervals of S1 to S3 assume independent samples. The circuits of a Latin hypercube
or a Sobol design are not independent, and those intervals are not given for them.
Their uncertainty comes from $k$ independent replications of the whole design, with
different seeds: each gives one estimate $\hat e_i$, the estimates are independent, and

$$\bar e \pm t_{k-1}\, \frac{s_e}{\sqrt{k}}$$

is the interval (`spicefault.statistics.replicated_interval`), with the Student $t$ of
$k - 1$ degrees of freedom. It assumes that the estimates are roughly normal.
