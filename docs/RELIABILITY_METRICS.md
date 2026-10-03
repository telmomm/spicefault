# Reliability metrics

Phase 0 deliverable 3 of 4. Status: the metrics marked for the first release in §11 are implemented in `spicefault.reliability` (Phase 6). An example of each one, on an RC filter, is in `examples/filter/reliability.py`.

This document defines the quantities `spicefault` computes from a campaign: what each one means, how it is estimated, with what uncertainty, and which ones belong to the first release. Notation follows [SCIENTIFIC_SCOPE.md](SCIENTIFIC_SCOPE.md) §2.

Several definitions generalise what the ECG repository already does in `ambiguity.py` and `evaluation.py`; where this document departs from that code, it says so.

## 1. Data

A campaign yields, for each fault condition $k = 0, \dots, K$ (0 is healthy) at a fixed tolerance setting and operating condition $u$:

- $N_k$ samples requested, of which $n_k$ succeeded and $N_k - n_k$ did not, each with a status;
- for each successful sample $i$, a feature vector $z_{k,i} \in \mathbb{R}^d$ and, if the application defines specifications, a compliance flag $c_{k,i}$.

Every metric below is conditional on the feature set, the measurement model $M$, the tolerance setting and $u$. These four must be reported with any figure.

## 2. General rules

**Failed simulations.** Metrics are computed on successful samples and always reported with $n_k / N_k$. Failures are not missing at random: a fault that prevents convergence is probably a severe one. When $n_k < N_k$, proportions are also reported as bounds, counting all failed samples first as detected and then as undetected.

**Reference population.** Thresholds are derived from one part of the healthy samples and the false-alarm rate is measured on another (`healthy_split`, half and half by default). The ECG code computes both on the same samples. That estimate stays close to the target whatever the true rate is, because the limits were placed on those very samples; on new healthy samples the rate is higher when the limits rest on few samples. With 60 healthy samples, two features and a target of 0.05, the in-sample estimate averages about 0.05 and the held-out one is more than 0.02 higher (this is a unit test).

**Uncertainty.** Every proportion carries a 95 % interval (Wilson by default, Clopper–Pearson where a conservative bound is needed). Other statistics carry a percentile bootstrap interval over samples.

**Robust statistics.** Centre and spread are the median and $\hat\sigma = \text{IQR}/1.349$ by default. Faults that saturate an output give heavy-tailed clouds, for which mean and standard deviation hide an obvious shift; this was observed in the ECG study.

## 3. Detectability

Detectability is the probability that a fault produces an observable change. It is not a property of the fault alone: it depends on a decision rule. Three views are defined, from most to least dependent on a specific rule.

### M1. Detection probability at a fixed false-alarm rate

For a decision statistic $T(z)$ and a threshold $\tau_\alpha$ such that $P(T > \tau_\alpha \mid \varnothing) = \alpha$:

$$P_D(k; \alpha) = P\big(T(z) > \tau_\alpha \mid f_k, u\big), \qquad \hat P_D = \frac{1}{n_k} \sum_i \mathbb{1}[T(z_{k,i}) > \tau_\alpha].$$

Default rule, the limit test: a sample is flagged when any feature falls outside the central interval of the healthy calibration samples, each interval taken at tail probability $\alpha / (2d)$ so that the joint false-alarm rate is about $\alpha$. The realised false-alarm rate $\hat\alpha$ is measured on held-out healthy samples and reported next to $\hat P_D$.

The rule is an argument, so that an application can pass its own detector.

### M2. Standardised shift

Per feature $j$, with a floor $\sigma_{\text{floor},j}$ equal to the measurement noise of that feature:

$$d_{k,j} = \frac{\lvert \tilde z_{k,j} - \tilde z_{0,j} \rvert}{\sqrt{\tfrac12\big(\max(\hat\sigma_{k,j}, \sigma_{\text{floor},j})^2 + \max(\hat\sigma_{0,j}, \sigma_{\text{floor},j})^2\big)}}, \qquad d'_k = \max_j d_{k,j}.$$

Univariate and conservative: a fault invisible in every single feature may still be visible jointly. The floor prevents numerically identical clouds from appearing infinitely separated. It is a map of where the information is, not a detection probability.

### M3. Area under the ROC curve

For a scalar statistic $T$: $\text{AUC}_k = P(T_k > T_0) + \tfrac12 P(T_k = T_0)$, estimated by the Mann–Whitney statistic. Threshold-free, bounded and insensitive to tails. When reported as the best single feature, $\max_j \max(\text{AUC}_{k,j}, 1 - \text{AUC}_{k,j})$, the selection over $d$ features is optimistic and the number of features must be stated.

### M4. Overlap coefficient (deferred)

$\text{OVL} = \int \min(p_k, p_0)$. Deferred: it needs a density estimate, and the choice of estimator would become part of the metric. For equal-variance normal clouds it is a function of the standardised shift, $\text{OVL} = 2\Phi(-d/2)$, which can be reported as an approximation with that assumption stated.

## 4. Functional failure and diagnostic coverage

These require a compliance predicate from the application.

### M5. Failure probability given a fault, and yield

$$\pi_k = P(c = 0 \mid f_k, u), \qquad Y = P(c = 1 \mid \varnothing, u).$$

$\pi_k$ is the functional severity of a fault condition as a probability. $Y$ is the yield of the healthy population.

### M6. Diagnostic coverage, escape rate, false-reject rate

With weights $w_k \geq 0$ over fault conditions:

$$\text{DC} = \frac{\sum_k w_k \, P(\text{detected}, c = 0 \mid f_k)}{\sum_k w_k \, \pi_k}, \qquad \text{escape rate} = 1 - \text{DC},$$

$$\text{false-reject rate} = P(\text{detected} \mid c = 1).$$

Default weights are equal, which makes DC a statement about the catalogue, not about field behaviour. If the user supplies failure-mode probabilities from a cited source, DC approximates the fraction of dangerous failures detected, in the sense used in functional safety. The weighting must be reported with the figure. The false-reject rate is computed over all compliant samples, faulty or not, and its composition must be stated: `diagnostic_coverage` returns how many of them are healthy. Healthy samples used to set the thresholds are left out. The interval of DC is a bootstrap over the samples of each fault.

### On `failure_rate()`

The project specification lists `analysis.failure_rate()`. In reliability engineering a failure rate is an occurrence rate in time, which a fault-injection campaign cannot estimate. The method is implemented as `failure_probability()` and returns $\pi_k$; there is no `failure_rate()`. See open decision 1.

## 5. Sensitivity

### M7a. Local sensitivity

At the nominal circuit, by central differences with relative step $h$ (default 1 %):

$$S_{j,i} = \frac{m_j(x_i(1+h)) - m_j(x_i(1-h))}{2h},$$

reported per 1 % change, and normalised by the healthy spread, $Z_{j,i} = 0.01\, S_{j,i} / \hat\sigma_{0,j}$. A value of 1 means that a 1 % deviation of the component moves the feature by one healthy standard deviation. Two derived quantities, both from the ECG code:

- **testability rank**: the number of singular values $\sigma_r$ of $Z$ with $\sigma_r \cdot \Delta \geq \tau$, for a deviation of $\Delta$ per cent and threshold $\tau$ (defaults 10 and 3). It bounds how many components can be told apart by deviations of that size;
- **collinear groups**: components whose columns of $Z$ have absolute cosine above a threshold (default 0.99), which cannot be distinguished by small deviations.

Local sensitivity is valid only near the nominal point and is noise-free.

### M7b. Severity response and minimum detectable magnitude

For a graded fault on one component, the curve $s \mapsto P_D(s; \alpha)$ over the simulated severities. The **minimum detectable magnitude** is the smallest simulated magnitude with $\hat P_D \geq 1 - \beta$ (default $\beta = 0.1$), reported with the grid, since it cannot be finer than the grid. This is the quantity $\partial y / \partial s$ of the specification made operational when the response is not linear.

## 6. Robustness

### M8. Detectability under tolerance

Let $\kappa$ scale all tolerances ($\kappa = 1$ is the declared design). The robustness curve of a fault condition is

$$\kappa \mapsto P_D(k; \alpha, \kappa),$$

with the healthy reference recomputed at each $\kappa$. Two summaries:

- **loss of detectability**: $P_D(k; \kappa_\text{ref}) - P_D(k; \kappa)$;
- **critical tolerance**: the largest simulated $\kappa$ with $\hat P_D \geq 1 - \beta$.

At $\kappa = 0$ the populations are degenerate unless measurement noise is applied, so the measurement model is mandatory for this metric and the standardised shift relies on its noise floor.

The same curve applied to the healthy population gives yield against tolerance, $Y(\kappa)$.

## 7. Separability

### M9a. Pairwise separation

$d'_{k,l}$ as in M2 between two fault conditions. Multivariate alternative, the Mahalanobis distance with pooled covariance:

$$D^2_{k,l} = (\mu_k - \mu_l)^\top \Sigma_\text{pooled}^{-1} (\mu_k - \mu_l).$$

It uses correlations that the univariate measure ignores, but it relies on means and covariances and is distorted by saturating faults. Report it only alongside the robust univariate measure.

### M9b. Ambiguity structure

From the graph that links conditions with $d' < \tau$ (default 3):

- **undetectable conditions**: those linked directly to healthy;
- **ambiguity groups**: connected components. Linking is transitive, so chains merge and the groups are an upper bound on ambiguity;
- **confusable components**: for each component, the others with at least one linked condition. Not transitive; it answers which components a case could be mistaken for.

Classifier confusion matrices measure the same thing downstream and belong to the application, not to this framework.

## 8. Coverage

### M10. Structural coverage of a campaign

$$\text{coverage} = \frac{\lvert \text{simulated conditions} \rvert}{\lvert \text{fault universe} \rvert},$$

with the coverage matrix (components × fault types) and the list of exclusions and reasons (FAULT_MODEL.md §8). It measures what was examined, not what was detected. The fraction of simulated conditions with $\hat P_D \geq 1 - \beta$ is reported separately as detection coverage, and must not be called fault coverage without that qualification.

## 9. Operating conditions

### M11. Dependence on operating conditions

Any metric above as a function of $u$ over the simulated set $U$. Summaries: the worst case $\min_{u} P_D(k; u)$, the range over $U$, and the conditions under which a fault that is detectable at the nominal condition stops being so.

## 10. Sample size

Indicative figures for planning, at 95 % confidence:

| Situation | Consequence |
|---|---|
| Proportion from $n = 200$ samples | Interval half-width up to about ±0.07 |
| No miss in $n = 200$ samples | Miss probability below about 1.5 % |
| Proportion from $n = 1000$ | Half-width up to about ±0.03 |
| Limit test with $\alpha = 0.01$, a few tens of features, 5,000 healthy samples | The per-feature tail $\alpha/(2d)$ holds about one sample, so each limit is close to the sample extreme and poorly estimated |

The last row is a reason to measure the false-alarm rate on held-out healthy samples, and to prefer a larger healthy population than fault populations.

A design choice affects all comparisons between conditions: whether replica $r$ of every condition uses the same draw $\theta_r$ (common random numbers) or an independent one. Common draws give paired comparisons and lower variance for differences; independent draws are what the ECG baseline uses. The seeding scheme must support both; see EXPERIMENT_PLAN.md §2.

## 11. First release

| Metric | State | Where |
|---|---|---|
| M1 detection probability | Implemented | `ReliabilityAnalysis.detectability`, `LimitTest` |
| M2 standardised shift | Implemented | `standardised_shift` |
| M3 AUC | Implemented | `auc` |
| M4 overlap | Deferred | |
| M5 failure probability, yield | Implemented | `failure_probability` |
| M6 diagnostic coverage, escape, false reject | Implemented | `diagnostic_coverage` |
| M7a local sensitivity, rank, collinear groups | Implemented | `local_sensitivity`, `normalised_sensitivity`, `testability_rank`, `collinear_groups` |
| M7b severity response | Implemented | `severity_response`, `minimum_detectable` |
| M8 robustness | Implemented | `robustness`, with campaigns at `VariationSet.scaled` tolerances |
| M9a pairwise separation (univariate) | Implemented | `separation` |
| M9a Mahalanobis | Deferred | |
| M9b ambiguity structure | Implemented | `ambiguity` |
| M10 coverage | Implemented | `FaultUniverse.coverage`, `coverage_matrix` |
| M11 operating-condition dependence | Implemented | `ReliabilityAnalysis.by`, `detectability_across` |

Each implemented metric has its definition in its documentation, unit tests against populations with known answers (normal clouds, tables with known counts, a resistive divider), and an example. M2, M5, M6, M7a, M9 and the limit test of M1 are also checked against the ECG code they were generalised from, on the dataset `data/v1`: the results are identical.

What the tests do not establish: that the limit test is a good detector, that equal weights are meaningful, or that the thresholds (α, β, τ) are the right ones. Those are choices of a study.

## 12. Open decisions

1. **`failure_rate()` is implemented as `failure_probability()`.** To confirm, or to add the other name as an alias.
2. **Default false-alarm rate** $\alpha$ (proposed 0.01) and miss level $\beta$ (proposed 0.1).
3. **Default separation threshold** $\tau = 3$, as in the ECG study.
4. **Common or independent random numbers** for comparisons between fault conditions. Both are implemented (`seeding="common"` against `"positional"` or `"content"`); independent is the default and is needed for Phase 1 equivalence. Still to decide: which one Experiments E to G use. Operating conditions and tolerance levels are always compared on the same circuits.
