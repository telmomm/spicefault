# Experiment plan

Status: revised for validation on generic circuits. The instruments and the validation circuits exist. The runs stored in `results/` and `benchmarks/results/` were made with version 0.1.0, on a machine that was not idle, and with earlier netlists of the biquad and the regulator: none of them is to be reported, and all are to be run again with the frozen release.

This document fixes the protocol of the experiments that the manuscript will report: design, outputs, acceptance criteria and numerical tolerances. Research questions are those of [SCIENTIFIC_SCOPE.md](SCIENTIFIC_SCOPE.md) §4, validation circuits those of its §8, metrics those of [RELIABILITY_METRICS.md](RELIABILITY_METRICS.md).

Thresholds marked *proposed* are starting values. They are to be confirmed or replaced when the experiment is first run, and the final values reported with the results.

## 1. Circuits and workloads

| Circuit | Used in | Role in the experiments |
|---|---|---|
| Resistive divider, RC low-pass | A | Closed-form answers for correctness and for the metric estimators |
| Sallen–Key band-pass filter | A, B, D, E, G | Small benchmark; comparison task of the state-of-the-art study |
| Four-op-amp biquad high-pass filter | A, C, D, E, G | Larger benchmark; workload of the scalability measurements; ambiguity analysis |
| Linear voltage regulator | A, D, E, F | Device-level models; operating conditions, temperature included |

Every circuit comes with its netlist, its fault rules, its measurements and its specification limits, in `validation/`. The Sallen–Key filter is the circuit of Aminian and Aminian (2000, doi:10.1109/82.823545, Fig. 3) and the biquad that of Aminian, Aminian and Collins (2002, doi:10.1109/TIM.2002.1017726, Fig. 2), with their component values and designators; the regulator has the SPICE models published by the manufacturer of its devices, named in its netlist. The specification limits of the two filters are the range of the central 99 % of healthy circuits at the declared tolerances: with the tolerances of the literature these filters have no tight specification to meet (the peak gain of healthy Sallen–Key circuits goes from 1 to 23). The limits of the regulator are engineering limits. All were fixed before any fault campaign.

The reference for what the framework costs is a script written directly against ngspice for the same task (Experiment C and the comparison of SCIENTIFIC_SCOPE.md §7). It is written once, frozen, and kept in the repository.

## 2. Reproducibility levels and seeding

| Level | Statement | Compared |
|---|---|---|
| L1 | Identical sample definitions | Fault, realised parameters, operating condition, per-sample seed: exact equality |
| L2 | Equivalent outputs on one platform | Measurements and waveforms, same machine and simulator version |
| L3 | Equivalent outputs across platforms | Same, across operating systems or simulator versions |

L1 is a property of the framework and must hold exactly. L2 and L3 depend on the simulator; the framework's role is to measure and report the differences. `Dataset.reproduce` is the instrument for the three: it simulates samples of a stored dataset again and reports, per sample, whether the definition and the drawn values are identical (L1) and the largest difference in measurements and waveform (L2 on the same platform, L3 on another).

Seeding. Each sample has its own random stream, derived from the master seed and a key. `Experiment` has three schemes for the key (`spicefault.experiments.seeding`):

| Scheme | Key of the stream | Use |
|---|---|---|
| `positional` | (fault index, replica) | The stream of a fault depends on its place in the fault list: adding or reordering faults changes the samples of the others. Kept for datasets generated this way |
| `content` | (hash of the fault identifier, replica) | A condition has the same samples in any experiment that contains it. Used in the experiments of this plan |
| `common` | (replica) | Common random numbers: every fault, and the healthy case, on the same drawn circuits |

The operating condition is never part of the key: one drawn circuit is simulated under every operating condition, so comparisons between operating conditions are always paired.

With `common`, comparisons between fault conditions are paired as well, which lowers the variance of differences between conditions. The price is that the conditions are no longer independent samples, and confidence intervals that assume independence between two conditions do not apply to their difference; a paired analysis must be used instead.

Tolerance levels (Experiment E) are compared on the same circuits by construction: `VariationSet.scaled` multiplies every spread and uses the same random numbers, so each sample at one tolerance level is the same circuit, with proportionally smaller deviations, at another.

## 3. Numerical tolerances

*Proposed.*

| Comparison | Quantity | Criterion |
|---|---|---|
| Same deck, direct ngspice against `spicefault`, same machine | Raw vectors | Exact equality expected. Any difference is investigated, not tolerated |
| Closed-form circuits | Measurements against theory | Within the simulator tolerances in force (`reltol`, `abstol`, `vntol`), which are recorded |
| L1 | Sample definitions | Exact equality |
| L2, 1 worker against $N$ workers | Measurements (float64) | Exact equality expected; accept relative difference ≤ 1e-12 |
| L2 | Waveforms (stored as float32) | Exact equality |
| L3 | Measurements | No criterion set in advance: measure the distribution of differences and report it, together with the number of statuses and labels that change |

## 4. Environment and timing protocol

Recorded for every run: `spicefault` version and commit, Python and dependency versions, simulator name and version, operating system, processor model, number of physical and logical cores, number of workers, chunk size, start and end time, and whether the run was resumed. The manifest of a dataset and the result files of the benchmarks hold these.

Timing: fixed workload; at least 5 repetitions per configuration; report median and range; machine otherwise idle and on mains power; one discarded warm-up run; output written to local disk. A machine that is running anything else gives timings that must not be reported.

## 5. Experiments

### A. Correctness

- **Question:** does the framework change what the simulator computes? (RQ3)
- **Design:** for each validation circuit, a set of decks covering every fault type is run directly with `ngspice -b` and through `spicefault`. For the closed-form circuits, results are also compared with theory.
- **Output:** maximum absolute and relative difference per vector.
- **Acceptance:** §3, first two rows.
- **State:** the unit tests do this for the closed-form circuits. For the validation circuits it is automated as `python -m benchmarks.correctness.run`: the healthy circuit and every fault are simulated through `spicefault` and with `ngspice -b` launched apart, and the vectors are compared bit by bit with a raw-file reader of its own. A first run with version 0.4.0 gave 219 decks and 408,938 values, all identical; it is to be run again with the frozen release.

### B. Reproducibility

Automated as `python -m benchmarks.reproducibility.run`.

Two things the runs made so far do not cover: L3 has not been measured, and every sample of both campaigns ended with `SUCCESS`, so the reproducibility of the failure statuses is not exercised by the validation circuits. A workload with faults that do not converge is needed for the second.

- **Question:** are results independent of parallelism and repeatable? (RQ3)
- **Design:** the same campaign with a fixed seed run with 1 worker and with the maximum available; run twice with the same worker count; with another chunk size; interrupted and resumed; then samples simulated again from the dataset folder. A subset rerun on a second platform for L3.
- **Size:** *proposed* 5,000 samples, covering every fault type.
- **Output:** L1 comparison; distribution of L2 and L3 differences; number of changed statuses.
- **Acceptance:** L1 exact in all cases, including the resumed run; L2 as in §3.

The project specification asks for 16 workers. The development machine has 8 cores; 16 workers there would test oversubscription, not parallelism. A 16-core run needs another machine (open decision 1).

### C. Parallel scalability

Automated as `python -m benchmarks.scalability.run`.

- **Question:** how does throughput scale, and what does the abstraction cost? (RQ4)
- **Design:** fixed workload at 1, 2, 4, 8 workers, and 16 where the hardware allows. The same task with the direct script of §1.
- **Size:** *proposed* 2,000 samples of the biquad filter.
- **Output:** wall time, simulations per second, $S(N) = T_1 / T_N$, $E(N) = S(N)/N$, peak memory, bytes written, and a breakdown of the time of one sample into netlist generation, simulator process, output parsing, measurement and storage.
- **Acceptance:** none on speed-up, which is reported as measured. *Proposed* for overhead: `spicefault` throughput at least 0.95 of the direct script at equal worker count.
- **Caveat:** the development machine has 4 performance and 4 efficiency cores, so efficiency beyond 4 workers falls for hardware reasons. For the paper the curve should be measured on a machine with homogeneous cores.
- **Found so far** (version 0.1.0, development machine, not to be reported): throughput relative to the direct script of 0.995, 0.954, 0.972 and 0.977 at 1, 2, 4 and 8 workers in one run and 1.059, 0.959, 1.055 and 0.938 in another, so the proposed 0.95 was met in seven of eight configurations; the simulator process is 92 to 93 % of the time of a sample. The direct script exists for the Sallen–Key campaign only, so the comparison cannot be made on the biquad without writing another.

Recovery time, listed in the project specification, is measured as the time to launch again a campaign whose chunks are complete, and the number of completed simulations that are repeated (none: an interruption loses at most the chunk in progress).

### D. Fault coverage

Automated as `python -m benchmarks.fault_coverage.run`.

- **Question:** is coverage systematic and auditable? (RQ1, RQ7)
- **Design:** generate the fault universe of each validation circuit from its rules; build the campaign; produce the coverage matrix and the exclusion list.
- **Output:** coverage matrix per circuit; counts of components × fault types × magnitudes × operating conditions; structural coverage (M10); components outside the fault model; parametric faults partly inside the tolerance band.
- **Acceptance:** every pair in the universe is simulated or excluded with a reason.
- **Gap:** no fault is excluded in any validation circuit, so the exclusion record is shown only by the unit tests. One justified exclusion in a validation circuit would show it in the evaluation.

### E. Impact of variability

- **Question:** how does tolerance degrade detectability and diagnostic coverage? (RQ2, RQ6)
- **Design:** every tolerance scaled by 0, 0.2, 1 and 2 times its declared value, same fault list, same measurement model. The filters are declared with 5 % resistors and 10 % capacitors, as in the diagnosis literature, so the scales are 0, 1, 5 and 10 % for the resistors. Healthy reference recomputed at each level. The same circuits at every level (§2).
- **Size:** *proposed* 5,000 healthy and 200 per fault condition at each level.
- **Output:** M1, M2 and M8 per fault condition with intervals; yield and failure probability against tolerance (M5); diagnostic coverage against tolerance (M6); the faults whose detection probability falls below $1 - \beta$ and the tolerance at which it happens.
- **Acceptance:** none; this is the main reliability result. The zero level requires the measurement model, otherwise the populations are degenerate.

### F. Impact of operating conditions

- **Question:** does detectability depend on operating conditions? (RQ2)
- **Circuit:** the linear regulator, whose device models depend on temperature.
- **Design:** one factor at a time around the nominal condition (input voltage, load current, temperature), then the corners.
- **Output:** M11: detection probability per fault and condition, worst case, and the conditions under which a fault detectable at the nominal condition stops being so.
- **Caveat:** a circuit made of behavioural models and passives without temperature coefficients does not change with temperature. Reporting that as robustness to temperature would be wrong, which is why this experiment uses device-level models.

### G. Fault separability

- **Question:** how distinguishable are faults from each other? (RQ5)
- **Circuit:** the biquad filter, at the declared tolerance, with the data of Experiment E.
- **Design:** M9 once on scalar measurements and once on waveforms. For waveforms the feature vector is the sampled response, reduced to a fixed number of components by a declared method.
- **Output:** pairwise separation matrix, undetectable conditions, ambiguity groups, confusable components; comparison with the collinear groups and the testability rank predicted by local sensitivity (M7a).
- **Acceptance:** none. The comparison between what local sensitivity predicts and what the campaign finds is the result.

### H. Machine learning as downstream application

- **Question:** are the generated data usable for automated diagnosis?
- **Design:** none in this manuscript. The application study of SCIENTIFIC_SCOPE.md §9 trains diagnosis models on data generated with the library; it is cited as the evidence.
- **Optional:** a short example with a standard classifier on one validation circuit, to show `Dataset.to_ml`. It would be an example of use, not a result.

## 6. What remains, in order

| Step | Content | Long runs |
|---|---|---|
| 1 | Literature search and positioning: first version in RELATED_WORK.md; full texts pending | None |
| 2 | The three validation circuits: done, in `validation/`, with the schematics of the two filters and the device models of the regulator taken from cited sources | Short checks only |
| 3 | The direct ngspice script for the comparison task: done, `validation/direct/` | Short checks only |
| 4 | Scripts of Experiments E, F and G: done, `validation/experiments/` | Short checks only |
| 5 | All campaigns and benchmarks, launched together on an idle machine: commands in RUNBOOK.md | A, B, C, D, E, F, G |
| 6 | State-of-the-art comparison table: in COMPARISON.md; spicelib and PySpice run | None |
| 7 | Release: licence, citation file, documentation, archive with DOI | None |
| 8 | Manuscript | None |

Steps 1 to 4 need no long simulation and can be done in any order; the methods sections of the manuscript can be drafted alongside them.

## 7. Threats to validity

| Threat | Mitigation |
|---|---|
| Results specific to one ngspice version | Record the version; L3 comparison on a second version |
| Fidelity of the device models | Declared; manufacturer models with their source. The zener model alone drifts +3.13 mV/K where its data sheet gives -3.5 to +0.2 mV/K, so the netlist adds a declared correction that brings it to the middle of that range; the temperature behaviour of the two transistor models is not checked against their data sheets |
| Open and short resistances chosen arbitrarily | Recorded; sensitivity of the results to `r_open` checked on one circuit |
| Detectability depends on the feature set and the decision rule | Both reported with every figure; at least two rules compared |
| Equal weights across fault conditions | Stated; weighted variant if failure-mode data with a source are available |
| Convergence failures concentrated in severe faults | Failures counted and bounds reported (RELIABILITY_METRICS.md §2) |
| Timing on a machine with heterogeneous cores | Second machine for the scalability figure |
| Direct script and framework written by the same author | Script frozen before the measurements; both kept in the repository |
| Validation circuits chosen by the authors | Two of them taken from the literature; all specifications declared before the campaigns are run |
| Generalisation shown on few circuits | Stated as a limitation |

## 8. Open decisions

1. **Second machine** with 16 or more homogeneous cores for experiments B and C, or restrict the claims to 8 workers.
2. **Random numbers between fault conditions** in Experiments E to G: independent (`content`) or common. Independent is proposed, because the intervals of the metrics assume it.
3. **Size of experiment E:** four tolerance levels on all three circuits, or on one.
4. **Measurement model** for the zero-tolerance level: the noise and resolution assumed for each circuit.
