# Scientific scope

Status: revised. The first version took an ECG front-end as its main case study; this one is about the framework alone, validated on generic circuits. The framework is implemented (phases 1 to 8); the validation circuits and the experiments on them are not.

Companion documents: [FAULT_MODEL.md](FAULT_MODEL.md), [RELIABILITY_METRICS.md](RELIABILITY_METRICS.md), [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md), [RELATED_WORK.md](RELATED_WORK.md).

This document fixes what `spicefault` is meant to demonstrate, in terms precise enough that the work can be checked against it. The target venue is *IEEE Transactions on Reliability*, so the contribution is a reliability methodology; the software is the instrument that makes it reproducible.

## 1. Problem statement

Simulation-based fault studies of electronic circuits are usually built as project-specific scripts around a SPICE simulator. The scripts work, but four things are decided implicitly and are rarely reported:

1. what counts as a fault, as opposed to normal component variation;
2. which faults were simulated out of those that could have been;
3. what happened to simulations that failed;
4. which random numbers produced which sample.

Without these, the resulting fault-response distributions cannot be audited or regenerated, and figures such as detectability or diagnostic coverage cannot be compared between studies.

`spicefault` treats the combination of circuit, normal variation, operating condition and fault as one structured experiment, and defines the reliability quantities that are computed from it. It is independent of the kind of circuit: nothing in it is specific to an application domain.

## 2. Formal setting

| Symbol | Meaning |
|---|---|
| $C$ | Circuit: topology and nominal parameter vector $x_0$ |
| $\theta \sim P_\theta$ | Realisation of the parameters under normal variation (manufacturing tolerance, healthy spread of active devices and of the environment) |
| $u \in U$ | Operating condition (supply, temperature, stimulus, load) |
| $f \in \mathcal{F} \cup \{\varnothing\}$ | Fault; $\varnothing$ is the fault-free case |
| $I_f$ | Injection operator: $I_f(C, \theta)$ is the netlist of the realised circuit with the fault applied |
| $y = S(I_f(C,\theta), u)$ | Noise-free simulator output (vectors and waveforms) |
| $m = g(y)$ | Measurements: deterministic functionals of $y$ |
| $z = M(m; \varepsilon)$, $\varepsilon \sim P_\varepsilon$ | Measurements as an instrument would observe them (noise, quantisation, clipping) |
| $c = \mathbb{1}[m_\text{spec} \in A]$ | Compliance with a set of specification limits $A$, when the application defines one |

The objects of study are the conditional distributions $P(z \mid f, u)$, obtained by marginalising over $\theta$ and $\varepsilon$, and their comparison with $P(z \mid \varnothing, u)$.

Three design choices have scientific consequences:

- **Variation first, then fault.** A fault is injected into an already realised circuit, so every other component keeps its spread. A fault is never simulated on the nominal circuit alone, except in sensitivity analysis.
- **Simulation and observation are separate.** The simulator output is stored noise-free and $M$ is applied afterwards. Measurement noise and resolution are then study parameters that do not require new simulations.
- **Injected fault and functional failure are separate.** $f \neq \varnothing$ states what was injected; $c = 0$ states that the circuit no longer meets its specification. An injected fault does not imply a failure, and how often it does is a result of the experiment.

## 3. Terminology

Working definitions, used consistently in code, documentation and manuscript. They should be checked against IEC 60050-192 and IEEE 2427 before submission.

| Term | Definition here |
|---|---|
| Defect | Physical, unintended change in a component or connection |
| Fault | Model of a defect: a defined transformation of the netlist (see FAULT_MODEL.md) |
| Fault condition | One fault with all its parameters fixed, e.g. `R11:parametric:+0.2`. The unit of a campaign |
| Failure | The circuit does not meet at least one specification ($c = 0$) |
| Normal variation | Spread of parameters within their declared tolerance; not a fault |
| Healthy population | Samples with $f = \varnothing$ |
| Detection | A decision rule applied to $z$ flags the sample as not healthy |
| Escape | A failed circuit that is not detected |
| False reject | A circuit that is flagged although it has not failed |

## 4. Research questions

Each question is tied to a quantity, an experiment and a criterion. Metric identifiers refer to RELIABILITY_METRICS.md, experiment letters to EXPERIMENT_PLAN.md, circuits to §8.

| RQ | Question | Quantity | Experiment | Evidence that answers it |
|---|---|---|---|---|
| RQ1 Fault coverage | Does a formal fault abstraction give systematic, auditable coverage of the failure modes of a circuit? | Coverage matrix and structural coverage (M10) | D | Every applicable (component, fault type) pair is either simulated or excluded with a recorded reason, for every validation circuit |
| RQ2 Variability | How do tolerance and operating variation change detectability? | Detection probability and standardised shift against tolerance scale and operating condition (M1, M2, M8, M11) | E, F | Detectability curves with confidence intervals; faults whose detectability falls below a stated level are identified |
| RQ3 Reproducibility | Can a campaign be reproduced exactly, or within stated tolerances? | Identity of sample definitions; numerical difference of outputs | A, B | Identical sample definitions for any worker count; output differences within the tolerances of EXPERIMENT_PLAN.md §3 |
| RQ4 Scalability | How efficiently do large campaigns run? | Throughput, speed-up $S(N)$, efficiency $E(N)$, overhead against direct scripting | C | Measured curves with repetitions; overhead of the abstraction against a script written for the same task |
| RQ5 Separability | How distinguishable are different faults from the observable response? | Pairwise separation, ambiguity groups (M9) | G | Ambiguity structure reported for measurements and for waveforms, and compared with what local sensitivity predicts |
| RQ6 Reliability assessment | Do fault-response distributions give useful quantitative evidence of robustness and diagnostic coverage? | Failure probability per fault, diagnostic coverage, escape and false-reject rates (M5, M6), against tolerance | E, F | Figures with confidence intervals that would change a design or test decision: which faults a given set of measurements cannot cover, and at what tolerance it stops covering others |
| RQ7 Generalisation | Does the method apply to different circuits without rewriting the infrastructure? | Circuit-specific code per circuit; framework code changed | all | Circuits of different kind and size run with no change to the framework; what each one needs is its netlist, rules, measurements and specifications |

RQ6 remains the weakest as written: "useful" is not measurable. The criterion above is a proposal and should be sharpened once Experiments E and F have results.

## 5. Claims and non-claims

The manuscript may claim, if the experiments support it:

- a fault abstraction in which every fault compiles to a small set of netlist transformations and carries explicit metadata;
- campaigns whose sample definitions are identical for any degree of parallelism;
- an experimental record in which failed simulations are kept and counted;
- reliability metrics with stated definitions, estimators and confidence intervals;
- the same infrastructure applied to circuits of different kind, with nothing specific to one application domain;
- use in an independent published study (§9).

It must not claim:

- that other tools cannot perform fault injection (the difference is one of scope, see §7);
- a faster simulator: the numerical work is done by ngspice;
- failure rates in time (FIT, MTBF): simulation gives conditional response distributions, not occurrence rates;
- completeness of the fault taxonomy with respect to physical failure mechanisms;
- that simulated evidence replaces physical testing.

## 6. Scope

Included in the first release: SPICE netlists as circuit description; single faults at component level (open, short, parametric, leakage, and composite degradation); tolerance distributions; operating conditions; deterministic sampling; local parallel execution; resumable campaigns; explicit simulation status; measurements; the metrics marked as implemented in RELIABILITY_METRICS.md; dataset export with provenance.

Excluded: a numerical solver; schematic capture; a graphical interface; FMEA or FTA engines; reliability prediction standards; machine-learning models; cluster execution; GPU solving; multiple simultaneous independent faults; intermittent and time-dependent faults.

Two boundaries with the applications that use the framework:

- **Specification compliance is an application concern.** The framework provides the mechanism (measurements, a compliance column, the metrics that use it). The limits and the test set-ups belong to the application.
- **Machine learning is downstream.** The framework exports features, waveforms and labels; models stay in the application.

## 7. Comparison with existing approaches

The comparison is functional and must be reproducible. Nothing in the table below has been verified yet; the entries say what has to be checked, at a fixed version of each tool, before a comparison table can appear in the paper.

| Class | Candidates to examine | What to establish |
|---|---|---|
| Direct scripting around ngspice | A script written for the benchmark task, with no framework | Measured: effort, throughput, reproducibility, behaviour on failure |
| Python/SPICE interfaces | PySpice; spicelib / PyLTSpice | Which of fault abstraction, tolerance analysis, campaign orchestration, resumption and provenance they provide. spicelib is reported to include Monte Carlo, worst-case and failure-mode analyses; this must be read from its documentation and tested |
| Simulator-native statistics | ngspice control-language Monte Carlo; LTspice `.step` with `mc()`; Spectre Monte Carlo | What is expressible, and what record is left of each sample |
| Commercial analog fault simulation | Cadence Legato Reliability, Siemens Tessent DefectSim, Synopsys TestMAX CustomFault | Scope (transistor-level defect-oriented test of integrated circuits, IEEE 2427) and how it differs from board-level reliability experiments. Likely not runnable; compare from documentation and state so |
| Academic fault-injection and diagnosis workflows | To be collected in a dedicated search for SPICE fault-injection frameworks and analog fault-diagnosis datasets | Whether fault definitions, seeds and failed simulations are reported, and whether code is available |

Criteria, one row per tool in the final table: fault abstraction; separation of variation and fault; operating conditions; campaign definition; coverage report; parallel execution; resumption; simulation status; provenance per sample; determinism under parallelism; measurement extraction; reliability metrics; dataset export; backend independence.

Method: one benchmark task (one validation circuit, a fixed fault list, a fixed number of samples) is implemented with direct scripting, with one Python/SPICE interface and with `spicefault`. Reported: lines of task-specific code, whether each criterion is met, and how. Lines of code is a weak measure of effort and is reported as such.

A first literature search is in [RELATED_WORK.md](RELATED_WORK.md). Its main consequences: fault injection in SPICE and the treatment of tolerance are established, with a standard (IEEE Std 2427) and commercial tools for integrated circuits, so the contribution to defend is the reproducible experiment and its metrics in an open implementation, not a new kind of simulation; and the terms of that standard must be used consistently. The capabilities of the software tools in the table are still to be established from their documentation.

## 8. Validation circuits

The circuits are generic and of increasing size. None belongs to a particular application domain. The two filters are the benchmarks of the fault-diagnosis literature (RELATED_WORK.md §2.4: named in 85 and 45 of 137 abstracts); a regulator is part of a published analog test benchmark (§2.1). They are implemented in `validation/`. Their exact schematics and values are still to be taken from a cited source (open decision 1).

| Circuit | Role | Why |
|---|---|---|
| Resistive divider and RC low-pass | Analytical reference | Closed-form response and closed-form distributions under tolerance: the estimators are tested against known values. Already in the unit tests and in `examples/filter` |
| Sallen–Key band-pass filter, one op-amp | Small benchmark | A circuit used in the analog fault-diagnosis literature, so results can be set beside published ones. Natural specifications: centre frequency, gain, quality factor |
| Four-op-amp biquad high-pass filter | Larger benchmark | The second benchmark of that literature. More components than independent measurements: the case where ambiguity groups appear. The full ambiguity analysis of the manuscript goes here |
| Linear voltage regulator with device-level models | Realistic, not a filter | Transistors and a reference with real temperature behaviour, so temperature is a meaningful operating condition, together with line and load. Specifications with engineering meaning: output voltage, line and load regulation, dropout |

Each circuit needs four things, and only these are specific to it: its netlist, its fault rules, its measurements, and its specification limits. How little that is, is the evidence for RQ7.

## 9. Relationship with the application study

`spicefault` was extracted from the simulation code of a study on self-diagnosis of ECG analog front-ends, and that study now uses the library for its simulations. It is independent research with its own paper, which is to be published before this manuscript.

Consequences:

- **This manuscript cites it as an application** of the framework, in one or two sentences. It reports none of its results and does not use its circuits as case studies.
- **Nothing in this manuscript depends on it.** Every result here comes from the validation circuits of §8.
- **Equivalence with the original code is a matter of software quality, not a result.** The library is tested against the code it was extracted from (`tests/regression/`): identical netlists, random numbers, measurements and datasets. That is how correctness was established during development; the manuscript may mention it in a sentence. Those tests are tied to the version of the application before it adopted the library.

## 10. Limitations to declare

Dependence on an external simulator and its version; convergence failures under hard faults; computational cost; fidelity of device models; incompleteness of any fault taxonomy; single-fault assumption; no physical validation; detectability figures depend on the chosen measurements and decision rule; validation on a small number of circuits.

## 11. Open decisions

1. **Validation circuits** (§8): align the schematics and values of the two filters with a cited source, and replace the device models of the regulator by vendor models with their source.
2. **Sharper criterion for RQ6** (§4).
3. **Literature**: the follow-up of RELATED_WORK.md §4 (full texts, software documentation, citation chaining, a second database).
4. **Licence**: MIT or BSD-3-Clause.
