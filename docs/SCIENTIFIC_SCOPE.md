# Scientific scope

Phase 0 deliverable 1 of 4. Status: draft for review, nothing here is implemented yet.

Companion documents: [FAULT_MODEL.md](FAULT_MODEL.md), [RELIABILITY_METRICS.md](RELIABILITY_METRICS.md), [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md).

This document fixes what `spicefault` is meant to demonstrate, in terms precise enough that each later phase can be checked against it. The target venue is *IEEE Transactions on Reliability*, so the contribution is a reliability methodology; the software is the instrument that makes it reproducible.

## 1. Problem statement

Simulation-based fault studies of electronic circuits are usually built as project-specific scripts around a SPICE simulator. The scripts work, but four things are decided implicitly and are rarely reported:

1. what counts as a fault, as opposed to normal component variation;
2. which faults were simulated out of those that could have been;
3. what happened to simulations that failed;
4. which random numbers produced which sample.

Without these, the resulting fault-response distributions cannot be audited or regenerated, and figures such as detectability or diagnostic coverage cannot be compared between studies.

`spicefault` treats the combination of circuit, normal variation, operating condition and fault as one structured experiment, and defines the reliability quantities that are computed from it.

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
| $z = M(m; \varepsilon)$, $\varepsilon \sim P_\varepsilon$ | Measurements as the instrument would observe them (noise, quantisation, clipping) |
| $c = \mathbb{1}[m_\text{spec} \in A]$ | Compliance with a set of specification limits $A$, when the application defines one |

The objects of study are the conditional distributions $P(z \mid f, u)$, obtained by marginalising over $\theta$ and $\varepsilon$, and their comparison with $P(z \mid \varnothing, u)$.

Three choices in this setting come from the ECG study and are kept because they have scientific consequences:

- **Variation first, then fault.** A fault is injected into an already realised circuit, so every other component keeps its spread. A fault is never simulated on the nominal circuit alone, except in sensitivity analysis.
- **Simulation and observation are separate.** The simulator output is stored noise-free and $M$ is applied afterwards. Measurement noise and resolution are then study parameters that do not require new simulations.
- **Injected fault and functional failure are separate.** $f \neq \varnothing$ states what was injected; $c = 0$ states that the circuit no longer meets its specification. In the ECG baseline dataset, 50,236 of 63,600 cases of the integrated circuit are compliant although only 5,000 are fault-free, so most injected faults there do not cause a specification failure.

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

Each question is tied to a quantity, an experiment and a criterion. Metric identifiers refer to RELIABILITY_METRICS.md, experiment letters to EXPERIMENT_PLAN.md.

| RQ | Question | Quantity | Experiment | Evidence that answers it |
|---|---|---|---|---|
| RQ1 Fault coverage | Does a formal fault abstraction give systematic, auditable coverage of the failure modes of a circuit? | Coverage matrix and structural coverage (M10) | D | Every applicable (component, fault type) pair is either simulated or excluded with a recorded reason, for three circuits |
| RQ2 Variability | How do tolerance and operating variation change detectability? | Detection probability and standardised shift against tolerance scale (M1, M2, M8) | E, F | Detectability curves with confidence intervals; faults whose detectability falls below a stated level are identified |
| RQ3 Reproducibility | Can a campaign be reproduced exactly, or within stated tolerances? | Identity of sample definitions; numerical difference of outputs | A, B | Identical sample definitions for any worker count; output differences within the tolerances of EXPERIMENT_PLAN.md §3 |
| RQ4 Scalability | How efficiently do large campaigns run? | Throughput, speed-up $S(N)$, efficiency $E(N)$, overhead against the baseline | C | Measured curves with repetitions; overhead of the abstraction against the project-specific baseline |
| RQ5 Separability | How distinguishable are different faults from the observable response? | Pairwise separability, ambiguity groups (M9) | G | Ambiguity structure reported for measurements and for waveforms |
| RQ6 Reliability assessment | Do fault-response distributions give useful quantitative evidence of robustness and diagnostic coverage? | Failure probability per fault, diagnostic coverage, escape and false-reject rates (M5, M6) | E, F, ECG case study | Figures with confidence intervals that change a design or test decision in the case study |
| RQ7 Generalisation | Does the method apply to another circuit without rewriting the infrastructure? | Lines of circuit-specific code; framework code changed | D, independent circuit | The independent circuit runs with no change to framework code |

RQ6 is the weakest as written: "useful" is not measurable. The criterion proposed above (the figures change a decision in the case study) should be replaced by a sharper one once the ECG results are reproduced. A candidate already visible in the baseline is the gap between percentage severity and functional severity.

## 5. Claims and non-claims

The manuscript may claim, if the experiments support it:

- a fault abstraction in which every fault compiles to a small set of netlist transformations and carries explicit metadata;
- campaigns whose sample definitions are identical for any degree of parallelism;
- an experimental record in which failed simulations are kept and counted;
- reliability metrics with stated definitions, estimators and confidence intervals;
- the same infrastructure applied to at least three circuits.

It must not claim:

- that other tools cannot perform fault injection (the difference is one of scope, see §7);
- a faster simulator: the numerical work is done by ngspice;
- failure rates in time (FIT, MTBF): simulation gives conditional response distributions, not occurrence rates;
- completeness of the fault taxonomy with respect to physical failure mechanisms;
- that simulated evidence replaces physical testing.

## 6. Scope

Included in the first release: SPICE netlists as circuit description; single faults at component level (open, short, parametric, leakage, and composite degradation); tolerance distributions; operating conditions; deterministic sampling; local parallel execution; resumable campaigns; explicit simulation status; measurements; the metrics marked "first release" in RELIABILITY_METRICS.md; dataset export with provenance.

Excluded: a numerical solver; schematic capture; a graphical interface; FMEA or FTA engines; reliability prediction standards; machine-learning models; cluster execution; GPU solving; multiple simultaneous independent faults; intermittent and time-dependent faults.

Two boundaries need to be stated because the ECG code crosses them:

- **Specification compliance is an application concern.** The framework provides the mechanism (measurements, a compliance predicate, labels). The limits and test set-ups of IEC 60601-2-25 stay in the ECG repository.
- **Machine learning is downstream.** The framework exports features, waveforms and labels; classifiers stay in the application.

## 7. Comparison with existing approaches

The comparison is functional and must be reproducible. Nothing in the table below has been verified yet; the entries say what has to be checked, at a fixed version of each tool, before a comparison table can appear in the paper.

| Class | Candidates to examine | What to establish |
|---|---|---|
| Project-specific scripting around ngspice | `ecgfd` (the baseline), other published scripts | Measured: effort, throughput, reproducibility, behaviour on failure |
| Python/SPICE interfaces | PySpice; spicelib / PyLTSpice | Which of fault abstraction, tolerance analysis, campaign orchestration, resumption and provenance they provide. spicelib is reported to include Monte Carlo, worst-case and failure-mode analyses; this must be read from its documentation and tested |
| Simulator-native statistics | ngspice control-language Monte Carlo; LTspice `.step` with `mc()`; Spectre Monte Carlo | What is expressible, and what record is left of each sample |
| Commercial analog fault simulation | Cadence Legato Reliability, Siemens Tessent DefectSim, Synopsys TestMAX CustomFault | Scope (transistor-level defect-oriented test of integrated circuits, IEEE 2427) and how it differs from board-level reliability experiments. Likely not runnable; compare from documentation and state so |
| Academic fault-injection and diagnosis workflows | The works collected in the ECG literature review (`docs/SOTA/` of the ECG repository), plus a dedicated search for SPICE fault-injection frameworks | Whether fault definitions, seeds and failed simulations are reported, and whether code is available |

Criteria, one row per tool in the final table: fault abstraction; separation of variation and fault; operating conditions; campaign definition; coverage report; parallel execution; resumption; simulation status; provenance per sample; determinism under parallelism; measurement extraction; reliability metrics; dataset export; backend independence.

Method: one benchmark task (Experiment D circuit, a fixed fault list, a fixed number of samples) is implemented with direct scripting, with one Python/SPICE interface and with `spicefault`. Reported: lines of task-specific code, whether each criterion is met, and how. Lines of code is a weak measure of effort and is reported as such.

A dedicated literature search for SPICE-based fault-injection frameworks has not been done. It is a prerequisite for the novelty claim and belongs to Phase 11 at the latest; it would be safer to do it before Phase 2.

## 8. Validation circuits

| Circuit | Role | Why |
|---|---|---|
| Resistive divider and RC low-pass | Analytical reference | Closed-form response and closed-form distributions under tolerance, so the metric estimators can be tested against known values |
| ECG front-end, `integrated` | Main case study | Realistic mixed discrete and integrated design with clinical specifications |
| ECG front-end, `reference` | Second architecture | Same signal chain, discrete instrumentation amplifier |
| Independent circuit | Generalisation (RQ7) | Not derived from the ECG work |

The specification proposes an instrumentation amplifier as the independent circuit. The `reference` ECG circuit already contains a discrete three-op-amp instrumentation amplifier, so that choice would be a weak test of generalisation. See open decision 3.

## 9. Limitations to declare

Dependence on an external simulator and its version; convergence failures under hard faults; computational cost; fidelity of behavioural device models; incompleteness of any fault taxonomy; single-fault assumption; no physical validation; detectability figures depend on the chosen measurements and decision rule.

## 10. Open decisions

1. **Sharper criterion for RQ6** (§4).
2. **Literature search on SPICE fault-injection frameworks**: before Phase 2, or at Phase 11.
3. **Independent circuit.** Recommended: a filter from the analog fault-diagnosis benchmark literature (a Sallen–Key band-pass or a state-variable filter), because published results exist for comparison, plus one circuit with transistor-level models if temperature is to be studied (see EXPERIMENT_PLAN.md, Experiment F).
4. **Licence**: MIT, as in the ECG repository, or BSD-3-Clause.
