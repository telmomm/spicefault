# SPICEFAULT

> **Revision note.** This is the original specification of the project and is kept as
> written. One decision in it has changed since: the ECG front-end is no longer the
> case study of the manuscript. The framework is validated on generic circuits, and
> the ECG study, an independent paper published first, is only cited as an application
> that uses the library.
>
> Superseded by that decision: sections 38 (ECG case study) and 39 (additional
> circuit), the ECG parts of sections 35, 45 (items 10 and 11) and 46 (experiment H),
> and phases 9 and 10 of the roadmap (sections 56 and 57). Section 48 describes how the
> library was started, which is done. The current plan is in
> [docs/SCIENTIFIC_SCOPE.md](docs/SCIENTIFIC_SCOPE.md) and
> [docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md).

**SPICE-based Fault Injection and Reliability Assessment Framework**

> A reproducible Python framework for fault injection, uncertainty analysis, large-scale simulation, and reliability-oriented assessment of electronic circuits using SPICE-compatible simulators.

---

# 1. Project vision

`spicefault` is an open-source Python framework for the **systematic and reproducible injection, simulation, characterization, and analysis of faults in electronic circuits**.

The primary scientific objective is not to provide another Python interface to SPICE.

Instead, `spicefault` addresses a higher-level reliability engineering problem:

> **How can electronic-circuit fault scenarios be systematically defined, simulated under realistic component and operating variability, quantified, reproduced, and subsequently analysed?**

The framework should provide a complete computational workflow:

```text
Circuit
   │
   ├── Component uncertainty
   │
   ├── Operating-condition variability
   │
   └── Fault model
          │
          ▼
    Fault injection
          │
          ▼
    SPICE simulation
          │
          ▼
   Electrical response
          │
          ├── Measurements
          ├── Waveforms
          └── Failure indicators
                  │
                  ▼
        Reliability analysis
                  │
                  ├── Fault detectability
                  ├── Fault separability
                  ├── Sensitivity
                  ├── Robustness
                  └── Diagnostic performance
```

Machine-learning dataset generation is one possible output of this framework, but **it is not the fundamental purpose of the library**.

---

# 2. Target publication

The principal scientific target for the project is:

**IEEE Transactions on Reliability (T-REL).**

The journal covers reliability engineering and allied disciplines, including hardware reliability, fault modelling, fault diagnosis, testing, prognostics and dependability of engineered systems. Its scope extends from individual components to complete systems and includes applications in medical devices and electronic systems.

Therefore, the project should be developed around a **reliability-engineering contribution**, rather than around a generic software contribution.

The software is the research instrument.

The scientific contribution is the methodology enabled by the software.

---

# 3. Central scientific hypothesis

The project should investigate the following proposition:

> **A structured and reproducible fault-injection framework can provide a more systematic basis for evaluating the reliability and diagnostic behaviour of electronic circuits under component variability and realistic fault conditions than ad-hoc SPICE simulation workflows.**

This leads to measurable research questions.

### RQ1 — Fault coverage

Can a formal fault abstraction provide systematic coverage of relevant circuit failure modes?

### RQ2 — Variability

How does component tolerance and operating variability affect fault detectability?

### RQ3 — Reproducibility

Can large-scale fault-injection experiments be reproduced exactly or within explicitly defined numerical tolerances?

### RQ4 — Scalability

How efficiently can large fault-injection campaigns be executed?

### RQ5 — Diagnostic separability

How distinguishable are different physical faults from observable circuit responses?

### RQ6 — Reliability assessment

Can simulation-derived fault response distributions provide useful quantitative evidence about circuit robustness and diagnostic coverage?

### RQ7 — Generalisation

Can the same methodology be applied to different analogue/electronic circuits without rewriting the complete simulation infrastructure?

---

# 4. Scientific positioning

The project should not be framed as:

> "A Python library for generating ML datasets."

It should instead be framed as:

> "A computational framework for reproducible fault injection and reliability-oriented analysis of electronic circuits."

The ML dataset becomes:

```text
reliability experiment
        │
        ├── raw waveforms
        ├── electrical measurements
        ├── fault labels
        └── provenance
                 │
                 ▼
          diagnostic dataset
                 │
                 ▼
          ML / statistical analysis
```

This distinction is critical for the target journal.

---

# 5. Scope

## 5.1 Included

The initial scope includes:

* SPICE-based circuit simulation;
* explicit fault modelling;
* component-level fault injection;
* parametric faults;
* open-circuit faults;
* short-circuit faults;
* component-value drift;
* component tolerance;
* Monte Carlo variation;
* operating-condition variation;
* large-scale simulation campaigns;
* parallel execution;
* simulation failure handling;
* measurement extraction;
* waveform extraction;
* fault-response characterisation;
* fault detectability analysis;
* reproducibility;
* provenance;
* dataset generation.

---

## 5.2 Explicitly excluded from the initial release

The project will not initially attempt to implement:

* a SPICE numerical solver;
* schematic capture;
* a graphical user interface;
* a complete reliability prediction standard;
* a full FMEA engine;
* a complete FTA engine;
* a proprietary circuit simulator;
* a machine-learning framework;
* distributed computing across clusters;
* GPU-based circuit solving.

These may become future extensions if scientifically justified.

---

# 6. Conceptual architecture

The framework should be structured around:

```text
Circuit
   │
   ├── Variation
   │
   ├── Fault
   │
   └── OperatingCondition
          │
          ▼
       Experiment
          │
          ▼
      Simulator
          │
          ▼
    SimulationResult
          │
          ▼
     Measurements
          │
          ▼
     FaultResponse
          │
          ▼
   ReliabilityAnalysis
```

The key abstraction is therefore not `Dataset`.

The key abstraction is:

```text
Experiment
```

A dataset is one possible representation of the experiment's results.

---

# 7. Core domain objects

The initial API should contain the following conceptual objects:

```text
Circuit
Fault
FaultSet
Variation
OperatingCondition
SimulationConfig
Simulator
Experiment
SimulationResult
Measurement
FaultResponse
Dataset
ReliabilityAnalysis
```

---

# 8. Circuit

A circuit represents the system under investigation.

```python
from spicefault import Circuit

circuit = Circuit.from_netlist(
    "ecg_frontend.cir"
)
```

The circuit object should provide sufficient information to:

* identify components;
* identify nodes;
* inspect parameters;
* generate a simulation netlist;
* apply controlled modifications.

Example:

```python
circuit.components()
circuit.nodes()
circuit.parameters()
circuit.to_netlist()
```

The API should remain deliberately narrower than a full schematic-design environment.

---

# 9. Fault model

Faults are the central scientific abstraction.

```python
from spicefault.faults import OpenCircuit

fault = OpenCircuit(
    component="R17"
)
```

Possible initial fault types:

```text
OpenCircuit
ShortCircuit
ParametricFault
DriftFault
LeakageFault
```

Potential future models:

```text
IntermittentFault
StuckAtFault
BridgingFault
DegradationFault
MultipleFault
```

Only fault types that can be defined with a clear physical interpretation should be included.

---

# 10. Fault metadata

Every fault must be explicitly described.

Example:

```python
fault.metadata()
```

should return information conceptually equivalent to:

```json
{
  "fault_type": "open_circuit",
  "component": "R17",
  "nominal_state": "connected",
  "fault_state": "open"
}
```

The metadata should become part of the experimental provenance.

---

# 11. Parametric faults

Parametric faults represent degradation or abnormal component values.

```python
from spicefault.faults import ParametricFault

fault = ParametricFault(
    component="R17",
    parameter="resistance",
    nominal_value=10000,
    fault_value=15000
)
```

The framework should distinguish:

```text
nominal variability
        ≠
fault condition
```

This distinction is scientifically essential.

---

# 12. Fault severity

Faults should optionally expose a normalised severity parameter.

Example:

```python
fault = ParametricFault(
    component="R17",
    parameter="resistance",
    nominal_value=10000,
    fault_value=15000,
    severity=0.5
)
```

The severity definition must be physically and mathematically documented.

Possible interpretation:

$$
s = \frac{x_f-x_0}{x_{\mathrm{reference}}}
$$

or another domain-appropriate definition.

The framework must never assume that all fault severities can be represented by a universal formula.

---

# 13. Component and manufacturing variability

Reliability assessment requires separating normal variability from faults.

Example:

```python
from spicefault.variation import NormalVariation

variation = NormalVariation(
    parameter="R17",
    mean=10000,
    sigma=500
)
```

Possible distributions:

```text
Fixed
Uniform
Normal
LogNormal
Custom
```

A variation set:

```python
from spicefault.variation import VariationSet

variation = VariationSet([
    NormalVariation("R1", 10000, 500),
    NormalVariation("R2", 1000, 50),
])
```

---

# 14. Operating conditions

The framework should distinguish component uncertainty from operating conditions.

Example:

```python
from spicefault.conditions import OperatingCondition

condition = OperatingCondition(
    supply_voltage=5.0,
    temperature=25.0
)
```

Potential future parameters:

* supply voltage;
* temperature;
* input amplitude;
* input frequency;
* load;
* sensor characteristics.

This is important because fault detectability may depend strongly on operating conditions.

---

# 15. Simulator abstraction

The simulator is an external numerical engine.

```python
from spicefault.simulation import Simulator

simulator = Simulator(
    backend="ngspice"
)
```

The public framework should not depend on the internals of ngspice.

Initial backend:

```text
NgspiceBackend
```

Potential future backends:

```text
XyceBackend
OtherSpiceBackend
```

The backend interface should be:

```python
class SimulatorBackend:

    def run(
        self,
        netlist: str,
        config: SimulationConfig
    ) -> SimulationResult:
        ...
```

---

# 16. Why not PySpice?

`spicefault` should explicitly distinguish itself from Python/SPICE interfaces such as PySpice.

The project should **not** claim that PySpice is incapable of performing fault-injection experiments.

Instead, the distinction is architectural.

A generic Python/SPICE interface primarily exposes functionality such as:

```text
Python
   │
   ├── circuit construction
   ├── simulator configuration
   └── simulation
```

`spicefault` operates one abstraction level above:

```text
Python
   │
   ├── circuit
   ├── fault model
   ├── uncertainty model
   ├── operating conditions
   ├── experiment definition
   ├── simulation orchestration
   ├── measurement extraction
   ├── provenance
   └── reliability analysis
             │
             ▼
       SPICE backend
```

Therefore:

> **PySpice-like tools can be simulation interfaces; `spicefault` is intended to be a reliability-experiment framework.**

The comparison should be demonstrated experimentally and functionally in the eventual paper.

---

# 17. Why direct ngspice execution?

The initial implementation should retain the current architecture:

```text
Python
   │
   ▼
SPICE netlist
   │
   ▼
ngspice process
   │
   ▼
simulation output
   │
   ▼
Python
```

Advantages include:

* explicit simulator boundary;
* independent simulator versioning;
* easy command-line reproducibility;
* straightforward fault-induced netlist modification;
* process isolation;
* compatibility with the existing research code;
* potential future simulator backends.

The project should document the limitations of this approach rather than claiming universal superiority.

---

# 18. Experiment

The central high-level object should be an experiment.

```python
experiment = Experiment(
    circuit=circuit,
    simulator=simulator,
    faults=faults,
    variations=variation,
    conditions=conditions,
    seed=42,
)
```

An experiment should define:

```text
what is simulated
why it is simulated
under which conditions
with which faults
with which uncertainty
using which simulator
```

---

# 19. Experiment execution

```python
result = experiment.run(
    workers=16
)
```

The experiment engine should provide:

* deterministic sample generation;
* parallel execution;
* failure tracking;
* progress information;
* chunking;
* resumability;
* metadata generation.

---

# 20. Monte Carlo experiments

Example:

```python
experiment = Experiment(
    circuit=circuit,
    variations=variation,
    faults=faults,
    samples=10000,
    seed=42,
)
```

Each simulation receives a deterministic derived seed:

```text
master seed
      │
      ├── sample 0
      ├── sample 1
      ├── sample 2
      └── ...
```

The generated results should remain reproducible independently of the number of workers.

This property should be explicitly tested.

---

# 21. Fault campaign

A fault campaign is a systematic set of experiments.

Example:

```python
campaign = FaultCampaign(
    circuit=circuit,
    faults=[
        OpenCircuit("R1"),
        ShortCircuit("C1"),
        ParametricFault("R2", ...),
    ],
    variations=variation,
    samples_per_fault=5000,
)
```

The campaign should provide:

```python
campaign.run()
campaign.summary()
campaign.validate()
```

---

# 22. Reliability-oriented outputs

The framework should not stop at generating waveforms.

It should support quantities such as:

### Fault detectability

Probability that a fault produces an observable change:

$$
P(\text{detect} \mid \text{fault}, \text{conditions})
$$

### Fault separability

Ability to distinguish two fault classes from measurements.

### Sensitivity

Response of measurements to fault severity:

$$
S = \frac{\partial y}{\partial s}
$$

### Robustness

Effect of normal component variability on fault signatures.

### Operating-condition dependence

Variation in fault detectability across:

$$
T,\ V_{CC},\ f,\ \text{load},\ldots
$$

These concepts move the project from "dataset generator" toward reliability engineering.

---

# 23. Measurement API

Measurements should be independent from the simulator.

```python
from spicefault.measurements import Measurement

measurements = [
    Measurement.rms("V(out)"),
    Measurement.peak("V(out)"),
    Measurement.peak_to_peak("V(out)"),
]
```

Potential measurements:

```text
RMS
mean
variance
peak
peak-to-peak
frequency
gain
phase
rise time
fall time
settling time
bandwidth
THD
noise
```

Only measurements with a well-defined mathematical or engineering interpretation should be included.

---

# 24. Fault response

A `FaultResponse` should combine:

```text
fault
+
simulation
+
operating condition
+
measurements
+
waveforms
```

Example:

```python
response = FaultResponse(
    fault=fault,
    result=result,
    measurements=measurements,
)
```

This object becomes the fundamental unit for subsequent reliability analysis.

---

# 25. Reliability analysis API

Example:

```python
from spicefault.reliability import ReliabilityAnalysis

analysis = ReliabilityAnalysis(
    responses
)
```

Potential methods:

```python
analysis.detectability()
analysis.sensitivity()
analysis.robustness()
analysis.separability()
analysis.failure_rate()
```

The first release should only implement metrics that can be defined rigorously.

---

# 26. Fault detectability

A central analysis should quantify whether a fault is distinguishable from nominal behaviour.

Conceptually:

```text
Nominal distribution
        │
        │ overlap
        ▼
Fault distribution
```

Possible metrics:

* effect size;
* distribution overlap;
* ROC-AUC;
* detection probability;
* false-positive probability;
* false-negative probability.

These metrics should be configurable and explicitly defined.

---

# 27. Fault diagnosis

Machine learning may be used as a downstream analysis:

```text
spicefault
    │
    ▼
fault-response dataset
    │
    ├── statistical diagnosis
    ├── threshold diagnosis
    └── machine learning
```

The library should not require a particular ML framework.

Example:

```python
X, y = dataset.to_ml()
```

The ML analysis itself belongs to the application/research layer.

---

# 28. Dataset representation

Datasets remain an important output.

Recommended conceptual structure:

```text
dataset/
├── manifest.json
├── metadata.json
├── samples.parquet
├── waveforms.npy
└── parts/
```

The dataset must preserve:

* sample identifier;
* fault;
* fault severity;
* circuit;
* component values;
* operating conditions;
* measurements;
* waveform reference;
* random seed;
* simulator version;
* spicefault version.

---

# 29. Provenance

Every simulation should be traceable.

Example:

```json
{
  "sample_id": 18231,
  "seed": 9138271,
  "circuit": "ecg_frontend_v1",
  "fault_type": "resistor_open",
  "fault_location": "R17",
  "fault_severity": 1.0,
  "simulator": "ngspice",
  "simulator_version": "43",
  "spicefault_version": "0.2.1"
}
```

The provenance model is a central scientific feature.

---

# 30. Reproducibility

The project should target:

> **Configuration-level reproducibility and sample-level traceability.**

A complete experiment should be represented by:

```text
experiment configuration
+
source circuit
+
fault definitions
+
variation definitions
+
operating conditions
+
random seed
+
software versions
+
simulator version
```

IEEE explicitly encourages research reproducibility through detailed methodology, shared data and shared code.

Therefore reproducibility should not be an afterthought.

It should be part of the software architecture.

---

# 31. Resumable execution

Large fault campaigns may require tens or hundreds of thousands of simulations.

The framework must support:

```python
experiment.run(
    workers=16,
    resume=True
)
```

The manifest should record:

```text
completed
failed
pending
```

Individual failures should not necessarily terminate the complete experiment.

---

# 32. Simulation status

Each simulation should have an explicit status:

```text
SUCCESS
FAILED
TIMEOUT
CONVERGENCE_ERROR
INVALID_OUTPUT
```

Example:

```json
{
  "sample_id": 12831,
  "status": "CONVERGENCE_ERROR",
  "message": "ngspice failed to converge"
}
```

This is particularly important for reliability-oriented simulation because numerical convergence failures must not silently disappear from the experimental record.

---

# 33. Parallel execution

The initial implementation should exploit the natural independence of Monte Carlo/fault simulations.

```text
Experiment
    │
    ├── Worker 1 → SPICE
    ├── Worker 2 → SPICE
    ├── Worker 3 → SPICE
    └── Worker N → SPICE
```

Initial implementation:

```python
workers=16
```

using local multiprocessing.

The project should benchmark:

```text
1
2
4
8
16
...
```

workers.

---

# 34. GPU strategy

GPU acceleration should **not** be a requirement for version 1.0.

The current workflow launches independent ngspice simulations:

```text
Python → ngspice process
```

A conventional GPU does not accelerate that external SPICE solver automatically.

The framework should therefore first optimise:

* process parallelism;
* simulation configuration;
* I/O;
* netlist generation;
* output extraction;
* batching;
* failure recovery.

A future GPU backend could investigate batch/vectorised circuit solving.

This should be treated as a separate research direction.

---

# 35. Reliability benchmark

A major part of the T-REL paper should be a quantitative benchmark.

The benchmark should compare:

### Baseline

Current project-specific implementation.

### Framework

`spicefault`.

Measurements:

```text
simulation time
simulations/second
CPU utilisation
memory
I/O
failure rate
recovery time
reproducibility
```

The purpose is not merely to demonstrate that the package runs.

The purpose is to demonstrate that the abstraction introduces **scientifically useful and computationally measurable advantages**.

---

# 36. Fault coverage benchmark

For each circuit:

```text
components
   ↓
fault types
   ↓
fault severities
   ↓
operating conditions
```

The framework should produce a fault coverage matrix.

Example:

| Component | Open | Short | Drift | Leakage |
| --------- | ---: | ----: | ----: | ------: |
| R1        |    ✓ |     ✓ |     ✓ |       — |
| R2        |    ✓ |     ✓ |     ✓ |       — |
| C1        |    ✓ |     ✓ |     ✓ |       ✓ |
| U1        |    — |     — |     — |       — |

This makes the fault campaign auditable.

---

# 37. Reliability analysis benchmark

The paper should demonstrate that normal component variability can alter fault detectability.

For example:

```text
Without tolerance:
fault A → clearly separated response

With ±5% tolerance:
fault A → partially overlapping response

With ±10% tolerance:
fault A → substantially reduced detectability
```

This is scientifically much more valuable for T-REL than simply reporting classification accuracy.

---

# 38. ECG frontend case study

The existing ECG frontend project should be the principal real-world validation case.

The case study should demonstrate:

```text
ECG analogue frontend
        │
        ├── nominal variability
        ├── component faults
        ├── operating conditions
        │
        ▼
     SPICE
        │
        ▼
   measurements
        │
        ▼
 fault-response distributions
        │
        ├── detectability
        ├── robustness
        ├── separability
        └── ML diagnosis
```

The ECG application should therefore be a **case study of reliability assessment**, rather than the sole definition of the software.

---

# 39. Additional validation circuit

For generalisation, the project should include at least one circuit independent from the ECG application.

Possible example:

```text
Instrumentation amplifier
```

or:

```text
Active low-pass filter
```

This demonstrates that the framework is not merely an ECG-specific implementation.

---

# 40. State-of-the-art comparison

The eventual paper should compare `spicefault` against at least:

* generic Python/SPICE interfaces;
* PySpice;
* SPICE command-line workflows;
* existing fault-injection approaches;
* relevant circuit reliability simulation approaches.

The comparison should consider:

```text
fault abstraction
Monte Carlo
fault campaigns
parallel execution
resumability
provenance
reproducibility
measurement extraction
reliability metrics
dataset generation
backend abstraction
```

The paper must avoid unsupported claims of superiority.

---

# 41. Key novelty claim

The central novelty should be:

> **The integration of explicit fault modelling, uncertainty modelling, large-scale simulation orchestration, measurement extraction, provenance and reliability-oriented analysis into a reusable SPICE-independent experimental framework.**

Not:

> "A Python wrapper around ngspice."

And not:

> "A faster SPICE simulator."

---

# 42. What makes the project scientifically relevant to T-REL?

The project should demonstrate that the framework enables analyses that are difficult to reproduce systematically with ad-hoc simulation scripts.

In particular:

```text
fault
+
uncertainty
+
operating conditions
+
observable response
```

can be treated as a structured reliability experiment.

This allows researchers to quantify:

* fault coverage;
* detectability;
* robustness;
* sensitivity;
* diagnostic ambiguity;
* operating-condition dependence.

These are reliability questions rather than purely software-engineering questions.

---

# 43. JOSS is no longer the primary design constraint

The project should not be optimised around a JOSS paper.

JOSS-style software quality remains desirable:

```text
tests
documentation
packaging
examples
CI
licensing
citation
reproducibility
```

but the scientific manuscript should target the contribution expected from a reliability journal.

The target question becomes:

> **What reliability-engineering knowledge or methodology becomes possible because of this framework?**

---

# 44. Target paper

Provisional title:

> **SPICEFAULT: A Reproducible Framework for Fault Injection and Reliability Assessment of Electronic Circuits**

Alternative:

> **A Reproducible SPICE-Based Framework for Fault Injection, Uncertainty Analysis, and Reliability Assessment of Electronic Circuits**

The second title is preferable if the uncertainty/reliability analysis becomes a major contribution.

---

# 45. Proposed T-REL paper structure

## 1. Introduction

Problem:

* reliability assessment of electronic circuits;
* systematic fault injection;
* component variability;
* simulation-based reliability analysis.

Contribution:

* framework;
* methodology;
* reproducibility;
* scalability;
* validation.

---

## 2. Related Work

Cover:

* circuit fault simulation;
* fault injection;
* SPICE-based reliability analysis;
* Monte Carlo circuit analysis;
* Python/SPICE tools;
* PySpice;
* ML-based circuit diagnosis.

---

## 3. Problem Formulation

Define:

$$
C
$$

as the nominal circuit,

$$
\theta
$$

as uncertain parameters,

$$
f
$$

as a fault,

$$
u
$$

as operating conditions,

and:

$$
y = S(C,\theta,f,u)
$$

as the simulated observable response.

The framework seeks to evaluate:

$$
P(y \mid f,\theta,u)
$$

and compare it with:

$$
P(y \mid f=0,\theta,u)
$$

for reliability and fault-diagnosis purposes.

---

## 4. Framework Architecture

Describe:

```text
Circuit
Fault
Variation
Condition
Experiment
Simulator
Measurement
Reliability Analysis
```

---

## 5. Fault Injection Methodology

Define:

* fault taxonomy;
* fault severity;
* fault locations;
* parameter perturbations;
* open/short faults.

---

## 6. Uncertainty and Monte Carlo

Describe:

* tolerance distributions;
* random sampling;
* operating-condition variability;
* deterministic seeds.

---

## 7. Simulation and Computational Architecture

Describe:

* ngspice;
* process isolation;
* parallelisation;
* chunking;
* resumability;
* error handling.

---

## 8. Reliability Metrics

Define:

* fault detectability;
* sensitivity;
* robustness;
* fault separability;
* diagnostic coverage.

---

## 9. Experimental Evaluation

Experiments should include:

1. correctness;
2. reproducibility;
3. computational scalability;
4. fault coverage;
5. tolerance sensitivity;
6. operating-condition sensitivity.

---

## 10. ECG Frontend Case Study

Use the existing ECG project.

---

## 11. Generalisation Case Study

Use an independent circuit.

---

## 12. Discussion

Discuss:

* strengths;
* limitations;
* simulator dependence;
* computational cost;
* numerical convergence;
* future GPU/distributed extensions.

---

## 13. Conclusion

Summarise the reliability contribution.

---

# 46. Experimental plan

The development should produce the experiments required by the eventual paper.

## Experiment A — Correctness

Compare:

```text
direct ngspice
vs.
spicefault
```

for identical netlists.

Expected result:

```text
equivalent waveforms
equivalent measurements
```

within defined numerical tolerances.

---

## Experiment B — Reproducibility

Run:

```text
seed = 42
workers = 1
```

and:

```text
seed = 42
workers = 16
```

Expected result:

```text
identical sample definitions
equivalent simulation outputs
```

within defined numerical tolerances.

This is an important scientific contribution.

---

## Experiment C — Parallel scalability

Measure:

```text
workers = 1, 2, 4, 8, 16
```

Report:

$$
S(N)=\frac{T_1}{T_N}
$$

and:

$$
E(N)=\frac{S(N)}{N}
$$

---

## Experiment D — Fault coverage

Measure the number of:

```text
components
×
fault types
×
severities
×
operating conditions
```

covered by a campaign.

---

## Experiment E — Variability impact

Compare:

```text
0% tolerance
±1%
±5%
±10%
```

and measure the resulting degradation in fault detectability.

---

## Experiment F — Operating-condition impact

Evaluate:

```text
temperature
supply voltage
input amplitude
frequency
load
```

where physically meaningful.

---

## Experiment G — Fault separability

Quantify how easily faults can be distinguished based on:

```text
measurements
```

and:

```text
waveforms
```

---

## Experiment H — ML as downstream application

Use the generated data with:

* classical classifiers;
* CNN;
* other ML methods.

The goal is not to claim that ML itself is the main contribution.

Instead:

> ML diagnosis demonstrates that the reliability-oriented fault-response data are useful for automated diagnosis.

---

# 47. Development roadmap

## Phase 0 — Scientific specification

Before coding:

* define reliability questions;
* define fault taxonomy;
* define uncertainty model;
* define reliability metrics;
* define experimental protocol;
* define comparison with existing approaches.

Deliverables:

```text
SCIENTIFIC_SCOPE.md
FAULT_MODEL.md
RELIABILITY_METRICS.md
EXPERIMENT_PLAN.md
```

---

# 48. Phase 1 — Extract existing implementation

Extract from the ECG project:

```text
netlist generation
fault injection
Monte Carlo
ngspice execution
raw parsing
feature extraction
dataset generation
```

without changing scientific results.

Target:

```text
existing ECG results
        ↓
spicefault
        ↓
equivalent results
```

---

# 49. Phase 2 — Core abstractions

Implement:

```text
Circuit
Fault
Variation
OperatingCondition
Simulator
Experiment
SimulationResult
```

No advanced reliability metrics yet.

---

# 50. Phase 3 — Fault framework

Implement and test:

```text
OpenCircuit
ShortCircuit
ParametricFault
FaultSeverity
FaultSet
```

Each must have:

```text
physical interpretation
metadata
serialisation
tests
```

---

# 51. Phase 4 — Uncertainty framework

Implement:

```text
NormalVariation
UniformVariation
LogNormalVariation
VariationSet
```

and deterministic random seeds.

---

# 52. Phase 5 — Experiment engine

Implement:

```text
Experiment
FaultCampaign
```

with:

* sample generation;
* parallel execution;
* chunking;
* resumability;
* failure tracking.

---

# 53. Phase 6 — Reliability analysis

Implement only scientifically justified metrics:

```text
detectability
sensitivity
robustness
separability
```

Each metric must have:

* mathematical definition;
* documentation;
* unit tests;
* reference implementation;
* example.

---

# 54. Phase 7 — Dataset layer

Implement:

```text
Dataset
Manifest
Provenance
```

The dataset is an output of the reliability experiment.

---

# 55. Phase 8 — Benchmarking

Build automated benchmarks for:

```text
runtime
parallel speedup
memory
I/O
fault coverage
reproducibility
```

Store benchmark results in machine-readable form.

---

# 56. Phase 9 — ECG validation

Reproduce the existing ECG study using `spicefault`.

The acceptance criterion should be:

> No scientifically relevant change in the generated results attributable to the software refactoring.

Any numerical differences must be documented.

---

# 57. Phase 10 — Independent validation circuit

Add a non-ECG circuit.

Recommended first candidate:

> Instrumentation amplifier.

Reason:

* common in biomedical instrumentation;
* easy to explain;
* naturally exposes gain/common-mode behaviour;
* provides a clear bridge between the ECG case study and general electronic reliability.

---

# 58. Phase 11 — State-of-the-art comparison

Build a reproducible comparison with:

```text
direct ngspice scripting
PySpice-based workflow
spicefault
```

Compare:

```text
implementation effort
fault abstraction
reproducibility
parallel execution
metadata
resumability
fault coverage
dataset generation
```

The comparison should be based on an identical benchmark.

---

# 59. Phase 12 — Final software release

Target:

```text
spicefault 1.0.0
```

Requirements:

```text
[x] stable public API
[x] tests
[x] CI
[x] documentation
[x] examples
[x] licence
[x] package
[x] reproducibility
[x] benchmark suite
[x] DOI
[x] CITATION.cff
```

---

# 60. Phase 13 — Manuscript

Prepare the T-REL manuscript only after the benchmark and case studies are complete.

The paper should be written around the scientific questions, not around the implementation chronology.

---

# 61. Phase 14 — Submission

Before submission:

```text
[ ] Scope checked against current T-REL instructions
[ ] Related work updated
[ ] Fault taxonomy justified
[ ] Reliability metrics formally defined
[ ] Benchmark complete
[ ] Reproducibility demonstrated
[ ] Code public
[ ] Dataset public or appropriately archived
[ ] DOI available
[ ] Supplementary material prepared
[ ] Limitations explicitly discussed
```

IEEE's current author guidance emphasises reproducibility through detailed methodology, online data and code sharing, which should therefore be integrated into the project before submission.

---

# 62. Repository architecture

Recommended repository:

```text
spicefault/
│
├── .github/
│   └── workflows/
│
├── docs/
│
├── examples/
│   ├── resistor/
│   ├── filter/
│   ├── instrumentation_amplifier/
│   └── ecg_frontend/
│
├── benchmarks/
│   ├── scalability/
│   ├── reproducibility/
│   └── fault_coverage/
│
├── src/
│   └── spicefault/
│       ├── circuit.py
│       ├── faults/
│       ├── variation/
│       ├── conditions/
│       ├── simulation/
│       ├── experiments/
│       ├── measurements/
│       ├── reliability/
│       ├── dataset/
│       └── cli.py
│
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── regression/
│   └── reproducibility/
│
├── paper/
│   ├── manuscript/
│   └── figures/
│
├── CITATION.cff
├── LICENSE
├── README.md
├── pyproject.toml
└── CHANGELOG.md
```

---

# 63. Documentation architecture

The README should immediately communicate:

```text
What is spicefault?
Why is it needed?
How is it different from a SPICE Python interface?
What reliability problem does it solve?
How do I perform my first fault campaign?
```

Documentation sections:

```text
Getting Started
Concepts
Fault Models
Uncertainty
Experiments
Simulation Backends
Reliability Analysis
Datasets
Reproducibility
Benchmarks
Examples
API Reference
```

---

# 64. Package dependencies

Core dependencies should remain minimal.

Potential:

```text
numpy
pandas
pyarrow
```

The simulator remains external:

```text
ngspice
```

ML libraries should not be mandatory.

Do not make:

```text
PyTorch
TensorFlow
scikit-learn
```

core dependencies.

This maintains the reliability-analysis focus.

---

# 65. Licensing

Use an OSI-approved permissive licence.

Candidate:

```text
BSD-3-Clause
```

or:

```text
MIT
```

The final decision should be made before the first public release.

Third-party SPICE models and circuits must retain their original licensing information.

---

# 66. Citation and archival

The project should use:

```text
GitHub
+
PyPI
+
Zenodo
+
CITATION.cff
```

Release:

```text
v1.0.0
```

should receive a DOI.

The DOI should identify the exact software version used in the paper.

---

# 67. Open-science strategy

The paper should provide:

```text
source code
configuration files
benchmark scripts
experiment definitions
dataset-generation scripts
trained models, if used
representative datasets
```

Large generated datasets may be archived separately through Zenodo or another appropriate repository.

The repository should contain enough information for an independent researcher to regenerate the data.

---

# 68. Limitations to acknowledge

The paper must explicitly discuss:

### SPICE dependence

The initial implementation depends on an external simulator.

### Numerical convergence

Some fault conditions may produce difficult or non-convergent simulations.

### Computational cost

Large campaigns remain computationally expensive.

### Model fidelity

Simulation results depend on the accuracy of the circuit model.

### Fault-model completeness

The library cannot guarantee that a selected fault taxonomy captures every physical failure mode.

### GPU

The initial implementation does not provide GPU-accelerated SPICE solving.

### Real-world validation

Simulation-based reliability evidence does not replace physical testing.

These limitations increase scientific credibility.

---

# 69. Future research

Potential extensions:

## 69.1 Xyce backend

Enable alternative high-performance SPICE-compatible simulation.

## 69.2 Distributed campaigns

Run experiments across multiple machines.

## 69.3 GPU circuit solver

Investigate vectorised circuit simulation.

## 69.4 Physical fault validation

Compare simulated fault signatures with laboratory measurements.

## 69.5 Reliability standards

Investigate integration with formal reliability assessment methodologies.

## 69.6 Automated fault generation

Derive fault campaigns automatically from circuit topology.

---

# 70. Final scientific contribution

The project should ultimately establish:

> A reproducible computational methodology for studying the reliability and fault-diagnosis behaviour of electronic circuits under component uncertainty, operating-condition variability and explicitly defined fault mechanisms.

The software is the implementation of that methodology.

The ECG frontend is the first real-world validation.

Machine learning is a downstream application.

SPICE is the simulation engine.

Reliability engineering is the scientific domain.

---

# 71. One-sentence project definition

The final README should be able to summarise the project as:

> **`spicefault` is a Python framework for reproducible SPICE-based fault injection and reliability assessment of electronic circuits under component uncertainty and operating-condition variability.**

---

# 72. One-sentence differentiation from PySpice

The project should describe the distinction as:

> **Whereas Python/SPICE interfaces such as PySpice primarily expose programmatic control of circuit simulation, `spicefault` provides a reliability-oriented experimental layer for defining faults and uncertainties, executing large fault campaigns, preserving provenance, and quantitatively analysing fault responses.**

This is a **scope distinction**, not a claim that PySpice cannot perform these tasks.

---

# 73. Final development philosophy

The project should follow this hierarchy:

```text
             RELIABILITY
                  │
          ┌───────┴────────┐
          │                │
       Faults          Uncertainty
          │                │
          └───────┬────────┘
                  │
              Experiment
                  │
               SPICE
                  │
             Measurements
                  │
          ┌───────┴────────┐
          │                │
      Reliability        ML
       analysis       diagnosis
```

The most important design decision is therefore:

> **Do not build `spicefault` around the existing ECG dataset generator. Build it around the general reliability experiment, and make the ECG dataset generator its first serious application.**

That architecture gives the project a much stronger scientific identity and a substantially better alignment with *IEEE Transactions on Reliability*.
