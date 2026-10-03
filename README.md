# spicefault

`spicefault` is a Python framework for reproducible SPICE-based fault injection and
reliability assessment of electronic circuits under component uncertainty and
operating-condition variability.

It is not another Python interface to SPICE. It works one level above: a fault, the
normal variation of the components and the operating conditions are defined
explicitly, simulated as one experiment with ngspice, and recorded so that every
sample can be traced and regenerated.

- Project specification: [spicefault-develoment.md](spicefault-develoment.md)
- Scientific scope and research questions: [docs/SCIENTIFIC_SCOPE.md](docs/SCIENTIFIC_SCOPE.md)
- Fault model: [docs/FAULT_MODEL.md](docs/FAULT_MODEL.md)
- Reliability metrics: [docs/RELIABILITY_METRICS.md](docs/RELIABILITY_METRICS.md)
- Experiment plan: [docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md)
- Related work, from a literature search: [docs/RELATED_WORK.md](docs/RELATED_WORK.md)
- Comparison with existing approaches: [docs/COMPARISON.md](docs/COMPARISON.md)
- The long runs, with their commands: [docs/RUNBOOK.md](docs/RUNBOOK.md)

## Status

The framework is implemented and tested, and so are the validation circuits and the
scripts of the experiments. What remains is to run the campaigns and benchmarks, which
are long ([docs/RUNBOOK.md](docs/RUNBOOK.md)), and to write the manuscript.

| Part | State |
|---|---|
| Fault model: fault types, severity, fault sets, fault universe and coverage | Done |
| Uncertainty: distributions of the healthy population, seeding schemes | Done |
| Simulation: ngspice backend, explicit status of every simulation | Done |
| Experiments: in memory, and campaigns to disk, resumable | Done |
| Measurements and waveforms | Done |
| Reliability analysis, with confidence intervals | Done |
| Dataset: integrity, provenance, reproduction from the folder | Done |
| Benchmarks: scalability, reproducibility, fault coverage | Written; timing runs pending |
| Validation circuits: Sallen–Key band-pass, four-op-amp biquad, voltage regulator | Written and tested; schematics and device models to be taken from cited sources |
| Experiments on them: tolerance, operating conditions, separability | Scripts written and tested at small size; campaigns pending |
| Literature search and comparison with existing tools | First version; full texts and tool documentation still to be read |
| Release: licence, citation file, archive with DOI | Not started |

The plan is in [docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md), section 6.

| Module | Content |
|---|---|
| `spicefault.circuit` | `Circuit`: a netlist seen as components, nodes and parameters |
| `spicefault.faults` | Fault types, each a list of primitives with its metadata; severity scales; `FaultSet`; `FaultUniverse` with its coverage matrix |
| `spicefault.variation` | Distributions of the healthy population and `VariationSet`, which draws them in a fixed order |
| `spicefault.conditions` | `OperatingCondition`: settings and temperature, applied after the fault |
| `spicefault.simulation` | `Simulator`, `SimulationConfig`, `SimulationResult` with an explicit status; ngspice backend, batch runner and raw-file reader |
| `spicefault.experiments` | `Experiment` (in memory) and `FaultCampaign` (to disk, resumable); per-sample random streams and seeding schemes |
| `spicefault.netlist` | Netlist as text and the three fault-injection primitives |
| `spicefault.measurements` | `Measurement` and `Waveform`: what is read from each simulation; ADC quantisation |
| `spicefault.reliability` | `ReliabilityAnalysis` and the metrics of docs/RELIABILITY_METRICS.md |
| `spicefault.dataset` | `Dataset`, `Manifest`, `Provenance`: the folder a campaign writes, as an object |

## Setup

Requires Python ≥ 3.10 and [ngspice](https://ngspice.sourceforge.io/) on the `PATH`
(developed with ngspice 44).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Tests that need ngspice are skipped if it is missing.

## Example

A resistive divider with 1 % resistors, two faults and two operating conditions:

```python
from spicefault import Circuit, Experiment, OperatingCondition, SimulationConfig
from spicefault.faults import OpenCircuit, ParametricFault
from spicefault.variation import ToleranceVariation

circuit = Circuit("""divider
V1 in 0 dc 1
R1 in out 10k
R2 out 0 10k
.end
""", name="divider")

experiment = Experiment(
    circuit,
    config=SimulationConfig(analyses=("op",), outputs=("v(out)",)),
    faults=[OpenCircuit("R2"), ParametricFault("R1", deviation=0.2)],
    variations=[ToleranceVariation("R1", 0.01), ToleranceVariation("R2", 0.01)],
    conditions=[OperatingCondition(), OperatingCondition("low supply", settings={("V1", "dc"): 0.5})],
    samples=100,
    seed=42,
)
result = experiment.run(workers=4)

print(result.status_counts())      # {'SUCCESS': 600}
table = result.to_frame()          # sample definitions, status and realised values
vout = [s.result.plot("op")["v(out)"][0] for s in result]
```

How a sample is built:

1. Its random stream depends only on (seed, fault, replica), so the result does not
   depend on the number of workers.
2. The component values are drawn; then the fault is injected into that realised
   circuit; then the operating condition is set.
3. One drawn circuit is simulated under every operating condition.
4. A simulation that fails is kept in the results with a status (`TIMEOUT`,
   `CONVERGENCE_ERROR`, `INVALID_OUTPUT`, `FAILED`) and does not stop the run.

`experiment.metadata()` returns everything needed to regenerate the samples: circuit
hash, fault records, variations, conditions, seed, simulator and package versions.

One point differs from the API sketched in the project specification: an operating
condition names the netlist elements it sets (`settings={("Vcc", "dc"): 3.0}`),
because a keyword such as `supply_voltage` cannot be mapped to a netlist without
knowing the circuit.

### Faults

| Type | Physical interpretation | What it does to the netlist |
|---|---|---|
| `OpenCircuit` | Broken connection | A large resistance in series with one terminal |
| `ShortCircuit` | Bridge or breakdown across a component | A small resistance between two terminals |
| `LeakageFault` | Finite parasitic conduction | A graded resistance between two terminals |
| `ParametricFault` | A parameter outside its tolerance band | The parameter is moved relative to its realised value, or set to a value |
| `CompositeFault` | One mechanism with several electrical consequences | The changes of its parts, in order |

Every fault has an identifier, a record (`fault.metadata()`) from which it can be
rebuilt, and optionally a severity on a declared scale. A `FaultSet` is the ordered
fault list of an experiment; it can be saved to JSON and filtered.

### Fault universe and coverage

The faults of a campaign can be generated from rules instead of listed by hand, so
that what was simulated and what was left out are both on record:

```python
from spicefault.faults import FaultUniverse, open_rule, parametric_rule, short_rule

universe = FaultUniverse(
    circuit,
    [open_rule("R"), short_rule("R"), parametric_rule([-0.2, -0.05, 0.05, 0.2], "R")],
)
universe.exclude("R2:short", reason="ties the output to ground: outside the study")

universe.coverage_matrix()   # components x fault types: conditions to simulate
universe.exclusions()        # every fault left out, with its reason
universe.coverage()          # 11 of 12 conditions

faults = universe.selected()
faults.tolerance_overlap(experiment.variations)   # parametric faults inside the tolerance band
```

### Normal variation

| Variation | Value drawn |
|---|---|
| `ToleranceVariation` | Within nominal × (1 ± tolerance), or nominal ± tolerance with `relative=False`; uniform or truncated normal |
| `NormalVariation` | Normal around a mean (the nominal value by default), optionally truncated |
| `LogNormalVariation` | Median × exp(N(0, σ)): positive, multiplicative spread |
| `UniformVariation` | Uniform between two limits |
| `LogUniformVariation` | Median times a factor between 1/spread and spread |
| `FixedVariation` | A set value, no spread |
| `CustomVariation` | Any function of the random stream and the nominal value |
| `JointVariation` | Several parameters drawn together, when they depend on each other |

```python
from spicefault.variation import NormalVariation, VariationSet, tolerances

population = tolerances(circuit, {"R": 0.01, "C": 0.05})   # by component kind
population = VariationSet([NormalVariation("R1", 10000, 500), NormalVariation("R2", 10000, 500)])
tighter = population.scaled(0.2)   # same random numbers, a fifth of the spread
```

A normal variation is not a fault: variations describe the healthy population, and a
fault is injected into a circuit already drawn from it. `scaled` multiplies every
spread while using the same random numbers, so a study of detectability against
tolerance compares the same circuits at each tolerance level.

### Seeding

`Experiment(..., seeding=...)` chooses what identifies the random stream of a sample.
In every scheme the stream depends only on the seed and on that key, never on the
number of workers or on the order of execution.

| Scheme | Key | Consequence |
|---|---|---|
| `positional` (default) | fault index, replica | The simplest. Adding or reordering faults changes the samples of the others |
| `content` | hash of the fault identifier, replica | A fault has the same samples in every experiment that contains it |
| `common` | replica | Every fault is applied to the same drawn circuits: paired comparisons, but the conditions are not independent |

Each sample records its key (`seed_key`), from which it can be drawn again alone.

### Measurements

A measurement is a named number read from one simulation. Time-domain statistics are
weighted by time, because a SPICE transient has a variable time step and the plain
average of its points is not the average of the signal.

| Measurement | Definition |
|---|---|
| `value`, `final`, `at_time` | First point (the DC value of an operating point), last point, interpolated point |
| `mean`, `rms`, `variance` | (1/T)∫x dt, √((1/T)∫x² dt), (1/T)∫(x − mean)² dt, over the run or a window |
| `peak`, `minimum`, `peak_to_peak` | Largest, smallest, and their difference |
| `magnitude`, `phase` | \|H(f)\| (optionally in dB) and its phase in degrees, interpolated over log-frequency |
| `custom` | Any function of one plot |

`Waveform` stores a transient vector resampled on a uniform grid, one row per sample.

### Fault campaign

`FaultCampaign` takes the same definitions as `Experiment` and writes the results to
a dataset folder. A long campaign can be interrupted and launched again: it continues
after the last complete chunk.

```python
from spicefault import Circuit, FaultCampaign, Measurement, SimulationConfig, Waveform
from spicefault.faults import FaultUniverse, open_rule, parametric_rule, short_rule
from spicefault.variation import tolerances

circuit = Circuit.from_netlist("examples/filter/rc_lowpass.cir")
universe = FaultUniverse(
    circuit, [open_rule(), short_rule(), parametric_rule([-0.5, -0.2, 0.2, 0.5])]
)

campaign = FaultCampaign(
    circuit,
    universe.selected(),
    out_dir="data/rc_lowpass",
    samples_per_fault=200,
    healthy_samples=2000,
    variations=tolerances(circuit, {"R": 0.01, "C": 0.05}),
    config=SimulationConfig(("op", "ac dec 20 1 1e5", "tran 10u 10m"), outputs=("v(out)",)),
    measurements=[
        Measurement.value("v(out)", name="dc"),
        Measurement.magnitude("v(out)", 1e3, name="gain_1k", db=True),
        Measurement.peak("v(out)", name="peak"),
    ],
    waveform=Waveform("v(out)", fs=10e3, duration=10e-3),
    seed=42,
)

if __name__ == "__main__":             # needed on macOS and Windows: workers are processes
    print(campaign.validate())         # definitions against the circuit, before running
    campaign.run(workers=8)            # resumes if it was interrupted
    print(campaign.status())           # total, completed, failed, pending
    print(campaign.summary())          # per fault: samples, success rate, each status
    samples, waveforms, manifest = campaign.load()
```

What the engine guarantees:

- the dataset is the same for any number of workers and any chunk size (only the
  `elapsed_s` column differs);
- a simulation that fails, an output that cannot be measured and a fault that cannot
  be injected are rows with a status and a message; none of them stops the run;
- a partial run is never mixed with a campaign whose definitions differ;
- the folder holds the complete definition of the experiment and its source netlist,
  the simulator and package versions, the number of workers, whether the run was
  resumed and the time spent over all its runs.

Functions passed to custom or joint variations and to custom measurements must be
defined at module level, so that worker processes can receive them.

### Dataset

A campaign writes a folder that stands on its own:

| File | Content |
|---|---|
| `samples.parquet` | One row per simulation: what was injected, its status, the realised component values, the measurements |
| `waveforms.npy` | float32 `[samples, points]`, aligned with the table row by row (if a waveform was declared) |
| `metadata.json` | The definition of the experiment: faults, variations, conditions, analyses, measurements, seed |
| `circuit.cir` | The source netlist |
| `manifest.json` | The record of the run: versions, platform, workers, counts by status, and the size and SHA-256 of every file |

```python
from spicefault import Dataset

dataset = Dataset("data/rc_lowpass")

dataset.verify()                 # [] if the files match the manifest and each other
dataset.provenance(1234)         # fault, seed key, realised values, versions of one sample
dataset.netlist(1234)            # the netlist that was simulated for it
dataset.reproduce(n=50)          # simulate samples again and compare with what is stored

X, y = dataset.to_ml(target="fault_location")       # for any ML or statistical tool
analysis = dataset.analysis()                       # a ReliabilityAnalysis
```

- `netlist` rebuilds a sample from the values stored in the table, so it needs
  nothing but the folder.
- `reproduce` rebuilds the experiment from `metadata.json` and reports, per sample,
  whether the definition, the drawn values and the status are the same, and the
  largest difference in the measurements and the waveform. On the machine and
  simulator version that wrote the dataset the differences are expected to be zero.
- Functions cannot be stored. If the experiment used custom or joint variations, or
  custom measurements, pass it again: `dataset.reproduce(experiment=experiment)`.
- The library has no ML dependency: `to_ml` returns NumPy arrays.

### Reliability analysis

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

## Validation circuits and experiments

[validation/](validation/) holds the circuits the framework is validated on. Each is
one module with what is specific to the circuit: its netlist, fault rules,
measurements and specification limits.

| Study | Circuit | Faults |
|---|---|---|
| `sallen_key` | Sallen–Key band-pass filter, a benchmark of the fault-diagnosis literature | 58 |
| `biquad` | Four-op-amp biquad high-pass filter, the other benchmark of that literature | 96 |
| `regulator` | Discrete series voltage regulator with device-level models; line, load and temperature conditions | 54 |

```bash
python -m validation.run_campaign --circuit sallen_key --out data/sallen_key
python -m validation.experiments.e_variability run --circuit sallen_key      # tolerance
python -m validation.experiments.f_conditions run                            # operating conditions
python -m validation.experiments.g_separability --circuit biquad             # ambiguity
```

These are long. [docs/RUNBOOK.md](docs/RUNBOOK.md) lists them in order with their size.

## Benchmarks

[benchmarks/](benchmarks/README.md) automates the measurements of the experiment
plan and stores each result as JSON under `benchmarks/results/`:

```bash
python -m benchmarks.scalability.run --workers 1 2 4 8   # runtime, speed-up, memory, I/O
python -m benchmarks.reproducibility.run --workers 8     # 1 worker against several, resumed runs
python -m benchmarks.fault_coverage.run                  # coverage matrix of each circuit
```

The timing benchmarks need an idle machine. Their workloads are the validation
circuits; with the Sallen–Key one, the same campaign is also run by a script written
directly against ngspice, as the reference for what the framework costs.

## Origin and use

The library was extracted from the simulation code of a study on self-diagnosis of
ECG analog front-ends
([ecg-frontend-fault-diagnosis](https://github.com/telmomm/ecg-frontend-fault-diagnosis)),
which uses it for its simulations. Nothing in the library is specific to that study
or to biomedical circuits.

During development the library was checked against that original code: identical
netlists, random numbers, measurements and datasets. Those tests are optional and are
described in [tests/regression/README.md](tests/regression/README.md).
