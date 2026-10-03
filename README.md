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

## Status

| Phase | State |
|---|---|
| 0. Scientific specification | Drafted, open decisions listed at the end of each document |
| 1. Extraction from the ECG project | Done: the generic parts of `ecgfd` live here and are checked against it |
| 2. Core abstractions | Done: `Circuit`, `Fault`, `VariationSet`, `OperatingCondition`, `Simulator`, `SimulationResult`, `Experiment` |
| 3. Fault framework | Done: `OpenCircuit`, `ShortCircuit`, `LeakageFault`, `ParametricFault`, `CompositeFault`, `FaultSeverity`, `FaultSet`, `FaultUniverse` |
| 4. Uncertainty framework | Done: tolerance, normal, log-normal, uniform, log-uniform, fixed, custom and joint variations; three seeding schemes |
| 5. Experiment engine | Done: `FaultCampaign` (to disk, in chunks, resumable, every simulation accounted for) and the `Measurement` API |
| 6. Reliability analysis (detectability, sensitivity, robustness, separability) | Not started |

The first application is the ECG front-end study
([ecg-frontend-fault-diagnosis](https://github.com/telmomm/ecg-frontend-fault-diagnosis)),
whose generic code was extracted into this package.

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
| `spicefault.dataset` | Dataset directory: `samples.parquet`, `waveforms.npy`, `manifest.json` |

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
| `positional` (default) | fault index, replica | As in the ECG baseline. Adding or reordering faults changes the samples of the others |
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
- the manifest holds the complete definition of the experiment, the simulator and
  package versions, the number of workers, and whether the run was resumed.

Functions passed to custom or joint variations and to custom measurements must be
defined at module level, so that worker processes can receive them.

## Equivalence with the ECG baseline

`tests/regression/` compares this package with `ecgfd`, the code it was extracted
from. These tests run only if `ecgfd` is installed:

```bash
pip install -e ../ecg-frontend-fault-diagnosis
pytest tests/regression
```

| Test | What is compared | Criterion |
|---|---|---|
| `test_ecg_runner.py` | Output of the same decks through both ngspice runners | Identical vectors |
| `test_ecg_sampling.py` | Random streams and tolerance draws | Identical numbers |
| `test_ecg_variation.py` | The whole healthy population (passives, amplifiers, electrodes) drawn by a `VariationSet`, and the netlists `Experiment` builds from it | Identical text |
| `test_ecg_injection.py` | Netlist of every fault condition of both ECG circuits, in service and on the test bench, built by `ecgfd` and by `Fault` plus `OperatingCondition` | Identical text |
| `test_ecg_universe.py` | Fault universe generated from rules against the fault catalogue of `ecgfd`: 293 and 307 conditions | Same conditions, identical netlists |
| `test_ecg_experiment.py` | Self-test measurement of faulty circuits through `Circuit`, `Fault` and `Simulator` | Identical vectors |
| `test_ecg_service_dataset.py` | The self-test half of `data/v1` (features and waveforms) regenerated with spicefault objects only | Tolerances of docs/EXPERIMENT_PLAN.md §3; no difference found |
| `test_ecg_campaign.py` | A reduced campaign run by both engines, with 1 and with several workers | Identical tables and waveforms |
| `test_ecg_baseline_data.py` | Cases regenerated here against the published dataset `data/v1` | Tolerances of docs/EXPERIMENT_PLAN.md §3 |

The last one needs the dataset; its location is taken from the `ECGFD_DATA`
environment variable and defaults to `data/v1` in the ECG repository.
