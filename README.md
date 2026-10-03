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
| 4. Uncertainty framework (normal, uniform and log-normal variations) | Not started |

The first application is the ECG front-end study
([ecg-frontend-fault-diagnosis](https://github.com/telmomm/ecg-frontend-fault-diagnosis)),
whose generic code was extracted into this package.

| Module | Content |
|---|---|
| `spicefault.circuit` | `Circuit`: a netlist seen as components, nodes and parameters |
| `spicefault.faults` | Fault types, each a list of primitives with its metadata; severity scales; `FaultSet`; `FaultUniverse` with its coverage matrix |
| `spicefault.variation` | `ToleranceVariation`, `VariationSet`; tolerance draws: uniform, truncated normal, log-uniform |
| `spicefault.conditions` | `OperatingCondition`: settings and temperature, applied after the fault |
| `spicefault.simulation` | `Simulator`, `SimulationConfig`, `SimulationResult` with an explicit status; ngspice backend, batch runner and raw-file reader |
| `spicefault.experiments` | `Experiment`; per-sample random streams; parallel, chunked and resumable execution of a campaign |
| `spicefault.netlist` | Netlist as text and the three fault-injection primitives |
| `spicefault.measurements` | Interpolation of frequency responses, resampling of transients, ADC quantisation |
| `spicefault.dataset` | Dataset directory: `samples.parquet`, `waveforms.npy`, `manifest.json` |

`Experiment.run` keeps its results in memory. Writing them to a dataset in chunks,
with resumption, is done today by the lower-level `run_campaign`; joining the two
belongs to the experiment engine (phase 5).

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
| `test_ecg_injection.py` | Netlist of every fault condition of both ECG circuits, in service and on the test bench, built by `ecgfd` and by `Fault` plus `OperatingCondition` | Identical text |
| `test_ecg_universe.py` | Fault universe generated from rules against the fault catalogue of `ecgfd`: 293 and 307 conditions | Same conditions, identical netlists |
| `test_ecg_experiment.py` | Self-test measurement of faulty circuits through `Circuit`, `Fault` and `Simulator` | Identical vectors |
| `test_ecg_campaign.py` | A reduced campaign run by both engines, with 1 and with several workers | Identical tables and waveforms |
| `test_ecg_baseline_data.py` | Cases regenerated here against the published dataset `data/v1` | Tolerances of docs/EXPERIMENT_PLAN.md §3 |

The last one needs the dataset; its location is taken from the `ECGFD_DATA`
environment variable and defaults to `data/v1` in the ECG repository.
