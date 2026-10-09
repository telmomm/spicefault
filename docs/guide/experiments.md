# Experiments and campaigns

`Experiment` defines what is simulated and runs it in memory. `FaultCampaign` runs the
same definition to disk, in chunks, and can be resumed.

## Experiment

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

`experiment.metadata()` returns everything needed to regenerate the samples: circuit
hash, fault records, variations, conditions, seed, simulator and package versions.

## Measurements

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
| `custom_result` | Any function of the complete result, including multiple plots |

`Waveform` stores a transient vector resampled on a uniform grid, one row per sample.

`SimulationConfig` lists the analyses. An entry that is not an analysis, such as
`"alter @vin[acmag]=1"`, is run in its place and saves nothing, so a source can be
changed between two sweeps:

```python
SimulationConfig(("op", "alter @v1[acmag]=1", "ac dec 20 1 1e5"), outputs=("v(out)",))
```

Outputs can be selected per analysis, and `expected_plots` validates plot order even when
the netlist owns its `.control` block. Noise supports selecting ngspice's spectrum or
integrated-noise plot:

```python
SimulationConfig(
    analyses=(("op", ("v(out)", "v(ref)")),
              ("noise v(out) V1 dec 20 1 1e5", ("onoise_spectrum",), "integrated")),
    expected_plots=("Operating Point", "Integrated Noise"),
)
```

An `OperatingCondition` can override `config`, `measurements` and `waveform`; measurements
not declared for a condition are stored as `NaN` in their dataset columns. See
[From a schematic](from-schematic.md) for imported netlists.

```python
OperatingCondition("service", config=..., measurements=..., waveform=Waveform("v(out)", 1e3, 1.0))
OperatingCondition("bench", config=..., measurements=...)        # stores no waveform
```

The waveform of the experiment is the default of the conditions that do not give one;
`waveform=False` declines it. The rows of a condition without waveform hold `NaN`, and
every stored waveform has the same number of points. `Dataset.cases` joins the rows of
one drawn circuit across its conditions (see [Datasets](datasets.md)).

When several values come out of one evaluation, a group computes them in one call and
gives each a column:

```python
def specifications(result):          # at module level
    ...
    return {"spec_gain_error": gain_error, "spec_cmrr_db": cmrr}

Measurement.group(("spec_gain_error", "spec_cmrr_db"), specifications)
```

## The nominal circuit

`experiment.nominal()` simulates the circuit with the values of its netlist, with no
variation drawn, under each operating condition, and returns `{condition: SampleResult}`:
the reference against which an error is defined, or the response to draw next to the
Monte Carlo band.

```python
nominal = experiment.nominal()
nominal["bench"].measurements["gain"]
nominal["bench"].result                       # the complete simulation result
experiment.nominal(fault="R1:open")           # the nominal circuit with one fault
```

`experiment.evaluate(values, fault=None)` is the same with given parameter values, as
`{(component, parameter): value}`: what an optimiser or a sensitivity method that brings
its own points needs. The [scope page](../scope.md) has a recipe for each.

```python
result = experiment.evaluate({("R1", "value"): 10.1e3, ("C1", "value"): 95e-9})
result["bench"].measurements["gain"]
```

## Designed samples and corners

The circuits of an experiment are drawn from its variations. They can instead be
chosen: a `Design` is a table of parameter values, one row per circuit, run by the same
engine, so a corner study is resumable, keeps its failed simulations and ends in a
dataset.

```python
from spicefault import Campaign, corners
from spicefault.variation import tolerances

design = corners(tolerances(circuit, {"R": 0.01, "C": 0.05}), circuit)   # 2^n corners, and the nominal
campaign = Campaign(circuit, out_dir="data/corners", design=design, measurements=[...])
campaign.run()
campaign.dataset().extremes()     # lowest and highest value of each measurement, and where
```

- `corners` takes the band of each variation (a tolerance, a range, a truncated normal)
  and refuses more corners than its `limit`, 1024 by default: the count doubles with
  every parameter.
- `Design.from_table(frame)` takes any table of values, such as the points another
  tool computed. The columns are `component`, `component.parameter` or a pair.
- Faults and operating conditions combine with a design as with drawn circuits: each
  fault is injected into every row.
- The row of each sample is recorded in the column `design_point`, and the design in
  the metadata of the dataset.

Two limits are part of the interface, not footnotes:

- **Corners bound a response only where it is monotonic.** The extremes over the
  corners are the extremes over the band if the measurement rises or falls with each
  parameter throughout the band. A response with a maximum or a minimum inside it, such
  as a resonance, a notch or an error that two parameters compensate, has its worst case
  where no corner is.
- **Chosen circuits are not a random sample.** `statistics`, `yield_report` and
  `analysis` refuse a designed dataset: a proportion over corners is not a probability.

## Fault campaign

`FaultCampaign` takes the same definitions as `Experiment` and writes the results to
a dataset folder. It is the class `Campaign`: without faults, the same campaign is a
[Monte Carlo study](monte-carlo.md) of the healthy population. A long campaign can be interrupted and launched again: it continues
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
  resumed and the time spent over all its runs;
- when the campaign is run from inside a git repository, the manifest records its
  commit and whether its tracked files had uncommitted changes, under `source`.

Functions passed to custom or joint variations and to custom measurements must be
defined at module level, so that worker processes can receive them.

A campaign folder is run by one process at a time: a second launch on a folder whose
campaign is running is refused.

## Seeding

`Experiment(..., seeding=...)` chooses what identifies the random stream of a sample.
In every scheme the stream depends only on the seed and on that key, never on the
number of workers or on the order of execution.

| Scheme | Key | Consequence |
|---|---|---|
| `positional` (default) | fault index, replica | The simplest. Adding or reordering faults changes the samples of the others |
| `content` | hash of the fault identifier, replica | A fault has the same samples in every experiment that contains it |
| `common` | replica | Every fault is applied to the same drawn circuits: paired comparisons, but the conditions are not independent |

Each sample records its key (`seed_key`), from which it can be drawn again alone.
