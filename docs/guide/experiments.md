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

`Waveform` stores a transient vector resampled on a uniform grid, one row per sample.

`SimulationConfig` lists the analyses. An entry that is not an analysis, such as
`"alter @vin[acmag]=1"`, is run in its place and saves nothing, so a source can be
changed between two sweeps:

```python
SimulationConfig(("op", "alter @v1[acmag]=1", "ac dec 20 1 1e5"), outputs=("v(out)",))
```

## Fault campaign

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
