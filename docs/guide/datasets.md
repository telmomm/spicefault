# Datasets

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

## Columns of the samples table

| Columns | Content |
|---|---|
| `sample_id` | The row number, and the position of the sample in the plan of the experiment |
| `fault_index`, `fault_id`, `fault_type`, `fault_location`, `fault_magnitude`, `fault_severity` | What was injected; `healthy` for the fault-free circuits |
| `replica`, `condition`, `seed_key` | Which draw, under which operating condition, from which random stream |
| `status`, `message`, `sim_ok` | The outcome of the simulation |
| `elapsed_s` | Time of the simulation; the only column that is not reproducible |
| labels | Quantities recorded by joint variations |
| `p_<component>_<parameter>` | The value each varied parameter was drawn with, before the fault |
| measurements | One column per measurement; empty for a failed simulation |
