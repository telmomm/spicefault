# Datasets

A campaign writes a folder that stands on its own:

| File | Content |
|---|---|
| `samples.parquet` | One row per simulation: what was injected, its status, the realised component values, the measurements |
| `waveforms.npy` | float32 `[stored samples, points]` (if a waveform was declared): one row per sample of the conditions that store one, in sample order; `NaN` for a failed simulation. `Dataset.waveforms` reads it aligned with the table, with `NaN` for the conditions that store none |
| `metadata.json` | The definition of the experiment: faults, variations, conditions, analyses, measurements, seed |
| `circuit.cir` | The source netlist |
| `manifest.json` | The record of the run: versions, platform, workers, counts by status, the git commit of the project that ran it (`source`), and the size and SHA-256 of every file |

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

`split_by_replica(samples, test_fraction, seed)` returns train/test row positions while
keeping all conditions of each replica together and stratifying by `fault_id`.
`split_by_magnitude(samples, test_magnitudes)` holds out every row at the selected fault
magnitudes. Both return NumPy index arrays. Derived labels can be added with
`dataset.update_columns(frame, note=...)`; it updates the sample fingerprint and appends
the operation to manifest history, while definitions, parameters and measurements remain
protected.

## Specifications

Whether a fault was injected and whether the circuit still meets its specifications
are two different questions. `Dataset.label` answers the second from limits that stay
with the dataset:

```python
from spicefault import Specification

limits = [
    Specification("spec_cmrr_db", minimum=89.0),
    Specification("spec_noise_uvpp", maximum=30.0),
]
dataset.label(limits, note="IEC 60601-2-25 limits")

dataset.samples[["ok_spec_cmrr_db", "compliant", "violated"]]
dataset.specifications           # the limits, read back from the manifest
```

It writes `ok_<name>` for each specification, `compliant` and `violated` (the names of
those that are not met), stores the limits in the manifest and appends to its history.
Calling it again with other limits relabels the dataset without simulating anything.
A value that could not be computed, also because the simulation failed, counts as a
violation. A specification belongs to the drawn circuit: with several operating
conditions, the rows of one circuit get the same labels, read in the conditions that
measure each quantity.

## One row per drawn circuit

When operating conditions have their own measurements, the table has one row per
circuit and condition, and each row holds only the measurements of its condition.
`Dataset.cases` joins them:

```python
cases, waveforms = dataset.cases(waveform_from="service")
cases.attrs["features"]          # the measurement columns, of every condition

X, y = dataset.to_ml(by_case=True)
analysis = dataset.analysis(by_case=True, compliant="compliant")
```

There is one row per fault and replica: the definition, the labels and the realised
parameters once, and the measurements of every condition side by side. A measurement
that several conditions make gets one column per condition, `<name>_<condition>`.
`sim_ok` is true only if every condition succeeded. `waveform_from` may be omitted
when a single condition stores the waveform.

## Columns of the samples table

| Columns | Content |
|---|---|
| `sample_id` | The row number, and the position of the sample in the plan of the experiment |
| `fault_index`, `fault_id`, `fault_type`, `fault_location`, `fault_magnitude`, `fault_severity` | What was injected; `healthy` for the fault-free circuits |
| `replica`, `condition`, `seed_key` | Which draw, under which operating condition, from which random stream |
| `status`, `message`, `sim_ok` | The outcome of the simulation |
| `elapsed_s` | Time of the simulation; the only column that is not reproducible |
| labels | Quantities recorded by joint and catalogue variations, fault tags, and derived labels such as `compliant` |
| `p_<component>_<parameter>` | The value each varied parameter was drawn with, before the fault |
| measurements | One column per measurement; empty for a failed simulation and for a condition that does not make it |
