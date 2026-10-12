# Evaluation data for spicefault

Fault-injection campaigns and benchmark results on three analog circuits, produced
with [spicefault](https://github.com/telmomm/spicefault), a Python library for
reproducible SPICE-based fault injection.

| | |
|---|---|
| spicefault | version 0.4.0, commit `7a189f0d9bb9aafa62cbc82072d72d5c23e010c9`, clean working tree |
| Simulator | ngspice 44.2 |
| Python, platform | 3.14.5, macOS 26.6.2, arm64 (Apple M2, 4 performance and 4 efficiency cores) |
| Date of the runs | 9 October 2026 |
| Seed | 42, seeding by content: a fault has the same samples whatever the other faults |

## Files

| File | Content | Size |
|---|---|---|
| `sallen_key.zip` | 4 campaigns on the Sallen-Key band-pass filter | 34 MB |
| `biquad.zip` | 4 campaigns on the four-op-amp biquad high-pass filter | 104 MB |
| `regulator.zip` | 6 campaigns on the discrete series voltage regulator | 23 MB |
| `results.zip` | What the analyses and the benchmarks computed (JSON and CSV) | 0.3 MB |
| `SHA256SUMS` | Checksums of the four archives | |

## The circuits

- **Sallen-Key band-pass filter**, centre frequency 25 kHz: Fig. 3 of M. Aminian and
  F. Aminian, "Neural-network based analog-circuit fault diagnosis using wavelet
  transform as preprocessor", *IEEE Trans. Circuits Syst. II*, vol. 47, no. 2,
  pp. 151-156, 2000, doi:10.1109/82.823545. 58 fault conditions.
- **Four-op-amp biquad high-pass filter**, cut-off 10 kHz: Fig. 2 of F. Aminian,
  M. Aminian and H. W. Collins, "Analog fault diagnosis of actual circuits using neural
  networks", *IEEE Trans. Instrum. Meas.*, vol. 51, no. 3, pp. 544-550, 2002,
  doi:10.1109/TIM.2002.1017726. 104 fault conditions. R1 to R4, C1 and C2 are named as
  in that paper; its figure repeats two labels on the other six resistors, which are
  R5 to R10 here.
- **Discrete series voltage regulator**, 15 V to about 10 V into 100 ohm: a Darlington
  pass transistor (FZT603), a one-transistor error amplifier (DMMT3904W) and a zener
  reference (BZX84C4V7), with the SPICE models of their manufacturer. 54 fault
  conditions.

Faults are opens, shorts and parametric deviations (-50, -25, -10, +10, +25 and +50 %)
of the passives, loss of gain and of bandwidth of the amplifiers, and opens and shorts
at the terminals of the transistors and of the zener. Tolerances are 5 % for resistors
and 10 % for capacitors in the filters, as in the two papers, and 1 % and 20 % in the
regulator. The excitation of the filters is a single pulse of 5 V and 10 us.

## The campaigns

Every campaign is a folder with the same four or five files.

| Folder | Experiment | Samples |
|---|---|---|
| `<circuit>/scale_0`, `scale_0.2`, `scale_1`, `scale_2` | Effect of tolerance: the same campaign with the tolerances multiplied by 0, 0.2, 1 and 2 | 5000 healthy and 200 per fault: 16,600 (Sallen-Key), 25,800 (biquad), 15,800 (regulator) |
| `regulator/conditions_one_factor` | Operating conditions, one factor at a time: nominal, low and high line, heavy and light load, -20 C and 85 C | 2000 healthy and 200 per fault at each of 7 conditions: 89,600 |
| `regulator/conditions_corners` | Nominal and the 8 corners of input voltage (13.5, 18 V), load (68, 1000 ohm) and temperature (-20, 85 C) | The same at 9 conditions: 115,200 |

All 414,400 simulations ended with status `SUCCESS`.

| File | Content |
|---|---|
| `samples.parquet` | One row per simulation: the fault (`fault_id`, `fault_type`, `fault_location`, `fault_magnitude`, `fault_severity`), the operating condition, the status of the simulation, the value drawn for every varied parameter (`p_<component>_<parameter>`) and every measurement |
| `waveforms.npy` | The sampled response to the pulse, one row per row of `samples.parquet` (filters only: 150 points for the Sallen-Key filter, 300 for the biquad) |
| `circuit.cir` | The netlist of the fault-free circuit |
| `metadata.json` | The definition of the experiment: seed, faults, tolerances, conditions, analyses, measurements, hash of the netlist |
| `manifest.json` | How it was run: versions, commit, date, workers, file checksums, counts by status |

A row whose `fault_id` is `healthy` is a fault-free circuit; the others name the
component and the fault, as in `R1:open` or `R1:parametric:-0.5`.

```python
import numpy as np
import pandas as pd

samples = pd.read_parquet("sallen_key/scale_1/samples.parquet")
waveforms = np.load("sallen_key/scale_1/waveforms.npy")
```

## The results

`results.zip` holds two folders, as they are in the repository.

| Folder | Content |
|---|---|
| `benchmarks/results/correctness` | Every fault of every circuit simulated through spicefault and with `ngspice -b` launched apart: 219 decks and 408,938 values, compared bit by bit |
| `benchmarks/results/reproducibility` | The same campaign with 1 and with 8 workers, repeated, with another chunk size, and interrupted and resumed |
| `benchmarks/results/scalability` | Wall time and throughput at 1, 2, 4 and 8 workers, with every timing of every repetition; for the Sallen-Key filter, also against a script written directly for ngspice |
| `benchmarks/results/fault_coverage` | The fault universe of each circuit: coverage matrix, exclusions, faults partly inside the tolerance band |
| `benchmarks/results/sampling` | Random, Latin hypercube and Sobol sampling at equal numbers of simulations |
| `results/e_variability` | Failure probability and detection of every fault at each tolerance scale |
| `results/f_conditions` | Detection of every fault of the regulator at each operating condition |
| `results/g_separability` | Which faults and which components can be told apart from the measurements and from the waveform |

Every benchmark file records the environment it ran in. The timings were taken on a
fanless laptop with performance and efficiency cores, and the load average rose during
the measurement: they are a record of these runs and not a characterisation of the
library.

## Regenerating the data

With spicefault at the commit above and ngspice 44.2, from the root of the repository:

```bash
python -m validation.experiments.e_variability run --circuit sallen_key --workers 8
python -m validation.experiments.e_variability run --circuit biquad --workers 8
python -m validation.experiments.e_variability run --circuit regulator --workers 8
python -m validation.experiments.f_conditions run --conditions one_factor --workers 8
python -m validation.experiments.f_conditions run --conditions corners --workers 8
```

The number of workers does not change the result. `docs/RUNBOOK.md` in the repository
gives the commands of the analyses and of the benchmarks.

## Known limits

- The device models of the regulator are those of the manufacturer, not measured parts,
  and the spread of the devices themselves is not modelled: only the passives vary.
- The zener model alone drifts +3.13 mV/K where its data sheet gives -3.5 to +0.2 mV/K.
  The netlist adds a declared source in series that brings it to the middle of that
  range. The size of the temperature drift of the regulator is therefore one plausible
  value, not that of a measured part.
- Self-heating is not simulated.
- No simulation failed, so these data do not show how failed simulations are recorded.

## Licence

The data are under the Creative Commons Attribution 4.0 International licence
(CC BY 4.0), with one exception. The files `regulator/*/circuit.cir` hold SPICE models
of Diodes Incorporated, copied intact under the notice of the manufacturer, which is
reproduced in those files and allows their use for research and design only.
