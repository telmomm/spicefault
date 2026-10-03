# Equivalence with the code the library was extracted from

`spicefault` began as the simulation code of a study on self-diagnosis of ECG analog
front-ends (package `ecgfd`, repository `ecg-frontend-fault-diagnosis`). These tests
compare the library with that code, piece by piece and end to end. They are how its
correctness was established during development. They are not part of what the
library is about, and nothing else in the repository depends on them.

They are skipped unless `ecgfd` is installed:

```bash
pip install -e ../ecg-frontend-fault-diagnosis
pytest tests/regression
```

## They are tied to one version of `ecgfd`

The tests compare against the implementation `ecgfd` had before it adopted this
library: its own ngspice runner, sampling, fault injection, dataset generation and
testability code. Once `ecgfd` uses `spicefault` for those, the comparison is of the
library with itself, or fails because the modules are gone.

They are valid against `ecgfd` at commit `6a49f58` and the dataset `data/v1` generated
by it. To keep them runnable, tag that commit before migrating, and install that tag
when running them. Otherwise they should be retired at the migration; their result is
recorded here.

## What was compared

| Test | What is compared | Result |
|---|---|---|
| `test_ecg_runner.py` | Output of the same decks through both ngspice runners | Identical vectors |
| `test_ecg_sampling.py` | Random streams and tolerance draws | Identical numbers |
| `test_ecg_variation.py` | The whole healthy population (passives, amplifiers, electrodes) drawn by a `VariationSet`, and the netlists `Experiment` builds | Identical text |
| `test_ecg_injection.py` | Netlist of every fault condition of both circuits, in service and on the test bench, built by `ecgfd` and by `Fault` plus `OperatingCondition` | Identical text |
| `test_ecg_universe.py` | Fault universe generated from rules against the fault catalogue: 293 and 307 conditions | Same conditions, identical netlists |
| `test_ecg_experiment.py` | Self-test measurement of faulty circuits through `Circuit`, `Fault` and `Simulator` | Identical vectors |
| `test_ecg_campaign.py` | A reduced campaign run by both engines, with 1 and with several workers | Identical tables and waveforms |
| `test_ecg_baseline_data.py` | Cases simulated through the library against the dataset `data/v1` | No difference |
| `test_ecg_service_dataset.py`, `test_ecg_dataset.py` | The self-test half of `data/v1` (features and waveforms) regenerated with library objects only, for the whole fault catalogue | No difference |
| `test_ecg_reliability.py` | Fault dictionary, separation, ambiguity groups, limit test, escape and false-reject rates, sensitivities, on `data/v1` | Identical results |

Not covered: the specification tests on the bench, which are the second ngspice run
of each case of `ecgfd` and use analyses of their own; and the two `ina_cmrr` fault
conditions, whose injected value depends on a sign drawn per sample.

## `ecg_adapter.py`

This file is the ECG study written with the library: its fault catalogue as fault
types and as universe rules, its healthy population as a `VariationSet`, its test
bench as an `OperatingCondition`, and its self-test measurements as an `Experiment`.
It is the starting point for moving `ecgfd` onto `spicefault`, and belongs in that
repository once it does.

How the fault kinds of `ecgfd` map onto the fault types of the library:

| `ecgfd` kind | Type | What it does |
|---|---|---|
| `open` | `OpenCircuit` | 1 GΩ in series with the second terminal |
| `short` | `ShortCircuit` | 1 Ω across the component |
| `parametric` | `ParametricFault` | Relative deviation of the drawn value |
| `cap_degradation` | `CompositeFault` | Capacitance scaled down and a series resistance added |
| `opamp_vos`, `ina_vos` | `ParametricFault` | Offset set to a value, replacing the drawn one |
| `opamp_aol` | `ParametricFault` | Open-loop gain multiplied by a factor |
| `ina_cmrr` | `ParametricFault` | Rejection set to a signed ratio; the sign is drawn per sample, so this fault is built per sample |
| `ina_gain` | `ParametricFault` | Gain error set to a value |
| `electrode_off` | `ParametricFault`, reported under its own name | Series resistance of the electrode replaced by 1 GΩ |
| `electrode_high_z` | `CompositeFault` | Resistance multiplied and capacitance divided by a factor, on one or two electrodes |
