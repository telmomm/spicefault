# Comparison with existing approaches

Status: first version. It covers what could be established without long runs: what a direct script and two Python libraries provide, and the code each approach needs. The throughput comparison needs the scalability benchmark on an idle machine ([RUNBOOK.md](RUNBOOK.md)).

The criteria are those of [SCIENTIFIC_SCOPE.md](SCIENTIFIC_SCOPE.md) §7. This is a comparison of scope. None of the entries says that a tool cannot be used to build the missing part: with any of them one can write what is absent, and the direct script of the first column is exactly that exercise.

## 1. What was examined, and how

| Approach | Version | How it was examined |
|---|---|---|
| Direct script against ngspice | `validation/direct/sallen_key_direct.py` in this repository | Written for the Sallen–Key campaign, with the same circuit, tolerances, faults, analyses and measurements as the study. Tested to give the same measurements as `spicefault` for every fault |
| spicelib | 1.6.3 | Installed from PyPI; its modules and classes listed. Not run |
| PySpice | 1.5 | Installed from PyPI; its modules listed and searched. Not run |
| `spicefault` | this repository | Its tests |

Commercial analog fault simulators and the tools of the literature are discussed in [RELATED_WORK.md](RELATED_WORK.md); they could not be examined and are not in the table.

**The entries for spicelib and PySpice rest on reading names of modules and classes, not on documentation or use.** A capability provided under a name that was not looked for would be missed. They must be confirmed by reading the documentation and by implementing the benchmark task with each, before any of this goes into the manuscript.

## 2. Capabilities

"Written" means that the script contains code for it; "—" that it does not.

| Criterion | Direct script | spicelib 1.6.3 | PySpice 1.5 | `spicefault` |
|---|---|---|---|---|
| Runs ngspice and reads its output | Written | Yes; also LTspice, QSPICE and Xyce | Yes; also Xyce | Yes, ngspice only |
| Netlist editing | Written (text template) | Yes (editor classes) | Yes (circuit built in Python) | Yes (three primitives on netlist text) |
| Tolerance analysis | Written (uniform draws) | Yes: `Montecarlo`, `WorstCaseAnalysis`, `FastWorstCaseAnalysis`, `QuickSensitivityAnalysis` | Not found | Yes: eight kinds of variation, scalable |
| Fault abstraction | Written (three kinds, hard-coded) | Not found: no module or class for faults | Not found | Yes: five fault types, metadata, serialisation |
| Variation kept apart from fault, fault applied to a drawn circuit | Written | Not applicable without a fault abstraction | Not applicable | Yes |
| Operating conditions as objects | — | Not looked for | Not looked for | Yes |
| Fault universe from rules, coverage report | — | Not found | Not found | Yes |
| Parallel execution | Written (`multiprocessing`) | Yes (`SimRunner`, `parallel_sims`) | Not found | Yes |
| Resumable campaign | — | Not found | Not found | Yes |
| Status of each simulation | Ok or not | Not looked for | Not looked for | Five statuses, with message |
| Samples independent of the number of workers and of the fault list | Of the workers, yes; of the fault list, no (seed by position) | Not looked for | Not applicable | Yes (`content` seeding) |
| Per-sample provenance and a record of the definition | — | Not looked for | Not found | Yes |
| Measurement extraction | Written | Not looked for | Not looked for | Yes |
| Reliability metrics | — | Not found | Not found | Yes |
| Dataset with integrity check and reproduction | CSV and array | Not found | Not found | Yes |

Two observations, both subject to the caveat of §1:

- spicelib is the closest open library: it covers tolerance analysis and parallel batches for four simulators. What was not found in it is the fault side: fault types, a fault universe, and analyses that compare a faulty population with a healthy one.
- PySpice is an interface to the simulator, as its own description says; campaign-level functions were not found in it.

`spicefault` supports one simulator. spicelib supports four, and that is a point in its favour that the manuscript must state.

## 3. Code needed for one study

The Sallen–Key campaign: 58 faults, tolerances on eight parameters, two analyses, ten measurements, a waveform.

| | Direct script | `spicefault` |
|---|---|---|
| Code specific to the study | 180 lines (`sallen_key_direct.py`) | 59 lines (`validation/sallen_key.py`) and 12 netlist lines |
| Measurements written for the study | Included in the 180 | 37 lines (`validation/functions.py`), shared by the two filters |
| Shared by all studies | — | 81 lines (`validation/study.py`) |
| Second and third circuit | A new script each | 53 lines (biquad) and 101 lines (regulator) |

Lines counted are lines of code, without blank lines, comments and docstrings. As a measure of effort this is weak: it ignores the time to get a script right, and the study relies on a library that had to be written. What it does show is where the work goes: in the script, most lines are plumbing (building the netlist text, running the simulator, reading its file, the process pool, writing files), repeated for every circuit; in the study, the lines are the definitions of the circuit, its faults and its measurements.

What the script does not have, and the study gets from the framework, is listed at the top of the script itself: resuming, a status and a message per simulation, the record of the experiment, a seed that does not depend on the position of a fault, and everything after the simulation (integrity, provenance, reproduction, metrics).

## 4. Throughput

Not measured yet. The scalability benchmark runs the same campaign through both, at each number of workers, and reports the ratio. A short run on a loaded machine, in the tests, only shows that both work.

## 5. Still to do

1. Read the documentation of spicelib and PySpice, and implement the Sallen–Key task with each.
2. Run the scalability benchmark with the `sallen_key` workload on an idle machine.
3. Add the tools of the literature whose full text can be read (RELATED_WORK.md §4).
