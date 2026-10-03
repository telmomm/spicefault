# Benchmarks

Automated measurements for the experiments of [docs/EXPERIMENT_PLAN.md](../docs/EXPERIMENT_PLAN.md).
Each one prints a summary and writes its full result as JSON under
`benchmarks/results/<benchmark>/`, with the record of the environment it ran in.

Run them from the repository root:

```bash
python -m benchmarks.scalability.run --workers 1 2 4 8
python -m benchmarks.reproducibility.run --workers 8
python -m benchmarks.fault_coverage.run
```

| Benchmark | Experiment | What it measures |
|---|---|---|
| `scalability` | C | Wall time and simulations per second for each number of workers, speed-up and efficiency; time of one sample by phase; peak memory; bytes written; cost of resuming |
| `reproducibility` | B | The same campaign with 1 and with several workers, repeated, with another chunk size, and interrupted and resumed: are the sample definitions identical (L1), and how much do the outputs differ (L2)? Then samples simulated again from the dataset folder |
| `fault_coverage` | D | For each circuit: components, fault types, magnitudes, fault conditions, the coverage matrix, exclusions, components outside the fault model, and faults partly inside the tolerance band |

## Workloads

| Name | Circuit | Needs |
|---|---|---|
| `rc` | RC low-pass filter, 30 faults, three analyses per simulation (about 15 ms each) | ngspice |

The RC filter only checks that the benchmarks work. A simulation of it is so short
that starting the worker processes takes a large part of the time, so its timings
say little about the framework. The validation circuits of the experiment plan are
to be added to `workloads.py` as they are written, and the figures of the manuscript
come from them.

Not implemented yet: the comparison with a script written directly against ngspice
for the same task, which measures what the framework costs.

## Protocol

The defaults follow section 4 of the experiment plan: a fixed workload of 2000
samples, 5 repetitions, one discarded warm-up run, median and range reported. The
machine must be idle and on mains power; the result records the load average at the
start, which is the only evidence of that the benchmark can collect. Timings taken
while anything else is running must not be kept.

## Reading the results

- **Speed-up and efficiency** are relative to the run with the fewest workers. On a
  processor with performance and efficiency cores, efficiency falls beyond the number
  of performance cores for hardware reasons; the core counts are in the result.
- **Wall time includes** starting the worker processes and writing the dataset.
- **Phase breakdown** is measured in one process over 100 samples: netlist generation,
  the ngspice process, reading its raw file, measurements, and storage.
- **Peak memory** is the peak resident size of the main process and of its largest
  child process (a worker or an ngspice run), not their sum.
- **Resume overhead** is the time to launch again a campaign whose chunks are all
  complete. An interruption loses at most the chunk in progress; complete chunks are
  never simulated again.

## Result files

JSON, one file per run, named by UTC time and label. Common fields: `benchmark`,
`environment` (date, host, platform, processor and cores, Python and package
versions, `spicefault` version and commit, simulator version, load average),
`protocol`, and the measurements. Every timing of every repetition is kept under
`runs`, not only the summary.

`tests/benchmarks/` runs each benchmark at a tiny size to check that it works and
that its result has the expected form.
