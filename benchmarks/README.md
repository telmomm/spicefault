# Benchmarks

Automated measurements for the experiments of [docs/EXPERIMENT_PLAN.md](../docs/EXPERIMENT_PLAN.md).
Each one prints a summary and writes its full result as JSON under
`benchmarks/results/<benchmark>/`, with the record of the environment it ran in.

Run them from the repository root:

```bash
python -m benchmarks.correctness.run
python -m benchmarks.scalability.run --workload sallen_key --workers 1 2 4 8
python -m benchmarks.reproducibility.run --workload biquad --workers 8
python -m benchmarks.fault_coverage.run
python -m benchmarks.sampling.run --sizes 64 256 --repetitions 16 --workers 8
```

| Benchmark | Experiment | What it measures |
|---|---|---|
| `correctness` | A | For each validation circuit, the healthy circuit and every fault, simulated through `spicefault` and with `ngspice -b` launched apart and read by a raw-file reader of its own: are the vectors identical bit by bit? |
| `scalability` | C | Wall time and simulations per second for each number of workers, speed-up and efficiency; time of one sample by phase; peak memory; bytes written; cost of resuming |
| `reproducibility` | B | The same campaign with 1 and with several workers, repeated, with another chunk size, and interrupted and resumed: are the sample definitions identical (L1), and how much do the outputs differ (L2)? Then samples simulated again from the dataset folder |
| `fault_coverage` | D | For each circuit: components, fault types, magnitudes, fault conditions, the coverage matrix, exclusions, components outside the fault model, and faults partly inside the tolerance band |

| `sampling` | - | Error of the mean and of the yield for random, Latin hypercube and Sobol sampling at equal numbers of simulations, on a circuit with a known answer |

## Workloads

| Name | Circuit | Needs |
|---|---|---|
| `sallen_key`, `biquad`, `regulator` | The validation studies of `validation/` | ngspice |
| `rc` | RC low-pass filter, 30 faults; only to check that the benchmarks work | ngspice |

The figures of the manuscript come from the validation studies. With `sallen_key`,
the scalability benchmark also runs the campaign with
`validation/direct/sallen_key_direct.py`, a script written directly against ngspice
for the same task, and reports the throughput of `spicefault` relative to it.

## Protocol

The defaults follow section 4 of the experiment plan: a fixed workload of 2000
samples, 5 repetitions, one discarded warm-up run, median and range reported. The
machine must be idle and on mains power. The scalability benchmark first waits, up to
four minutes, for the load average to fall below 0.4 per core: a job that has just
ended is still in that average for about a minute. The result records the load it
started with and the load before every timed run, which is the only evidence of an
idle machine the benchmark can collect. Timings taken
while anything else is running must not be kept.

## Reading the results

- **Correctness** compares the execution and the reading of the output on the deck the
  library built, one sample per fault. It does not check that the netlist of a fault
  is the right one: `tests/validation/test_direct_script.py` does, against the direct
  script. It is not a timing: it needs no idle machine and takes seconds.

- **Range** is the difference between the slowest and the fastest repetition, relative
  to the median. It is the evidence that the machine was undisturbed: a few per cent
  when it was. The load average is recorded too, but on a desktop with an editor and
  background services it rests well above zero and says less.
- **A laptop slows down as it warms up.** On a fanless machine the later repetitions
  of a long benchmark are slower than the first. The direct script is timed right
  after the framework in every repetition, so their ratio is not affected; the absolute
  throughput is.

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
