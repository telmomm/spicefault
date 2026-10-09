# Runbook: the long runs

Everything that takes long, in the order to launch it. Run on an idle machine on mains power: the timing results are invalid otherwise, and the campaigns take longer.

All commands are run from the repository root with the project environment. The command blocks have no comments, so that they can be pasted into an interactive shell. Launch each block once, in one terminal: a campaign that is already running refuses a second launch. Every campaign resumes if it is interrupted: launch the same command again.

```bash
cd /path/to/spicefault
source .venv/bin/activate
```

The durations are estimates from a few simulations on a loaded machine, not measurements: about 30 ms per simulation of the filters and 20 ms of the regulator, on 8 workers. Expect them to be off by a factor of two.

## 0. The test suite (about 6 minutes)

```bash
python -m pytest -q
```

It has not been run in full since the validation circuits, the experiment scripts and the benchmarks changed; its parts were run separately. Run it before anything else.

## 1. Campaigns of experiment E: tolerance (about 1 h in total)

Four tolerance scales per circuit, 5000 healthy samples and 200 per fault at each.

```bash
python -m validation.experiments.e_variability run --circuit sallen_key --workers 8
python -m validation.experiments.e_variability run --circuit biquad     --workers 8
python -m validation.experiments.e_variability run --circuit regulator  --workers 8
```

Datasets go to `data/validation/<circuit>/scale_<k>/`.

## 2. Campaigns of experiment F: operating conditions (about 30 minutes)

The regulator under seven conditions, then under nine.

```bash
python -m validation.experiments.f_conditions run --conditions one_factor --workers 8
python -m validation.experiments.f_conditions run --conditions corners    --workers 8
```

## 3. Analyses (minutes, no campaign)

```bash
python -m validation.experiments.e_variability analyse --circuit sallen_key
python -m validation.experiments.e_variability analyse --circuit biquad
python -m validation.experiments.e_variability analyse --circuit regulator
python -m validation.experiments.f_conditions analyse --conditions one_factor
python -m validation.experiments.f_conditions analyse --conditions corners
python -m validation.experiments.g_separability --circuit biquad
python -m validation.experiments.g_separability --circuit sallen_key
```

Each prints a summary and writes JSON and CSV under `results/`.

## 4. Benchmarks (about 1 h 30 min; the machine must be idle)

```bash
python -m benchmarks.correctness.run
python -m benchmarks.scalability.run --workload sallen_key --workers 1 2 4 8
python -m benchmarks.scalability.run --workload biquad --workers 1 2 4 8
python -m benchmarks.reproducibility.run --workload biquad --workers 8
python -m benchmarks.reproducibility.run --workload regulator --workers 8
python -m benchmarks.fault_coverage.run
```

Results go to `benchmarks/results/`. The machine has 4 performance and 4 efficiency cores, so efficiency above 4 workers falls for hardware reasons.

## 5. A quick pass first (about 5 minutes)

To check the whole chain before the long runs, with sizes too small to mean anything:

```bash
python -m validation.experiments.e_variability run --circuit sallen_key --healthy 400 --samples-per-fault 10 --data data/try
python -m validation.experiments.e_variability analyse --circuit sallen_key --data data/try
python -m validation.experiments.g_separability --circuit sallen_key --data data/try
python -m benchmarks.scalability.run --workload sallen_key --workers 2 4 --repetitions 2 --samples 400 --label try
```

`data/` and `results/` of a quick pass can be deleted afterwards.

## What to bring back

The summaries printed by the analyses, or the files under `results/` and `benchmarks/results/`. The datasets under `data/` are large and stay out of the repository.
