# spicefault

`spicefault` is a Python framework for reproducible SPICE-based fault injection and
reliability assessment of electronic circuits under component uncertainty and
operating-condition variability.

It is not another Python interface to SPICE. It works one level above: a fault, the
normal variation of the components and the operating conditions are defined
explicitly, simulated as one campaign with ngspice, and recorded so that every
sample can be traced and regenerated.

- Project specification: [spicefault-develoment.md](spicefault-develoment.md)
- Scientific scope and research questions: [docs/SCIENTIFIC_SCOPE.md](docs/SCIENTIFIC_SCOPE.md)
- Fault model: [docs/FAULT_MODEL.md](docs/FAULT_MODEL.md)
- Reliability metrics: [docs/RELIABILITY_METRICS.md](docs/RELIABILITY_METRICS.md)
- Experiment plan: [docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md)

## Status

| Phase | State |
|---|---|
| 0. Scientific specification | Drafted, open decisions listed at the end of each document |
| 1. Extraction from the ECG project | Done: the generic parts of `ecgfd` live here and are checked against it |
| 2. Core abstractions (`Circuit`, `Fault`, `Experiment`, ...) | Not started |

What exists today is the layer extracted from the ECG front-end study
([ecg-frontend-fault-diagnosis](https://github.com/telmomm/ecg-frontend-fault-diagnosis)),
without the high-level objects of the specification:

| Module | Content |
|---|---|
| `spicefault.simulation` | ngspice batch runner and reader of binary raw files |
| `spicefault.netlist` | Netlist as text and the three fault-injection primitives: `set_parameter`, `insert_series`, `insert_parallel` |
| `spicefault.variation` | Tolerance draws: uniform, truncated normal, log-uniform |
| `spicefault.experiments` | Per-sample random streams; parallel, chunked and resumable execution of a campaign |
| `spicefault.measurements` | Interpolation of frequency responses, resampling of transients, ADC quantisation |
| `spicefault.dataset` | Dataset directory: `samples.parquet`, `waveforms.npy`, `manifest.json` |

## Setup

Requires Python ≥ 3.10 and [ngspice](https://ngspice.sourceforge.io/) on the `PATH`
(developed with ngspice 44).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Tests that need ngspice are skipped if it is missing.

## Equivalence with the ECG baseline

`tests/regression/` compares this package with `ecgfd`, the code it was extracted
from. These tests run only if `ecgfd` is installed:

```bash
pip install -e ../ecg-frontend-fault-diagnosis
pytest tests/regression
```

| Test | What is compared | Criterion |
|---|---|---|
| `test_ecg_runner.py` | Output of the same decks through both ngspice runners | Identical vectors |
| `test_ecg_sampling.py` | Random streams and tolerance draws | Identical numbers |
| `test_ecg_injection.py` | Netlist of every fault condition of both ECG circuits, built by `ecgfd` and by the primitives | Identical text |
| `test_ecg_campaign.py` | A reduced campaign run by both engines, with 1 and with several workers | Identical tables and waveforms |
| `test_ecg_baseline_data.py` | Cases regenerated here against the published dataset `data/v1` | Tolerances of docs/EXPERIMENT_PLAN.md §3 |

The last one needs the dataset; its location is taken from the `ECGFD_DATA`
environment variable and defaults to `data/v1` in the ECG repository.

## Example

```python
from spicefault.netlist import Netlist
from spicefault.simulation import run_deck

deck = Netlist("""divider
V1 in 0 dc 1
R1 in out 10k
R2 out 0 10k
.control
op
write out.raw v(out)
.endc
.end
""")
deck.insert_series("R2", 2, 1e9)          # R2 open
print(run_deck(str(deck))[0]["v(out)"])   # close to 1 V instead of 0.5 V
```
