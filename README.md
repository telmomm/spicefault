# spicefault

<p align="center">
  <img src="https://raw.githubusercontent.com/telmomm/spicefault/main/docs/assets/branding/github-banner.png" alt="spicefault: reproducible fault injection for electronic circuits" width="100%">
</p>

[![PyPI](https://img.shields.io/pypi/v/spicefault)](https://pypi.org/project/spicefault/)
[![Python](https://img.shields.io/pypi/pyversions/spicefault)](https://pypi.org/project/spicefault/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/telmomm/spicefault/blob/main/LICENSE)
[![CI](https://github.com/telmomm/spicefault/actions/workflows/ci.yml/badge.svg)](https://github.com/telmomm/spicefault/actions/workflows/ci.yml)
[![Documentation](https://readthedocs.org/projects/spicefault/badge/?version=latest)](https://spicefault.readthedocs.io/en/latest/)
[![Quality gate](https://sonarcloud.io/api/project_badges/measure?project=telmomm_spicefault&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=telmomm_spicefault)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=telmomm_spicefault&metric=coverage)](https://sonarcloud.io/summary/new_code?id=telmomm_spicefault)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/telmomm/spicefault/main?labpath=binder%2Fnotebooks%2F01_quickstart.ipynb)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
<!-- DOI: after the first release archived by Zenodo, add its badge here:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX) -->

**Reproducible SPICE-based fault injection and reliability assessment of electronic
circuits.**

`spicefault` is a Python framework for studying how a circuit behaves when something
in it fails, under realistic component tolerances and operating conditions. You
describe a circuit, the faults it can have and how its good units vary; the framework
simulates the campaign with [ngspice](https://ngspice.sourceforge.io/), keeps a record
of every sample, and computes what the results say: which faults are detected, which
are confused with each other, and how tolerance erodes both.

It is not another Python interface to SPICE. It works one level above: the fault, the
normal variation of the components and the operating conditions are defined
explicitly, simulated as one experiment, and recorded so that every sample can be
traced and simulated again.

**[Documentation](https://spicefault.readthedocs.io)** ·
**[Examples](https://github.com/telmomm/spicefault/tree/main/examples/)** ·
**[Changelog](https://github.com/telmomm/spicefault/blob/main/CHANGELOG.md)**

## Install

```bash
pip install spicefault
```

and ngspice, which does the simulations (`brew install ngspice` on macOS,
`sudo apt install ngspice` on Debian and Ubuntu). Python 3.10 or later.

## A first experiment

A resistive divider with 1 % resistors, two faults and two operating conditions:

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
    conditions=[
        OperatingCondition(),
        OperatingCondition("low supply", settings={("V1", "dc"): 0.5}),
    ],
    samples=100,
    seed=42,
)
result = experiment.run(workers=4)

print(result.status_counts())      # {'SUCCESS': 600}
table = result.to_frame()          # sample definitions, status and realised values
vout = [s.result.plot("op")["v(out)"][0] for s in result]
```

- The component values are drawn first, the fault is injected into that drawn
  circuit, and then the operating condition is set.
- Each sample has its own random stream: the result is the same on one worker or on
  sixteen.
- A simulation that fails stays in the results, with a status and a message.

## What it gives you

| | |
|---|---|
| **Faults as objects** | Open, short, leakage, series-resistance, parametric and composite faults, each a recorded change of the netlist with a physical interpretation |
| **An auditable fault list** | A fault universe generated from rules, a coverage matrix, and a reason for every fault left out |
| **A healthy population** | Tolerance, normal, log-normal, uniform, correlated and lot-level distributions, kept apart from the faults |
| **Statistics without faults** | The same campaign as a Monte Carlo study: quantiles and yield with their intervals, tolerance corners, Latin hypercube and Sobol sampling, and the effect of the instrument that reads the measurements |
| **Reproducible campaigns** | To disk in chunks, resumable; the same samples for any number of workers; five explicit simulation statuses |
| **Datasets that stand on their own** | Integrity check, provenance of each sample, the netlist that was simulated for it, and simulation again from the folder |
| **Reliability metrics with intervals** | Detection probability, diagnostic coverage, minimum detectable deviation, robustness against tolerance, ambiguity groups |

```python
from spicefault import FaultCampaign, Dataset
from spicefault.reliability import ReliabilityAnalysis

campaign = FaultCampaign(circuit, faults, out_dir="data/study", samples_per_fault=200,
                         healthy_samples=2000, variations=population, config=config,
                         measurements=measurements, seed=1)
campaign.run(workers=8)                       # resumes if it was interrupted

dataset = Dataset("data/study")
dataset.verify()                              # the files match their fingerprints
dataset.provenance(1234)                      # everything about one sample
dataset.reproduce(n=50)                       # simulate again, compare with what is stored

analysis = ReliabilityAnalysis.from_dataset("data/study")
analysis.detectability(alpha=0.01).table      # per fault: P(detect) and its interval
analysis.minimum_detectable()                 # smallest deviation that is visible
analysis.ambiguity()                          # faults that cannot be told apart
```

## Examples

Runnable scripts in [examples/](https://github.com/telmomm/spicefault/tree/main/examples/), the same ones shown in the documentation:

| Script | What it shows |
|---|---|
| `01_quickstart.py` | A circuit, a fault, a simulation, a small experiment |
| `02_faults_and_coverage.py` | Fault types, a fault universe from rules, coverage |
| `03_campaign.py` | A campaign to disk, interrupted and resumed |
| `04_dataset.py` | Integrity, provenance and reproduction of a dataset |
| `05_reliability.py` | Detection, minimum detectable deviation, ambiguity, tolerance |
| `06_operating_conditions.py` | The same circuits under several conditions |
| `07_custom.py` | Your own distributions and measurements |

Open the executable [Binder notebooks](https://github.com/telmomm/spicefault/tree/main/binder/notebooks)
for the same examples, paired to the `.py` sources that the documentation includes, or launch one directly:

| Example | Binder notebook |
|---|---|
| 01 Quickstart | [Open](https://mybinder.org/v2/gh/telmomm/spicefault/main?labpath=binder%2Fnotebooks%2F01_quickstart.ipynb) |
| 02 Faults and coverage | [Open](https://mybinder.org/v2/gh/telmomm/spicefault/main?labpath=binder%2Fnotebooks%2F02_faults_and_coverage.ipynb) |
| 03 Fault campaign | [Open](https://mybinder.org/v2/gh/telmomm/spicefault/main?labpath=binder%2Fnotebooks%2F03_campaign.ipynb) |
| 04 Dataset | [Open](https://mybinder.org/v2/gh/telmomm/spicefault/main?labpath=binder%2Fnotebooks%2F04_dataset.ipynb) |
| 05 Reliability | [Open](https://mybinder.org/v2/gh/telmomm/spicefault/main?labpath=binder%2Fnotebooks%2F05_reliability.ipynb) |
| 06 Operating conditions | [Open](https://mybinder.org/v2/gh/telmomm/spicefault/main?labpath=binder%2Fnotebooks%2F06_operating_conditions.ipynb) |
| 07 Custom variation | [Open](https://mybinder.org/v2/gh/telmomm/spicefault/main?labpath=binder%2Fnotebooks%2F07_custom.ipynb) |

```bash
git clone https://github.com/telmomm/spicefault.git && cd spicefault
pip install -e .
python examples/01_quickstart.py
```

Or try the quickstart in the browser, without installing anything, with the Binder
badge above.

## Limits

- One simulator: ngspice.
- One fault per sample; faults do not change during a simulation.
- Faults act on the top level of the netlist. A component inside a subcircuit is
  reached through the parameters of its instance.
- Opens and shorts are finite resistances, recorded with the fault.
- Simulated evidence does not replace testing hardware.

## Project

The library is the instrument of a research project on reproducible fault-injection
experiments. Its definitions and plan are part of the documentation:
[fault model](https://github.com/telmomm/spicefault/blob/main/docs/FAULT_MODEL.md),
[reliability metrics](https://github.com/telmomm/spicefault/blob/main/docs/RELIABILITY_METRICS.md),
[scientific scope](https://github.com/telmomm/spicefault/blob/main/docs/SCIENTIFIC_SCOPE.md),
[experiment plan](https://github.com/telmomm/spicefault/blob/main/docs/EXPERIMENT_PLAN.md),
[related work](https://github.com/telmomm/spicefault/blob/main/docs/RELATED_WORK.md) and
[comparison with other tools](https://github.com/telmomm/spicefault/blob/main/docs/COMPARISON.md).
The circuits it is validated on are in [validation/](https://github.com/telmomm/spicefault/tree/main/validation/), the benchmarks in
[benchmarks/](https://github.com/telmomm/spicefault/blob/main/benchmarks/README.md), and the results obtained so far in `results/`.

It was extracted from the simulation code of a study on self-diagnosis of ECG analog
front-ends
([ecg-frontend-fault-diagnosis](https://github.com/telmomm/ecg-frontend-fault-diagnosis)),
which uses it. Nothing in it is specific to that study.

## Contributing

Bug reports and contributions are welcome: see [CONTRIBUTING.md](https://github.com/telmomm/spicefault/blob/main/CONTRIBUTING.md) and
the [code of conduct](https://github.com/telmomm/spicefault/blob/main/CODE_OF_CONDUCT.md).

## Citing

If you use `spicefault` in your work, please cite it. The citation data are in
[CITATION.cff](https://github.com/telmomm/spicefault/blob/main/CITATION.cff); GitHub shows them under "Cite this repository".

## Licence

[MIT](https://github.com/telmomm/spicefault/blob/main/LICENSE).
