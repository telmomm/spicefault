# Getting started

## Requirements

- Python 3.10 or later.
- [ngspice](https://ngspice.sourceforge.io/) on the `PATH`. The library is developed
  with ngspice 44. Check with `ngspice --version`.

```bash
pip install spicefault
```

To work on the library itself:

```bash
git clone https://github.com/telmomm/spicefault.git
cd spicefault
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,docs]"
pytest tests/unit
```

## 1. A circuit

A circuit is a SPICE netlist, from a file or from text. `spicefault` reads its
components, nodes and numeric parameters and never modifies it: every simulation works
on a copy.

```python
from spicefault import Circuit

circuit = Circuit.from_netlist("rc_lowpass.cir")
circuit.components()     # name, kind, nodes and parameters of each
circuit.parameters()     # {("R1", "value"): 10000.0, ...}
```

The netlist needs no analysis commands: the experiment adds them.

## 2. Faults, and the healthy population

```python
from spicefault.faults import OpenCircuit, ParametricFault
from spicefault.variation import tolerances

faults = [OpenCircuit("R2"), ParametricFault("C1", deviation=0.5)]
population = tolerances(circuit, {"R": 0.01, "C": 0.05})   # 1 % and 5 % parts
```

## 3. What to simulate and what to read

```python
from spicefault import Measurement, SimulationConfig

config = SimulationConfig(("op", "ac dec 20 1 1e5"), outputs=("v(out)",))
measurements = [
    Measurement.value("v(out)", name="dc"),
    Measurement.magnitude("v(out)", 1e3, name="gain_1k"),
]
```

## 4. Run it

In memory, for a small study:

```python
from spicefault import Experiment

experiment = Experiment(circuit, faults=faults, variations=population, config=config,
                        measurements=measurements, samples=50, seed=1)
table = experiment.run(workers=4).to_frame()
```

To disk, for a campaign that takes long and must survive an interruption:

```python
from spicefault import FaultCampaign

campaign = FaultCampaign(circuit, faults, out_dir="data/rc", samples_per_fault=200,
                         healthy_samples=2000, variations=population, config=config,
                         measurements=measurements, seed=1)

if __name__ == "__main__":
    print(campaign.validate())
    campaign.run(workers=8)
```

!!! note "Scripts that use several workers"
    Workers are processes. On macOS and Windows the part of a script that launches
    them must be under `if __name__ == "__main__":`, and functions given to custom
    variations or measurements must be defined at module level.

## 5. Analyse

```python
from spicefault.reliability import ReliabilityAnalysis

analysis = ReliabilityAnalysis.from_dataset("data/rc")
analysis.detectability(alpha=0.01).table
```

The [examples](examples/index.md) do all of this on a real circuit and can be run as
they are.
