# spicefault

**Reproducible SPICE-based fault injection and reliability assessment of electronic
circuits.**

`spicefault` is a Python framework for studying how a circuit behaves when something
in it fails, under realistic component tolerances and operating conditions. You
describe a circuit, the faults it can have and how its good units vary; the framework
simulates the campaign with [ngspice](https://ngspice.sourceforge.io/), keeps a record
of every sample, and computes what the results say: which faults are detected, which
are confused with each other, and how tolerance erodes both.

It is not another Python interface to SPICE. It works one level above:

```text
   circuit  +  normal variation  +  faults  +  operating conditions
                              |
                         experiment            one seed, one definition
                              |
                    SPICE simulations          every one with a status
                              |
        dataset: samples, waveforms, provenance
                              |
   detectability, diagnostic coverage, robustness, ambiguity
```

## Install

```bash
pip install spicefault
```

and ngspice, which does the simulations:

```bash
brew install ngspice          # macOS
sudo apt install ngspice      # Debian, Ubuntu
```

## A first experiment

A resistive divider with 1 % resistors, two faults and two operating conditions:

```python
--8<-- "docs/snippets/first_experiment.py"
```

Go on with [Getting started](getting-started.md), or run the
[examples](examples/index.md).

## What it gives you

- **Faults as objects.** Open, short, leakage, parametric and composite faults, each a
  recorded change of the netlist with a physical interpretation.
  [Faults](guide/faults.md)
- **An auditable fault list.** A fault universe generated from rules, with a coverage
  matrix and a reason for every fault left out.
- **A healthy population.** Tolerance, normal, log-normal and joint distributions,
  kept apart from the faults. [Normal variation](guide/variation.md)
- **Reproducible campaigns.** The same samples on any number of workers; resumable;
  every failed simulation kept with its status. [Experiments](guide/experiments.md)
- **Datasets that stand on their own.** Integrity checks, the provenance of each
  sample, and the netlist that was simulated for it. [Datasets](guide/datasets.md)
- **Reliability metrics with intervals.** Detection probability, diagnostic coverage,
  minimum detectable deviation, robustness against tolerance, ambiguity groups.
  [Reliability analysis](guide/reliability.md)

## Limits

One simulator, ngspice. One fault per sample. Faults act on the top level of the
netlist: a component inside a subcircuit is reached through the parameters of its
instance. Simulated evidence does not replace testing hardware.

## Citing

If you use `spicefault` in your work, please cite it: see `CITATION.cff` in the
repository.
