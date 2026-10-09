# Faults

A fault is a list of primitive changes to the netlist, with a record of what it is.
There are three primitives, and nothing else changes a netlist:

| Primitive | Effect |
|---|---|
| `SetParameter` | Change a value or an instance parameter: set it, or move it relative to the value the component has |
| `InsertSeries` | Reconnect one terminal to its node through a resistance |
| `InsertParallel` | Add a resistance between the nodes of two terminals |

## Fault types

| Type | Physical interpretation | What it does to the netlist |
|---|---|---|
| `OpenCircuit` | Broken connection | A large resistance in series with one terminal |
| `ShortCircuit` | Bridge or breakdown across a component | A small resistance between two terminals |
| `LeakageFault` | Finite parasitic conduction | A graded resistance between two terminals |
| `SeriesResistanceFault` | Increased resistance in a conducting path: a degraded joint, the series resistance of an ageing capacitor | A graded resistance in series with one terminal |
| `ParametricFault` | A parameter outside its tolerance band | The parameter is moved relative to its realised value, or set to a value |
| `CompositeFault` | One mechanism with several electrical consequences | The changes of its parts, in order |

Every fault has an identifier, a record (`fault.metadata()`) from which it can be
rebuilt, and optionally a severity on a declared scale. A `FaultSet` is the ordered
fault list of an experiment; it can be saved to JSON and filtered.

```python
from spicefault.faults import (
    CompositeFault, LeakageFault, OpenCircuit, ParametricFault, ShortCircuit,
)

OpenCircuit("R1")                              # 1 GOhm in series with its second terminal
ShortCircuit("C1", r_short=0.1)                # 0.1 Ohm across it
LeakageFault("C1", 1e6, r_min=1e4, r_max=1e8)  # graded, severity on a log scale
ParametricFault("R2", deviation=-0.2)          # 20 % below the value it was drawn with
ParametricFault("XU1", "vos", fault_value=0.02, nominal_value=0.0, reference=0.5e-3)
CompositeFault("capacitor_ageing", [
    ParametricFault("C1", factor=0.7),         # less capacitance ...
    OpenCircuit("C1", r_open=50.0),            # ... and more series resistance
])
```

Opens and shorts are finite resistances: an ideal open leaves a node floating and the
operating point has no solution. `r_open` and `r_short` belong to the model, are
recorded with the fault, and should be checked against the impedances around the
component.

## Magnitude and severity

The *magnitude* of a fault is its physical size, with a unit: a deviation, a
resistance, an offset. Its *severity* is an optional dimensionless number on a scale
that is recorded with it. There is no universal formula, and severities on different
scales cannot be compared.

## Fault universe and coverage

The faults of a campaign can be generated from rules instead of listed by hand, so
that what was simulated and what was left out are both on record:

```python
from spicefault.faults import FaultUniverse, open_rule, parametric_rule, short_rule

universe = FaultUniverse(
    circuit,
    [open_rule("R"), short_rule("R"), parametric_rule([-0.2, -0.05, 0.05, 0.2], "R")],
)
universe.exclude("R2:short", reason="ties the output to ground: outside the study")

universe.coverage_matrix()   # components x fault types: conditions to simulate
universe.exclusions()        # every fault left out, with its reason
universe.coverage()          # 11 of 12 conditions

faults = universe.selected()
faults.tolerance_overlap(experiment.variations)   # parametric faults inside the tolerance band
```

The rules are `open_rule`, `short_rule`, `parametric_rule`, `leakage_rule` and
`series_rule`; a `FaultRule` with a function of its own covers what they do not.

A rule selects components by element letter (`kinds`), by name (`components`) and by
the subcircuit of an instance or the model of a device (`models`, compared with
`Component.model`). It can tag its faults, with a dictionary or a function of the
component, and a parametric rule takes relative deviations, factors or absolute
values, reported under the fault type given:

```python
rules = [
    parametric_rule(kinds="X", models=("opamp",), parameter="vos", fault_values=(20e-3,),
                    fault_type="offset", tags=lambda c: {"part": c.name, "origin": "amplifier"}),
    parametric_rule(kinds="X", models=("opamp",), parameter="aol", factors=(0.01,),
                    fault_type="gain_loss"),
    open_rule("R", tags={"origin": "assembly"}),
]
```

With `tag_columns=("part", "origin")`, a campaign writes these tags as columns of the
dataset.

The definitions behind this page are in the [fault model](../FAULT_MODEL.md).
