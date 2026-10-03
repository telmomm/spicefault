# Normal variation

Variations describe the healthy population: how the parameters of good circuits
differ. They are not faults. A fault is injected into a circuit already drawn from
this population.

| Variation | Value drawn |
|---|---|
| `ToleranceVariation` | Within nominal × (1 ± tolerance), or nominal ± tolerance with `relative=False`; uniform or truncated normal |
| `NormalVariation` | Normal around a mean (the nominal value by default), optionally truncated |
| `LogNormalVariation` | Median × exp(N(0, σ)): positive, multiplicative spread |
| `UniformVariation` | Uniform between two limits |
| `LogUniformVariation` | Median times a factor between 1/spread and spread |
| `FixedVariation` | A set value, no spread |
| `CustomVariation` | Any function of the random stream and the nominal value |
| `JointVariation` | Several parameters drawn together, when they depend on each other |

```python
from spicefault.variation import NormalVariation, VariationSet, tolerances

population = tolerances(circuit, {"R": 0.01, "C": 0.05})   # by component kind
population = VariationSet([NormalVariation("R1", 10000, 500), NormalVariation("R2", 10000, 500)])
tighter = population.scaled(0.2)   # same random numbers, a fifth of the spread
```

A normal variation is not a fault: variations describe the healthy population, and a
fault is injected into a circuit already drawn from it. `scaled` multiplies every
spread while using the same random numbers, so a study of detectability against
tolerance compares the same circuits at each tolerance level.

## Parameters that are not independent

`JointVariation` draws several parameters together: the parameters of a part whose
type is drawn first, two resistors from the same reel, or a netlist parameter computed
from two drawn quantities. It can also record labels, such as the type that was drawn.
See the [custom example](../examples/custom.md).

Functions given to `CustomVariation` and `JointVariation` must be defined at module
level, so that worker processes can import them. A record cannot store a function: a
dataset keeps the name of the function, and the experiment has to be passed again to
simulate its samples anew.
