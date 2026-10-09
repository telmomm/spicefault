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
| `CatalogueVariation` | The type of a part, then its parameters around the medians of that type |
| `CorrelatedVariation` | Several variations that keep their distributions and vary together |
| `LotVariation` | Components that share the deviation of their manufacturing lot |

```python
from spicefault.variation import NormalVariation, VariationSet, tolerances

population = tolerances(circuit, {"R": 0.01, "C": 0.05})   # by component kind
population = VariationSet([NormalVariation("R1", 10000, 500), NormalVariation("R2", 10000, 500)])
tighter = population.scaled(0.2)   # same random numbers, a fifth of the spread
```

`instance_tolerances` gives every instance of a subcircuit the same spread in its
parameters, relative or, for a parameter that is nominally zero, absolute:

```python
from spicefault.variation import instance_tolerances

amplifiers = instance_tolerances(circuit, "opamp", {"vos": (0.5e-3, "absolute"), "aol": 0.5})
```

A normal variation is not a fault: variations describe the healthy population, and a
fault is injected into a circuit already drawn from it. `scaled` multiplies every
spread while using the same random numbers, so a study of detectability against
tolerance compares the same circuits at each tolerance level.

Every variation except the custom, joint and catalogue ones also has a quantile
function, `variation.quantile(u, nominal)`: the value at probability `u`. Sampling does
not use it; the designs that choose their uniform numbers jointly do.

## Parameters that are not independent

Independent tolerances overstate the spread of a ratio and understate that of a sum.
Two variations describe the common cases without a function, so a dataset rebuilds them
from its record:

```python
from spicefault.variation import CorrelatedVariation, LotVariation, ToleranceVariation

# two resistors from the same reel: each within 1 %, tracking each other
CorrelatedVariation([ToleranceVariation("R1", 0.01), ToleranceVariation("R2", 0.01)],
                    correlation=0.9)

# a whole group shifted by its lot (4 %), each part 1 % around it
LotVariation(("R1", "R2", "R3"), lot=0.04, within=0.01)
```

`CorrelatedVariation` keeps the distribution of each parameter and joins them with a
Gaussian copula. `correlation`, one number or the full matrix, is that of the
underlying normals: the correlation of the values equals it only for normal variations,
and their rank correlation is (6/π) asin(correlation/2) for any. Each variation it joins
needs a quantile function. `LotVariation` draws one deviation for the group and one of
each component, and records the first as a label.

`JointVariation` draws several parameters together: the parameters of a part whose
type is drawn first, two resistors from the same reel, or a netlist parameter computed
from two drawn quantities. It can also record labels, such as the type that was drawn.
See the [custom example](../examples/custom.md).

When the dependence is a part that comes in several types, `CatalogueVariation` needs
no function: the type is drawn first, with the given weights, and each parameter is
the median of that type times a factor between 1/spread and spread. The type is
recorded as a label, and the variation is rebuilt from the record of a dataset.

```python
from spicefault.variation import CatalogueVariation

CatalogueVariation(
    "Xelectrode",
    options={"gel": {"r": 2e3, "c": 50e-9}, "steel": {"r": 2e5, "c": 5e-9}},
    weights=(3, 1), spread=2.0, label="electrode_kind",
)
```

Functions given to `CustomVariation` and `JointVariation` must be defined at module
level, so that worker processes can import them. A record cannot store a function: a
dataset keeps the name of the function, and the experiment has to be passed again to
simulate its samples anew.
