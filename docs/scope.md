# Scope of the library

What belongs in `spicefault` and what is left to other tools. A feature request or a
pull request can be checked against this page.

This is the scope of the software. The scope of the manuscript, which is narrower, is in
[Scientific scope](SCIENTIFIC_SCOPE.md) and is not changed by it.

## The criterion

Something belongs in the library if either holds:

1. **It needs the experiment engine and its provenance to be done right.** Drawing or
   designing the samples, building each netlist, simulating in parallel and resumably,
   keeping the failed simulations, and recording what produced every row.
2. **It defines a quantity that every study would otherwise implement differently.** A
   metric with a written definition, an estimator, an interval and a test against a
   population with a known answer.

What pandas, SciPy, SALib or scikit-learn do on the exported table stays outside, and
is documented here as a recipe. The library has no dependency on any of them.

## Inside

| Area | What |
|---|---|
| Fault injection | Fault types as netlist primitives, the fault universe and its coverage |
| Healthy population | Variations and tolerances, also correlated and by manufacturing lot |
| Experiments | Drawn samples, designed samples and tolerance corners, operating conditions, the nominal circuit, a given parameter vector |
| Sampling | Independent streams per sample; Latin hypercube and Sobol designs |
| Execution | Parallel, resumable campaigns in which a failure is a row with a status |
| Datasets | Provenance, verification, reproduction, specifications and pass/fail labels |
| Observation | The instrument model: noise, resolution and range applied after the simulation |
| Statistics | Statistics, quantiles, yield and margins of a fault-free population |
| Reliability | Detectability, failure probability, diagnostic coverage, separability and ambiguity |

Some of these are planned and not yet released: the
[roadmap](https://github.com/telmomm/spicefault/issues/33) tracks them.

## Outside, with a recipe

### Screening which parameters matter

The realised value of every varied parameter is a column of the dataset, so a rank
correlation with a measurement is one line. It is a screening measure: it ranks
monotonic effects and says nothing about interactions.

```python
samples = dataset.samples[dataset.ok]
samples[list(dataset.parameters)].corrwith(samples["gain_db"], method="spearman")
```

Variance-based global sensitivity indices need their own sample matrix. The library
simulates the points that [SALib](https://salib.readthedocs.io) asks for, with
`Experiment.evaluate`:

```python
--8<-- "examples/recipes/sensitivity_indices.py"
```

### Searching for the worst case

Tolerance corners are inside the library. A search by optimisation is not: it is
`scipy.optimize` around `Experiment.evaluate`, and its result is the worst case *found*,
which may be a local extreme. Neither a search nor a finite random sample proves a worst
case.

```python
--8<-- "examples/recipes/worst_case.py"
```

Both recipes are in `examples/recipes/` and run with the tests; their packages come
with `pip install "spicefault[recipes]"`.

### Confusion matrices and classifiers

`Dataset.to_ml` returns NumPy arrays and the splits keep the replicas of a circuit
together, so the evaluation of a classifier is scikit-learn's:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

X, y = dataset.to_ml(target="fault_location")
train, test = split_by_replica(dataset.samples[dataset.ok], test_fraction=0.3, seed=0)
model = RandomForestClassifier(random_state=0).fit(X[train], y[train])
confusion_matrix(y[test], model.predict(X[test]))
```

The detection, escape and false-reject rates of a decision rule, with their intervals,
are part of `ReliabilityAnalysis`.

### Temperature coefficients

SPICE models them in the netlist, and an `OperatingCondition` sets the temperature:

```text
R1 in out 10k tc1=100u tc2=0
```

```python
OperatingCondition("hot", temperature=85)
```

## Outside

- **Adaptive sampling and rare-event estimation.** Research methods whose samples
  depend on earlier results, which conflicts with a plan that is fixed, resumable and
  the same for any number of workers.
- **Ageing and time-dependent reliability.** They need failure models and field data
  that the library cannot validate. A drift of known size is already a parametric fault
  or a variation.
- **A graphical interface.** It could be a separate application on the public
  interface.
- **Other simulators**, until a study needs one: the backend is an interface.
- **Training of models.** The library produces the dataset.
