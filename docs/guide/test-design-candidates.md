# Test design: candidate tests

`spicefault.testdesign` helps design the self-test of a board-level analog circuit:
which nodes, stimuli and readings an embedded test uses, so that the diagnostic
coverage that IEC 61508 or ISO 26262 asks for is reached and certified with an
interval. It has three parts:

| Module | What |
|---|---|
| `testdesign.candidates` | The candidate tests and the experiment that simulates them (this page) |
| `testdesign.metrics` | Detection, false alarms and coverage metrics with bounds ([coverage metrics](test-design-metrics.md)); stopping rules in `metrics.adaptive` ([stopping rules](test-design-adaptive.md)) |
| `testdesign.select` | Test selection; published methods in `select.baselines` ([test selection](test-design-selection.md)) |

The example [Test design](../examples/test-design.md) goes through all of them on a
Sallen-Key band-pass.

## A test is node × stimulus × feature

A candidate test is one number the self-test can read: a feature of the voltage at an
accessible node while the circuit is driven by a stimulus. Its `test_id`,
`stimulus|node|feature`, is also the name of its column in the results.

```python
from spicefault import Circuit
from spicefault.testdesign import candidates

circuit = candidates.prepare_source(Circuit.from_netlist("sallen_key.cir"), "V1")
nodes = candidates.accessible_nodes(circuit)        # not ground, not a source's nodes
stimuli = [
    candidates.ac_tones("V1", [10e3, 25e3, 60e3]),  # gain (dB) and phase of each tone
    candidates.step("V1", 1.0, 200e-6, [10e-6, 50e-6]),  # values, final, peak, minimum
    *candidates.dc_levels("V1", [1.0]),             # the node voltage at each level
]
tests = candidates.enumerate_tests(nodes, stimuli)
candidates.listing(tests)                           # one row per test
```

| Stimulus | Analysis | Features |
|---|---|---|
| `ac_tones(source, frequencies)` | One AC sweep, half a decade beyond the outer tones | `gain_db@f` and `phase@f` of unit tones, read by interpolation |
| `step(source, amplitude, duration, times)` | A transient of `duration` | `v@t` at each instant, `final`, `peak` and `minimum` |
| `dc_levels(source, levels)` | One operating point per level | `value` |

A step needs a `pulse(...)` on the source: `prepare_source` adds a flat one, at 0 V, so
that the AC and DC analyses are not changed by it. `accessible_nodes` leaves out
ground and every node that a voltage source fixes, the input and the supply rails;
`exclude` removes more. Each `Feature` builds a `Measurement`, so other features can be
added in the same way.

## One experiment for all the stimuli

`conditions(tests)` gives one `OperatingCondition` per stimulus, with its analysis and
the measurements of its tests. In one `Experiment`, a drawn circuit is observed under
every condition, since the condition is never part of the key of its random stream, so
the tests of different stimuli are read on the same circuit and can be combined.

```python
from spicefault import Experiment

experiment = Experiment(circuit, faults=faults, variations=variations,
                        conditions=candidates.conditions(tests),
                        samples=40, healthy_samples=200, seed=1)
matrix = candidates.matrix(experiment.run(workers=4).to_frame(), tests)
candidates.constant_tests(matrix, tests)   # tests that no fault changes
```

`matrix` has one row per drawn circuit (`fault_id`, `replica`) and one column per
test. A circuit whose simulation failed under any stimulus is dropped, since its tests
cannot be combined; the count of rows per fault says how many remain. Nothing in this
module changes the experiment engine: the provenance, the seeding and the campaign
machinery are those of any experiment.
