# Concepts

`spicefault` treats a fault study as one explicit experiment. Four things define it,
and each is an object:

| Object | What it is |
|---|---|
| `Circuit` | The nominal circuit: a SPICE netlist. It is never modified |
| `VariationSet` | The healthy population: how each parameter varies between good circuits |
| `Fault` | A defect, as a recorded change of the netlist |
| `OperatingCondition` | Supply, load, temperature: what the circuit is used under |

An `Experiment` puts them together with a simulator and a seed.

## How a sample is built

1. Its random stream depends only on (seed, fault, replica), so the result does not
   depend on the number of workers.
2. The component values are drawn; then the fault is injected into that realised
   circuit; then the operating condition is set.
3. One drawn circuit is simulated under every operating condition.
4. A simulation that fails is kept in the results with a status (`TIMEOUT`,
   `CONVERGENCE_ERROR`, `INVALID_OUTPUT`, `FAILED`) and does not stop the run.

`experiment.metadata()` returns everything needed to regenerate the samples: circuit
hash, fault records, variations, conditions, seed, simulator and package versions.

Three consequences of this order matter for the results:

- **A fault is injected into a circuit that already has its tolerances.** A resistor
  that drifts by 20 % keeps its own manufacturing deviation, and every other component
  keeps its spread. A fault is never simulated on the nominal circuit alone.
- **A fault is not a failure.** The fault says what was injected. Whether the circuit
  still meets its specification is a result, read from the measurements.
- **A deviation inside the tolerance band is not detectable by any method.** A 5 %
  fault on a 5 % capacitor is, half of the time, a healthy capacitor.
  `FaultSet.tolerance_overlap` reports this for a fault list.

## A simulation always has a status

| Status | Meaning |
|---|---|
| `SUCCESS` | Every analysis asked for gave a plot, with finite values |
| `CONVERGENCE_ERROR` | The solver gave up: singular matrix, time step too small, no convergence |
| `TIMEOUT` | The simulator did not finish in the time allowed |
| `INVALID_OUTPUT` | The simulator ran, but an analysis is missing or a measurement could not be taken |
| `FAILED` | Anything else: no output, the simulator is not installed, a fault that does not fit the circuit |

A failed simulation is a row of the results, with its status and a message. It never
stops a run and is never dropped. Faults that prevent convergence are usually the
severe ones, so dropping them would bias every figure computed afterwards.

## Seeds

Every sample has its own random stream, derived from the seed of the experiment and a
key that identifies the sample. Nothing else feeds the draws. The same experiment
gives the same samples on one worker or on sixteen, in any order, in one run or
resumed after an interruption. See [Experiments](experiments.md#seeding).

## Operating conditions name what they set

An operating condition sets elements of the netlist by name:
`OperatingCondition("low supply", settings={("Vcc", "dc"): 3.0})`. A keyword such as
`supply_voltage` could not be mapped to a netlist without knowing the circuit.
Temperature is the simulation temperature; it only has an effect on devices whose
model depends on it.
