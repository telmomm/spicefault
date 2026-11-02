# Test design: cost model

A self-test runs on the board's own microcontroller, so its tests are read by that
microcontroller's ADC, not by the simulator. In CERTA, read through the ADC of an
STM32G474, tests at virtual-ground nodes detected 15 to 25 points fewer faults than in
simulation. `testdesign.costmodel` applies the instrument before any selection, says
which tests the board can run, and what a plan adds to it in euros and test time.

## The board

```python
from dataclasses import replace
from spicefault.testdesign import costmodel

board = replace(costmodel.STM32G474, channels=4, opamps=0)
```

A `Microcontroller` has an `ADC` (bits, reference, clock, SINAD, maximum offset and
gain errors after calibration, and the largest source impedance that settles at each
sampling time), a `DAC` (range, update rate, settling time), and the resources the
application leaves free for the self-test: ADC channels, internal op-amps that can
buffer a node, and DAC outputs. `STM32G474` is an example instance, from its datasheet
(DS12288 Rev. 2); the free resources are parameters of the board, changed with
`dataclasses.replace`.

`Acquisition` says how the firmware acquires each kind of feature: tones with a number
of DAC points per period and ADC samples per tone, read with a second channel on the
stimulus so that gain and phase are ratios; DC levels and step instants averaged over
a number of readings.

## What the board can run, and how each node is read

```python
costmodel.feasible(test, board)        # "" or why not: "tone above 100000 Hz"
access = costmodel.access(circuit, nodes, board, frequencies=[1.0, 10e3, 25e3])
access["a"].impedance, access["a"].buffer
```

`feasible` refuses tones the DAC or the ADC cannot follow and step instants earlier
than twice the DAC settling time. `access` simulates the source impedance of each node
(`node_impedance`, a 1 A AC probe with every source at zero AC) and asks for a buffer
when it exceeds what a slow ADC channel settles at the fast sampling time.

## What the ADC reads

```python
amplitude = costmodel.amplitudes(matrix, tests, board)    # per stimulus, in volts
read = costmodel.observe(matrix, tests, board, amplitude, seed=1)
```

`observe` returns the test matrix of `candidates.matrix` as the ADC reads it, in the
units of the simulation. The circuit is assumed biased at mid-range. Each row is one
board, with a static offset and gain error per channel drawn within the datasheet
maximum; each reading adds the ADC noise, from the SINAD and reduced by averaging;
voltages are rounded to one LSB and clipped to the range. A tone is a ratio to the
stimulus, so the channel offset cancels: voltage readings lose more than tones.

Every draw comes from `draws(seed, key, faults, replicas)`: one standard normal per
row that depends only on the seed, the test, the fault and the replica, so the same
board reads the same in any matrix and any order.

`measurements.Instrument` is the general instrument model, applied to any reading;
`observe` is the model of an embedded ADC as the cost model needs it, per kind of test.

## What a plan costs

```python
price = costmodel.prices("prices.csv")   # item, price_eur
c = costmodel.cost(plan, board, access, price)
c.euros, c.items, c.time
```

Each node read takes an ADC channel and, if it needs one, a buffer: an internal op-amp
while there are free ones, then an external op-amp channel. AC tests take one more
channel for the stimulus. Channels beyond the free ones go through 8:1 multiplexers.
Each stimulus source needs a switch to inject the DAC output. The test time is the sum
of the acquisitions and is kept apart, as a limit, not added to the euros. The prices
are inputs of the study, not data of the package.
