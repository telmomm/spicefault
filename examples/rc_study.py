"""The study the examples share: an RC low-pass filter, its faults and its measurements.

The other examples import this module. Datasets are written under `examples/output/`,
or under the folder named by the SPICEFAULT_EXAMPLES_OUTPUT environment variable.
"""

from __future__ import annotations

import os
from pathlib import Path

from spicefault import Circuit, FaultCampaign, Measurement, SimulationConfig, Waveform
from spicefault.faults import FaultUniverse, open_rule, parametric_rule, short_rule
from spicefault.variation import tolerances

HERE = Path(__file__).parent
OUTPUT = Path(os.environ.get("SPICEFAULT_EXAMPLES_OUTPUT", HERE / "output"))
WORKERS = 2

# 10 kOhm, 100 nF and a 10 kOhm load: 0.5 V/V at DC, corner at 318 Hz
circuit = Circuit.from_netlist(HERE / "filter" / "rc_lowpass.cir")

# every resistor and capacitor can be open, shorted, or off its value
universe = FaultUniverse(
    circuit,
    [open_rule(), short_rule(), parametric_rule([-0.5, -0.2, 0.2, 0.5])],
)

# 1 % resistors, 5 % capacitors
population = tolerances(circuit, {"R": 0.01, "C": 0.05})

# an operating point, a frequency sweep and the response to a pulse
config = SimulationConfig(("op", "ac dec 20 1 1e5", "tran 10u 10m"), outputs=("v(out)",))
measurements = [
    Measurement.value("v(out)", name="dc"),
    Measurement.magnitude("v(out)", 100.0, name="gain_100"),
    Measurement.magnitude("v(out)", 1e3, name="gain_1k"),
    Measurement.phase("v(out)", 1e3, name="phase_1k"),
    Measurement.peak("v(out)", name="peak"),
]
waveform = Waveform("v(out)", fs=10e3, duration=10e-3)


def campaign(name: str = "rc_campaign", tolerance_scale: float = 1.0) -> FaultCampaign:
    """300 healthy circuits and 20 of each of the 18 faults: 660 simulations."""
    return FaultCampaign(
        circuit,
        universe.selected(),
        out_dir=OUTPUT / name,
        samples_per_fault=20,
        healthy_samples=300,
        variations=population.scaled(tolerance_scale),
        config=config,
        measurements=measurements,
        waveform=waveform,
        seed=42,
        seeding="content",
    )
