"""Sallen-Key band-pass filter: the small benchmark of the fault-diagnosis literature.

The circuit, its component values and its designators are those of Fig. 3 of
M. Aminian and F. Aminian, "Neural-network based analog-circuit fault diagnosis using
wavelet transform as preprocessor", IEEE Trans. Circuits Syst. II, vol. 47, no. 2,
pp. 151-156, 2000 (doi:10.1109/82.823545): centre frequency 25 kHz. So are the
excitation, a single 5 V pulse of 10 us, and the tolerances, 5 % for resistors and 10 %
for capacitors. A later paper of the same authors (IEEE Trans. Instrum. Meas., 2002)
uses a Sallen-Key filter with other resistor values; this is the one of 2000.
"""

from __future__ import annotations

from spicefault import Measurement, SimulationConfig, Waveform
from spicefault.faults import FaultRule, ParametricFault, open_rule, parametric_rule, short_rule
from spicefault.variation import ToleranceVariation, tolerances

from . import functions
from .study import Study, load_circuit

DEVIATIONS = [-0.5, -0.25, -0.1, 0.1, 0.25, 0.5]


def amplifier_rule() -> FaultRule:
    """Degradation of an operational amplifier: loss of open-loop gain and of bandwidth."""

    def make(component):
        return [
            ParametricFault(component.name, "aol", factor=1e-3, fault_type="gain_loss"),
            ParametricFault(component.name, "gbw", factor=1e-2, fault_type="bandwidth_loss"),
        ]

    return FaultRule("amplifier", make, kinds="X")


def study() -> Study:
    circuit = load_circuit("sallen_key")
    measurements = [
        Measurement.custom("centre_frequency", functions.centre_frequency, analysis="ac"),
        Measurement.custom("peak_gain", functions.peak_gain, analysis="ac"),
        Measurement.custom("bandwidth", functions.bandwidth, analysis="ac"),
        Measurement.magnitude("v(out)", 10e3, name="gain_10k"),
        Measurement.magnitude("v(out)", 25e3, name="gain_25k"),
        Measurement.magnitude("v(out)", 60e3, name="gain_60k"),
        Measurement.phase("v(out)", 25e3, name="phase_25k"),
        Measurement.peak("v(out)", name="pulse_peak"),
        Measurement.minimum("v(out)", name="pulse_minimum"),
        Measurement.rms("v(out)", name="pulse_rms"),
    ]
    return Study(
        name="sallen_key",
        description="Sallen-Key band-pass filter, one operational amplifier",
        circuit=circuit,
        rules=[open_rule("RC"), short_rule("RC"), parametric_rule(DEVIATIONS, "RC"),
               amplifier_rule()],  # fmt: skip
        variations=tolerances(circuit, {"R": 0.05, "C": 0.10})
        + [ToleranceVariation("XU1", 0.2, parameter="gbw")],
        config=SimulationConfig(("ac dec 100 1k 1e6", "tran 0.5u 400u"), outputs=("v(out)",)),
        measurements=measurements,
        waveform=Waveform("v(out)", fs=500e3, duration=300e-6),
        specifications=SPECIFICATIONS,
        noise=NOISE,
    )


# Declared limits. With the tolerances this benchmark is usually given, the filter has
# no tight specification to meet: its gain depends on the ratio of two 5 % resistors
# close to the edge of instability, and in 600 healthy circuits (seed 7) the peak gain
# went from 1.01 to 22.8 and the bandwidth from 0.56 to 10.9 kHz, around nominal values
# of 2.00 and 6.06 kHz. The limits are therefore the range that held the central 99 %
# of those healthy circuits, rounded outwards: a circuit fails when it is outside what
# the declared design produces. They were fixed before any fault campaign was run.
SPECIFICATIONS = {
    "centre_frequency": (21.5e3, 27.5e3),
    "peak_gain": (1.0, 25.0),
    "bandwidth": (500.0, 11.5e3),
}
# An instrument that reads gains and voltages to 0.2 % of their design value,
# frequencies to 0.1 % and phase to 0.2 degrees.
NOISE = {
    "centre_frequency": 25.0,
    "peak_gain": 0.004,
    "bandwidth": 6.0,
    "gain_10k": 0.0005,
    "gain_25k": 0.004,
    "gain_60k": 0.0005,
    "phase_25k": 0.2,
    "pulse_peak": 0.004,
    "pulse_minimum": 0.005,
    "pulse_rms": 0.002,
}
