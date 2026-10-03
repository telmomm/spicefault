"""Four-op-amp biquad high-pass filter: the larger benchmark of the diagnosis literature.

This is a circuit of that class, designed here: a summer, two integrators and an
output stage, with a corner at about 5.1 kHz, quality factor 1 and unity gain. It is
not the schematic of a cited paper, and must be aligned with one before publication.
Excitation and tolerances as in the Sallen-Key study.
"""

from __future__ import annotations

from spicefault import Measurement, SimulationConfig, Waveform
from spicefault.faults import open_rule, parametric_rule, short_rule
from spicefault.variation import ToleranceVariation, tolerances

from . import functions
from .sallen_key import DEVIATIONS, amplifier_rule
from .study import Study, load_circuit


def study() -> Study:
    circuit = load_circuit("biquad")
    amplifiers = [c.name for c in circuit.components() if c.kind == "X"]
    measurements = [
        Measurement.custom("corner_frequency", functions.corner_frequency, analysis="ac"),
        Measurement.custom("passband_gain", functions.passband_gain, analysis="ac"),
        Measurement.custom("peak_gain", functions.peak_gain, analysis="ac"),
        Measurement.magnitude("v(out)", 1.7e3, name="gain_1k7"),
        Measurement.magnitude("v(out)", 5.1e3, name="gain_5k1"),
        Measurement.magnitude("v(out)", 15e3, name="gain_15k"),
        Measurement.phase("v(out)", 5.1e3, name="phase_5k1"),
        Measurement.peak("v(out)", name="pulse_peak"),
        Measurement.minimum("v(out)", name="pulse_minimum"),
        Measurement.rms("v(out)", name="pulse_rms"),
    ]
    return Study(
        name="biquad",
        description="Four-op-amp biquad high-pass filter",
        circuit=circuit,
        rules=[open_rule("RC"), short_rule("RC"), parametric_rule(DEVIATIONS, "RC"),
               amplifier_rule()],  # fmt: skip
        variations=tolerances(circuit, {"R": 0.05, "C": 0.10})
        + [ToleranceVariation(name, 0.2, parameter="gbw") for name in amplifiers],
        config=SimulationConfig(("ac dec 100 100 2e5", "tran 2u 1.4m"), outputs=("v(out)",)),
        measurements=measurements,
        waveform=Waveform("v(out)", fs=250e3, duration=1.2e-3),
        specifications=SPECIFICATIONS,
        noise=NOISE,
    )


# Declared limits: corner frequency within 15 %, pass-band gain within 10 % and
# peaking within 20 % of the values of the nominal circuit (4.03 kHz, 1.00 and 1.158).
SPECIFICATIONS = {
    "corner_frequency": (3.43e3, 4.65e3),
    "passband_gain": (0.90, 1.10),
    "peak_gain": (0.92, 1.39),
}
NOISE = {
    "corner_frequency": 4.0,
    "passband_gain": 0.002,
    "peak_gain": 0.0023,
    "gain_1k7": 0.0002,
    "gain_5k1": 0.002,
    "gain_15k": 0.002,
    "phase_5k1": 0.2,
    "pulse_peak": 0.01,
    "pulse_minimum": 0.0035,
    "pulse_rms": 0.001,
}
