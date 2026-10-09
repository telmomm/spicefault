"""Four-op-amp biquad high-pass filter: the larger benchmark of the diagnosis literature.

The circuit of Fig. 2 of F. Aminian, M. Aminian and H. W. Collins, "Analog fault
diagnosis of actual circuits using neural networks", IEEE Trans. Instrum. Meas.,
vol. 51, no. 3, pp. 544-550, 2002 (doi:10.1109/TIM.2002.1017726): a lossy integrator,
an integrator, an inverter and an output summer, with a cut-off at 10 kHz, a quality
factor of about 2 and unity gain in the pass band. R1 to R4, C1 and C2 are named as in
that paper; its figure repeats two labels on the other six resistors, which are R5 to
R10 here. Excitation and tolerances as in the Sallen-Key study, and as in that paper.
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
        Measurement.magnitude("v(out)", 3.3e3, name="gain_3k3"),
        Measurement.magnitude("v(out)", 10e3, name="gain_10k"),
        Measurement.magnitude("v(out)", 30e3, name="gain_30k"),
        Measurement.phase("v(out)", 10e3, name="phase_10k"),
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


# Declared limits: as for the Sallen-Key study, the range that held the central 99 % of
# 600 healthy circuits at the declared tolerances (seed 7), rounded outwards, around
# nominal values of 6.80 kHz, 1.00 and 2.05. The corner frequency is where the response
# is 3 dB below the pass band; with a quality factor of 2 it lies below the 10 kHz
# cut-off of the design. Fixed before any fault campaign was run on this circuit.
SPECIFICATIONS = {
    "corner_frequency": (5.7e3, 8.0e3),
    "passband_gain": (0.91, 1.09),
    "peak_gain": (1.75, 2.45),
}
NOISE = {
    "corner_frequency": 7.0,
    "passband_gain": 0.002,
    "peak_gain": 0.004,
    "gain_3k3": 0.0002,
    "gain_10k": 0.004,
    "gain_30k": 0.002,
    "phase_10k": 0.2,
    "pulse_peak": 0.0055,
    "pulse_minimum": 0.01,
    "pulse_rms": 0.001,
}
