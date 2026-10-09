"""Discrete series voltage regulator: device-level models, not a filter.

A pass transistor, a one-transistor error amplifier, a zener reference and a feedback
divider. Its transistors and its zener depend on temperature, so temperature is a
meaningful operating condition here, together with input voltage and load.

The pass transistor (FZT603, an NPN Darlington), the error amplifier (one transistor
of a DMMT3904W pair) and the reference (BZX84C4V7, a 4.7 V zener) have the SPICE models
published by their manufacturer, Diodes Incorporated; the netlist gives their source
and the notice they are distributed under, which is not the licence of this repository.
The pass transistor and the zener are subcircuits, so they are instances XQ1 and XD1.
The spread of the devices themselves (current gain, zener voltage) is not modelled:
only the passives vary.

The zener model alone gives a reference that rises with temperature (+3.13 mV/K at
5 mA), where the data sheet of the part gives a coefficient between -3.5 and +0.2 mV/K.
The netlist therefore puts a source in series with it, zero at 27 C, that brings the
coefficient to the middle of that range, -1.65 mV/K: the model is left intact and the
correction is declared as ours. The data sheet gives a range and no typical value, so
the size of the temperature drift of this circuit is one plausible value and not that
of a measured part; the temperature behaviour of the two transistor models has not been
checked against their data sheets.
"""

from __future__ import annotations

from spicefault import Measurement, OperatingCondition, SimulationConfig
from spicefault.faults import (
    FaultRule,
    OpenCircuit,
    ShortCircuit,
    open_rule,
    parametric_rule,
    short_rule,
)
from spicefault.variation import tolerances

from .study import Study, load_circuit

PASSIVES = ("R1", "R2", "R3", "R4", "C1")  # the load RL is an operating condition
DEVIATIONS = [-0.5, -0.25, -0.1, 0.1, 0.25, 0.5]
TERMINALS = {1: "collector", 2: "base", 3: "emitter"}  # also the pins of the FZT603
TRANSISTORS = ("FZT603", "DI_DMMT3904W")
ZENER = ("BZX84C4V7_TC",)  # the BZX84C4V7 with its temperature correction; anode, cathode
PAIRS = {(1, 3): "collector-emitter", (2, 3): "base-emitter", (1, 2): "collector-base"}


def transistor_rule() -> FaultRule:
    """Every terminal open, and every pair of terminals shorted."""

    def make(component):
        name = component.name
        opens = [
            OpenCircuit(name, terminal, fault_id=f"{name}:open:{label}")
            for terminal, label in TERMINALS.items()
        ]
        shorts = [
            ShortCircuit(name, pair, fault_id=f"{name}:short:{label}")
            for pair, label in PAIRS.items()
        ]
        return opens + shorts

    return FaultRule("transistor", make, kinds="QX", models=TRANSISTORS)


def study() -> Study:
    circuit = load_circuit("regulator")
    measurements = [
        Measurement.value("v(out)", name="output_voltage"),
        Measurement.value("v(nz)", name="reference_voltage"),
        Measurement.value("v(nb)", name="base_voltage"),
        Measurement.value("v(nd)", name="divider_voltage"),
        Measurement.value("i(vin)", name="input_current"),
        Measurement.magnitude("v(out)", 100.0, analysis=1, name="ripple_gain_100"),
        Measurement.magnitude("v(out)", 10e3, analysis=1, name="ripple_gain_10k"),
        Measurement.magnitude("v(out)", 100.0, analysis=2, name="output_impedance_100"),
    ]
    nominal = OperatingCondition()

    def condition(name, vin=15.0, load=100.0, temperature=None):
        return OperatingCondition(name, temperature, {("Vin", "dc"): vin, ("RL", "value"): load})

    corners = [
        condition(f"{vin:g}V_{load:g}ohm_{t:g}C", vin, load, t)
        for vin in (13.5, 18.0)
        for load in (68.0, 1000.0)
        for t in (-20.0, 85.0)
    ]
    return Study(
        name="regulator",
        description="Discrete series voltage regulator, 15 V to 10 V",
        circuit=circuit,
        rules=[
            open_rule("RC", components=PASSIVES),
            short_rule("RC", components=PASSIVES),
            parametric_rule(DEVIATIONS, "RC", components=PASSIVES),
            transistor_rule(),
            FaultRule("diode_open", lambda c: [OpenCircuit(c.name)], kinds="X", models=ZENER),
            FaultRule("diode_short", lambda c: [ShortCircuit(c.name)], kinds="X", models=ZENER),
        ],
        variations=tolerances(circuit, {"R": 0.01, "C": 0.20}, components=PASSIVES),
        config=SimulationConfig(
            ("op", "ac dec 10 10 1e5", "alter @vin[acmag]=0", "alter @iout[acmag]=1",
             "ac dec 10 10 1e5"),
            outputs=("v(out)", "v(nb)", "v(nd)", "v(nz)", "i(vin)"),
        ),  # fmt: skip
        measurements=measurements,
        waveform=None,
        specifications=SPECIFICATIONS,
        noise=NOISE,
        conditions={
            "nominal": [nominal],
            # one factor at a time around the nominal condition
            "one_factor": [
                nominal,
                condition("low_line", vin=13.5),
                condition("high_line", vin=18.0),
                condition("heavy_load", load=68.0),
                condition("light_load", load=1000.0),
                OperatingCondition("cold", -20.0),
                OperatingCondition("hot", 85.0),
            ],
            "corners": [nominal, *corners],
        },
    )


# Declared limits, of the kind a regulator is specified with: output within 7 % of
# 10.4 V, ripple gain at 100 Hz at most 0.2 (14 dB of rejection), output impedance at
# most 0.6 ohm. The nominal circuit meets them at the nominal condition (10.39 V, 0.144,
# 0.094 ohm) and at each of the eight corners of input voltage, load and temperature
# (9.85 to 11.06 V, at most 0.155, at most 0.512 ohm), and so did every one of 600
# healthy circuits at the nominal condition (seed 7). The rejection is poor because the
# zener has a dynamic resistance of about 80 ohm, in series with the emitter of the
# error amplifier. Fixed before any fault campaign was run with these device models.
SPECIFICATIONS = {
    "output_voltage": (9.65, 11.15),
    "ripple_gain_100": (0.0, 0.2),
    "output_impedance_100": (0.0, 0.6),
}
NOISE = {
    "output_voltage": 0.002,
    "reference_voltage": 0.002,
    "base_voltage": 0.002,
    "divider_voltage": 0.002,
    "input_current": 0.0002,
    "ripple_gain_100": 0.0002,
    "ripple_gain_10k": 0.0002,
    "output_impedance_100": 0.002,
}
