"""Discrete series voltage regulator: device-level models, not a filter.

A pass transistor, a one-transistor error amplifier, a zener reference and a feedback
divider. Its transistors and its zener depend on temperature, so temperature is a
meaningful operating condition here, together with input voltage and load.

The device models are generic parameter sets without a cited source and must be
replaced by vendor models before any result is published. The spread of the devices
themselves (current gain, zener voltage) is not modelled: only the passives vary.
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
TERMINALS = {1: "collector", 2: "base", 3: "emitter"}
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

    return FaultRule("transistor", make, kinds="Q")


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
            FaultRule("diode_open", lambda c: [OpenCircuit(c.name)], kinds="D"),
            FaultRule("diode_short", lambda c: [ShortCircuit(c.name)], kinds="D"),
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


# Declared limits: output within 3 % of 10.03 V, ripple gain at 100 Hz at most 0.05
# (26 dB of rejection), output impedance at most 0.5 ohm.
SPECIFICATIONS = {
    "output_voltage": (9.73, 10.33),
    "ripple_gain_100": (0.0, 0.05),
    "output_impedance_100": (0.0, 0.5),
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
