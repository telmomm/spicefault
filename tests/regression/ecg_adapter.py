"""The ECG front-end study expressed with spicefault, for the equivalence tests.

`ecgfd` keeps what is specific to the application (circuits, specifications,
electrodes); the generic parts come from spicefault.
"""

from __future__ import annotations

import ecgfd.dataset
import ecgfd.simulate
import ecgfd.specs
import ecgfd.spice
from ecgfd.circuit import ELECTRODES, CircuitInstance, get_circuit, nominal_instance
from ecgfd.faults import Fault
from ecgfd.sampling import passive_tolerance

import spicefault
from spicefault import OperatingCondition, VariationSet
from spicefault.faults import (
    CompositeFault,
    FaultRule,
    InsertSeries,
    OpenCircuit,
    ParametricFault,
    ShortCircuit,
)
from spicefault.netlist import Netlist
from spicefault.simulation import SimulationError, run_deck
from spicefault.variation import ToleranceVariation


def to_fault(fault: Fault, healthy: CircuitInstance, cfg: dict) -> spicefault.Fault | None:
    """The `ecgfd` fault as a spicefault fault; None for the healthy condition.

    This is the mapping of docs/FAULT_MODEL.md, section 6. Identifier, reported type
    and the `origin` label are those of `ecgfd`. Amplifiers are subcircuit instances,
    so designator `U2` is element `XU2` in the netlist.
    """
    fcfg = cfg["faults"]
    kind, target, level = fault.kind, fault.target, fault.level
    common = {"fault_type": kind, "fault_id": fault.id, "tags": {"origin": fault.origin}}
    if kind == "healthy":
        return None
    if kind == "open":
        return OpenCircuit(target, r_open=fcfg["r_open"], **common)
    if kind == "short":
        return ShortCircuit(target, r_short=fcfg["r_short"], **common)
    if kind == "parametric":
        return ParametricFault(target, deviation=level, **common)
    if kind == "cap_degradation":
        esr = float(next(d["esr"] for d in fcfg["cap_degradation"] if d["c_loss"] == level))
        parts = [
            ParametricFault(target, factor=1.0 - level),
            spicefault.Fault("esr", [InsertSeries(target, 2, esr)], model_parameters={"esr": esr}),
        ]
        return CompositeFault(parts=parts, magnitude=level, unit="capacitance loss", **common)
    if kind in ("opamp_vos", "ina_vos"):
        # the healthy offset is drawn around zero, within +-vos_max
        limit = float(cfg["opamp" if kind == "opamp_vos" else "ina"]["vos_max"])
        return ParametricFault(
            f"X{target}", "vos", fault_value=level, nominal_value=0.0, reference=limit, **common
        )
    if kind == "opamp_aol":
        return ParametricFault(f"X{target}", "aol", factor=level, **common)
    if kind == "ina_cmrr":
        # the model takes the rejection as a signed ratio; the sign is part of the draw
        ratio = healthy.inas[target]["cmrr_sign"] * 10 ** (level / 20)
        return ParametricFault(f"X{target}", "cmrr", fault_value=ratio, **common)
    if kind == "ina_gain":
        return ParametricFault(f"X{target}", "gerr", fault_value=level, **common)
    if kind == "electrode_off":
        # the series resistance is replaced, not extended: an open reported as parametric
        r_off = float(fcfg["electrode"]["r_off"])
        return ParametricFault(f"Rs_{target}", fault_value=r_off, **common)
    if kind == "electrode_high_z":
        parts = []
        for name in target.split("+"):
            parts += [
                ParametricFault(f"Rd_{name}", factor=level),
                ParametricFault(f"Cd_{name}", divisor=level),
            ]
        return CompositeFault(parts=parts, magnitude=level, unit="impedance factor", **common)
    raise ValueError(f"unknown fault kind: {kind}")


def fault_rules(cfg: dict) -> list[FaultRule]:
    """Applicability rules that generate the fault universe of the selected ECG circuit.

    The catalogue of `ecgfd.faults.fault_catalogue`, stated as rules over the netlist.
    """
    fcfg = cfg["faults"]
    circuit = get_circuit(cfg)
    nominal = nominal_instance(cfg)

    def rule(kind: str, levels, components, target=lambda name: name) -> FaultRule:
        def make(component):
            return [
                to_fault(Fault(kind, target(component.name), float(level)), nominal, cfg)
                for level in levels
            ]

        return FaultRule(kind, make, components=tuple(components))

    def designator(name: str) -> str:
        return name[1:]  # XU2 -> U2

    def electrode(name: str) -> str:
        return name.split("_")[1]  # Rs_la -> la

    passives = [p.name for p in circuit.passives]
    capacitors = [p.name for p in circuit.passives if p.kind == "C"]
    opamps = [f"X{u.name}" for u in circuit.opamps]
    inas = [f"X{u.name}" for u in circuit.inas]
    ecfg = fcfg["electrode"]

    def high_z(component):
        # single-electrode high impedance is an imbalance; "la+ra" is the balanced case
        targets = ["la", "la+ra"] if component.name == "Rd_la" else [electrode(component.name)]
        return [
            to_fault(Fault("electrode_high_z", t, float(k)), nominal, cfg)
            for t in targets
            for k in ecfg["high_z_factor"]
        ]

    return [
        rule("open", [0.0], passives),
        rule("short", [0.0], passives),
        rule("parametric", fcfg["parametric"], passives),
        rule("cap_degradation", [d["c_loss"] for d in fcfg["cap_degradation"]], capacitors),
        rule("opamp_vos", fcfg["opamp"]["vos"], opamps, designator),
        rule("opamp_aol", fcfg["opamp"]["aol_factor"], opamps, designator),
        rule("ina_vos", fcfg["ina"]["vos"], inas, designator),
        rule("ina_cmrr", fcfg["ina"]["cmrr_db"], inas, designator),
        rule("ina_gain", fcfg["ina"]["gain_error"], inas, designator),
        rule("electrode_off", [0.0], [f"Rs_{n}" for n in ELECTRODES], electrode),
        FaultRule("electrode_high_z", high_z, components=tuple(f"Rd_{n}" for n in ELECTRODES)),
    ]


def variations(cfg: dict) -> VariationSet:
    """Tolerances of the passives, as `ecgfd.sampling` draws them."""
    distribution = cfg["tolerances"]["distribution"]
    return VariationSet(
        [
            ToleranceVariation(p.name, passive_tolerance(p.name, cfg), distribution)
            for p in get_circuit(cfg).passives
        ]
    )


def inject(net: Netlist, fault: Fault, healthy: CircuitInstance, cfg: dict) -> Netlist:
    """Apply an `ecgfd` fault to the netlist of the healthy circuit."""
    converted = to_fault(fault, healthy, cfg)
    if converted is not None:
        converted.apply(net)
    return net


def _run_deck(netlist: str, timeout: float = 120.0, spiceinit: str | None = None):
    try:
        return run_deck(netlist, timeout, spiceinit)
    except SimulationError as exc:
        raise ecgfd.spice.SimulationError(str(exc)) from exc


def simulate_task(task, cfg: dict):
    """`ecgfd.dataset.simulate_task` with every ngspice run going through spicefault."""
    ecgfd.simulate.run_deck = _run_deck
    ecgfd.specs.run_deck = _run_deck
    return ecgfd.dataset.simulate_task(task, cfg)


def bench_condition(cfg: dict) -> OperatingCondition:
    """The test bench of the specification measurements, as an operating condition.

    Same settings as `ecgfd.specs._bench`: leads connected directly to the body and
    the standard common-mode source. Applied after the fault, it replaces the
    electrodes, so an electrode fault leaves no trace on the bench.
    """
    source = cfg["specs"]["test_network"]["cm_source"]
    settings: dict[tuple[str, str], float] = {}
    for name in ELECTRODES:
        settings[f"Vhc_{name}", "dc"] = 0.0
        settings[f"Rs_{name}", "value"] = 1.0
        settings[f"Rd_{name}", "value"] = 1.0
        settings[f"Cd_{name}", "value"] = 1e-12
    settings["Cpow", "value"] = float(source["c_series"])
    settings["Cbody", "value"] = float(source["c_shunt"])
    settings["Riso", "value"] = 1.0
    return OperatingCondition("bench", settings=settings)
