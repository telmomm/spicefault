"""The ECG front-end study expressed with spicefault, for the equivalence tests.

`ecgfd` keeps what is specific to the application (circuits, specifications,
electrodes); the generic parts come from spicefault.
"""

from __future__ import annotations

import ecgfd.dataset
import ecgfd.simulate
import ecgfd.specs
import ecgfd.spice
from ecgfd.circuit import ELECTRODES, CircuitInstance
from ecgfd.faults import HARD_KINDS, Fault

import spicefault
from spicefault import OperatingCondition
from spicefault.faults import InsertParallel, InsertSeries, SetParameter
from spicefault.netlist import Netlist
from spicefault.simulation import SimulationError, run_deck


def to_fault(fault: Fault, healthy: CircuitInstance, cfg: dict) -> spicefault.Fault | None:
    """The `ecgfd` fault as a spicefault fault; None for the healthy condition.

    This is the mapping of docs/FAULT_MODEL.md, section 6. Amplifiers are subcircuit
    instances, so designator `U2` is element `XU2` in the netlist.
    """
    fcfg = cfg["faults"]
    kind, target, level = fault.kind, fault.target, fault.level
    model: dict[str, float] = {}
    if kind == "healthy":
        return None
    if kind == "open":
        model = {"r_open": float(fcfg["r_open"])}
        primitives = [InsertSeries(target, 2, model["r_open"])]
    elif kind == "short":
        model = {"r_short": float(fcfg["r_short"])}
        primitives = [InsertParallel(target, 1, 2, model["r_short"])]
    elif kind == "parametric":
        primitives = [SetParameter(target, "value", "relative", level)]
    elif kind == "cap_degradation":
        esr = next(d["esr"] for d in fcfg["cap_degradation"] if d["c_loss"] == level)
        primitives = [
            SetParameter(target, "value", "scale", 1.0 - level),
            InsertSeries(target, 2, float(esr)),
        ]
    elif kind in ("opamp_vos", "ina_vos"):
        primitives = [SetParameter(f"X{target}", "vos", "absolute", level)]
    elif kind == "opamp_aol":
        primitives = [SetParameter(f"X{target}", "aol", "scale", level)]
    elif kind == "ina_cmrr":
        # the model takes the rejection as a signed ratio; the sign is part of the draw
        ratio = healthy.inas[target]["cmrr_sign"] * 10 ** (level / 20)
        primitives = [SetParameter(f"X{target}", "cmrr", "absolute", ratio)]
    elif kind == "ina_gain":
        primitives = [SetParameter(f"X{target}", "gerr", "absolute", level)]
    elif kind == "electrode_off":
        model = {"r_off": float(fcfg["electrode"]["r_off"])}
        primitives = [SetParameter(f"Rs_{target}", "value", "absolute", model["r_off"])]
    elif kind == "electrode_high_z":
        primitives = []
        for name in target.split("+"):
            primitives += [
                SetParameter(f"Rd_{name}", "value", "scale", level),
                SetParameter(f"Cd_{name}", "value", "divide", level),
            ]
    else:
        raise ValueError(f"unknown fault kind: {kind}")
    return spicefault.Fault(
        fault_type=kind,
        primitives=primitives,
        fault_id=fault.id,
        magnitude=None if kind in HARD_KINDS else level,
        model_parameters=model,
        tags={"origin": fault.origin},
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
