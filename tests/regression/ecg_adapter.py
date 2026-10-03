"""The ECG front-end study expressed with spicefault, for the equivalence tests.

`ecgfd` keeps what is specific to the application (circuits, specifications,
electrodes); the generic parts come from spicefault.
"""

from __future__ import annotations

import ecgfd.dataset
import ecgfd.simulate
import ecgfd.specs
import ecgfd.spice
from ecgfd.circuit import CircuitInstance
from ecgfd.faults import Fault

from spicefault.netlist import Netlist
from spicefault.simulation import SimulationError, run_deck


def inject(net: Netlist, fault: Fault, healthy: CircuitInstance, cfg: dict) -> Netlist:
    """Apply an `ecgfd` fault to the netlist of the healthy circuit, with the primitives.

    This is the mapping of docs/FAULT_MODEL.md, section 6. Amplifiers are subcircuit
    instances, so designator `U2` is element `XU2` in the netlist.
    """
    fcfg = cfg["faults"]
    kind, target, level = fault.kind, fault.target, fault.level
    if kind == "healthy":
        pass
    elif kind == "open":
        net.insert_series(target, 2, float(fcfg["r_open"]))
    elif kind == "short":
        net.insert_parallel(target, 1, 2, float(fcfg["r_short"]))
    elif kind == "parametric":
        net.set_parameter(target, "value", "relative", level)
    elif kind == "cap_degradation":
        esr = next(d["esr"] for d in fcfg["cap_degradation"] if d["c_loss"] == level)
        net.set_parameter(target, "value", "scale", 1.0 - level)
        net.insert_series(target, 2, float(esr))
    elif kind in ("opamp_vos", "ina_vos"):
        net.set_parameter(f"X{target}", "vos", "absolute", level)
    elif kind == "opamp_aol":
        net.set_parameter(f"X{target}", "aol", "scale", level)
    elif kind == "ina_cmrr":
        # the model takes the rejection as a signed ratio; the sign is part of the draw
        ratio = healthy.inas[target]["cmrr_sign"] * 10 ** (level / 20)
        net.set_parameter(f"X{target}", "cmrr", "absolute", ratio)
    elif kind == "ina_gain":
        net.set_parameter(f"X{target}", "gerr", "absolute", level)
    elif kind == "electrode_off":
        net.set_parameter(f"Rs_{target}", "value", "absolute", float(fcfg["electrode"]["r_off"]))
    elif kind == "electrode_high_z":
        for name in target.split("+"):
            net.set_parameter(f"Rd_{name}", "value", "scale", level)
            net.set_parameter(f"Cd_{name}", "value", "divide", level)
    else:
        raise ValueError(f"unknown fault kind: {kind}")
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
