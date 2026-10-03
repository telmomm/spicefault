"""The fault types of the first release (docs/FAULT_MODEL.md, section 3).

Every type has a physical interpretation, stated in its docstring, and is a list of
primitives. `fault_type` is the reported type; it defaults to the name of the class
and can be overridden when an application reports a fault under another name.
"""

from __future__ import annotations

from collections.abc import Sequence

from .base import Fault, InsertParallel, InsertSeries, SetParameter, register
from .severity import FaultSeverity

# what a designer calls the value of a passive component
_VALUE_NAMES = ("resistance", "capacitance", "inductance")


def _terminals(default: tuple[int, int], terminals: tuple[int, int]) -> str:
    return "" if tuple(terminals) == default else f":t{terminals[0]}-{terminals[1]}"


@register("open_circuit")
class OpenCircuit(Fault):
    """Broken connection: cracked solder joint, lifted lead, open track or open component.

    One terminal is reconnected through `r_open`. An ideal open would leave a node
    floating and make the operating point singular, so `r_open` is finite; it belongs
    to the model and is recorded. It must be large against the impedances around the
    component. On a two-terminal component either terminal gives the same circuit.
    """

    def __init__(
        self,
        component: str,
        terminal: int = 2,
        r_open: float = 1e9,
        *,
        fault_type: str = "open_circuit",
        fault_id: str = "",
        tags: dict[str, str] | None = None,
    ):
        r_open = float(r_open)
        where = "" if terminal == 2 else f":t{terminal}"
        super().__init__(
            fault_type=fault_type,
            primitives=(InsertSeries(component, terminal, r_open),),
            fault_id=fault_id or f"{component}:open{where}",
            model_parameters={"r_open": r_open},
            nominal_state="connected",
            fault_state="open",
            tags=tags or {},
        )


@register("short_circuit")
class ShortCircuit(Fault):
    """Solder bridge, dielectric breakdown or conductive contamination across a component.

    `r_short` is added between two terminals. It is a model parameter: small against
    the impedances around the component, and not zero, which would create a loop of
    ideal elements.
    """

    def __init__(
        self,
        component: str,
        terminals: tuple[int, int] = (1, 2),
        r_short: float = 1.0,
        *,
        fault_type: str = "short_circuit",
        fault_id: str = "",
        tags: dict[str, str] | None = None,
    ):
        r_short = float(r_short)
        super().__init__(
            fault_type=fault_type,
            primitives=(InsertParallel(component, terminals[0], terminals[1], r_short),),
            fault_id=fault_id or f"{component}:short{_terminals((1, 2), terminals)}",
            model_parameters={"r_short": r_short},
            nominal_state="isolated",
            fault_state="shorted",
            tags=tags or {},
        )


@register("leakage")
class LeakageFault(Fault):
    """Finite parasitic conduction: capacitor insulation loss, board contamination,
    degraded junction.

    The same primitive as a short, graded: `resistance` is the magnitude of the fault,
    usually studied over decades. With `r_min` and `r_max` the severity is taken on a
    logarithmic scale, 0 at `r_max` and 1 at `r_min`.
    """

    def __init__(
        self,
        component: str,
        resistance: float,
        terminals: tuple[int, int] = (1, 2),
        *,
        r_min: float | None = None,
        r_max: float | None = None,
        fault_type: str = "leakage",
        fault_id: str = "",
        tags: dict[str, str] | None = None,
    ):
        resistance = float(resistance)
        if (r_min is None) != (r_max is None):
            raise ValueError("give both r_min and r_max, or neither")
        super().__init__(
            fault_type=fault_type,
            primitives=(InsertParallel(component, terminals[0], terminals[1], resistance),),
            fault_id=fault_id
            or f"{component}:leakage:{resistance:g}{_terminals((1, 2), terminals)}",
            magnitude=resistance,
            unit="ohm",
            severity=None
            if r_min is None
            else FaultSeverity.log_resistance(resistance, r_min, r_max),
            nominal_state="isolated",
            fault_state=f"{resistance:g} ohm in parallel",
            tags=tags or {},
        )


@register("parametric")
class ParametricFault(Fault):
    """A parameter outside its tolerance band: wrong part fitted, aged or stressed
    component, degraded device parameter.

    Exactly one of:
    - `deviation`: x -> x (1 + deviation), on the value the component has after its
      normal variation;
    - `factor`: x -> factor * x, likewise;
    - `divisor`: x -> x / divisor, likewise;
    - `fault_value`: x -> fault_value, whatever the component had.

    The first three compound with the manufacturing spread of the faulty component;
    the last one replaces it. Severity is |deviation| (|factor - 1|, |1/divisor - 1|)
    for the relative forms. For `fault_value` it is |fault_value - nominal_value| /
    reference, with `reference` defaulting to `nominal_value`; a parameter that is
    nominally zero, such as an offset voltage, needs an explicit reference.
    `parameter` may be `resistance`, `capacitance` or `inductance` for the value of a
    passive component.
    """

    def __init__(
        self,
        component: str,
        parameter: str = "value",
        *,
        deviation: float | None = None,
        factor: float | None = None,
        divisor: float | None = None,
        fault_value: float | None = None,
        nominal_value: float | None = None,
        reference: float | None = None,
        severity: FaultSeverity | float | None = None,
        fault_type: str = "parametric",
        fault_id: str = "",
        tags: dict[str, str] | None = None,
    ):
        given = {
            "relative": deviation, "scale": factor, "divide": divisor, "absolute": fault_value,
        }  # fmt: skip
        given = {rule: float(v) for rule, v in given.items() if v is not None}
        if len(given) != 1:
            raise ValueError("give exactly one of deviation, factor, divisor and fault_value")
        (rule, value), = given.items()
        if parameter.lower() in _VALUE_NAMES:
            parameter = "value"
        where = component if parameter == "value" else f"{component}.{parameter}"

        if rule == "relative":
            relative, size, state = value, f"{value:+g}", f"{value:+.3g} relative to realised"
        elif rule == "scale":
            relative, size, state = value - 1.0, f"x{value:g}", f"realised value times {value:g}"
        elif rule == "divide":
            relative, size = 1.0 / value - 1.0, f"/{value:g}"
            state = f"realised value over {value:g}"
        else:
            relative, size, state = None, f"={value:g}", f"set to {value:g}"

        if severity is None:
            if relative is not None:
                severity = FaultSeverity.relative_deviation(relative)
            elif nominal_value is not None and (reference or nominal_value):
                severity = FaultSeverity.deviation_over_reference(
                    value, nominal_value, reference or nominal_value
                )
        super().__init__(
            fault_type=fault_type,
            primitives=(SetParameter(component, parameter, rule, value),),
            fault_id=fault_id or f"{where}:parametric:{size}",
            magnitude=value,
            unit={"relative": "relative deviation", "scale": "factor", "divide": "divisor"}.get(
                rule, "parameter value"
            ),
            severity=severity,
            nominal_state="within tolerance",
            fault_state=state,
            tags=tags or {},
        )


@register("composite")
class CompositeFault(Fault):
    """One physical mechanism with several electrical consequences.

    For example an ageing electrolytic capacitor loses capacitance and gains series
    resistance together. The parts are applied in order. This is one fault, not
    several independent ones occurring at once.
    """

    def __init__(
        self,
        fault_type: str,
        parts: Sequence[Fault],
        *,
        fault_id: str = "",
        magnitude: float | None = None,
        unit: str = "",
        severity: FaultSeverity | float | None = None,
        nominal_state: str = "",
        fault_state: str = "",
        tags: dict[str, str] | None = None,
    ):
        model: dict[str, float] = {}
        for part in parts:
            model.update(part.model_parameters)
        super().__init__(
            fault_type=fault_type,
            primitives=tuple(p for part in parts for p in part.primitives),
            fault_id=fault_id,
            magnitude=magnitude,
            unit=unit,
            severity=severity,
            model_parameters=model,
            nominal_state=nominal_state,
            fault_state=fault_state or "; ".join(p.fault_state for p in parts if p.fault_state),
            tags=tags or {},
        )
