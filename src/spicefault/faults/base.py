"""Primitives and the fault built from them (docs/FAULT_MODEL.md, sections 1 and 7)."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field

from ..netlist import RULES, Netlist
from .severity import FaultSeverity

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SetParameter:
    component: str
    parameter: str
    rule: str  # absolute | relative | scale | divide
    value: float

    def __post_init__(self):
        if self.rule not in RULES:
            raise ValueError(f"unknown rule {self.rule!r}; expected one of {RULES}")

    def apply(self, netlist: Netlist) -> None:
        netlist.set_parameter(self.component, self.parameter, self.rule, self.value)


@dataclass(frozen=True)
class InsertSeries:
    component: str
    terminal: int  # 1-based
    resistance: float

    def apply(self, netlist: Netlist) -> None:
        netlist.insert_series(self.component, self.terminal, self.resistance)


@dataclass(frozen=True)
class InsertParallel:
    component: str
    terminal_a: int
    terminal_b: int
    resistance: float

    def apply(self, netlist: Netlist) -> None:
        netlist.insert_parallel(self.component, self.terminal_a, self.terminal_b, self.resistance)


Primitive = SetParameter | InsertSeries | InsertParallel
_OPS = {
    "set_parameter": SetParameter,
    "insert_series": InsertSeries,
    "insert_parallel": InsertParallel,
}
_OP_NAMES = {cls: op for op, cls in _OPS.items()}


# name written in the record of a fault -> class that rebuilds it
_CLASSES: dict[str, type[Fault]] = {}


def register(name: str):
    """Class decorator: faults of this class are recorded and rebuilt under `name`."""

    def decorator(cls):
        cls.class_name = name
        _CLASSES[name] = cls
        return cls

    return decorator


@register("fault")
@dataclass(frozen=True)
class Fault:
    """One fault condition: what is injected, and what is recorded about it.

    `magnitude` is the physical size of the fault, with `unit`; `severity` is an
    optional dimensionless number on a declared scale (a bare number is taken as
    declared by the user). `model_parameters` holds values that belong to the model
    and not to the defect, such as the resistance that stands for an open circuit.
    `tags` are application labels.

    This class builds a fault directly from primitives; `OpenCircuit`,
    `ShortCircuit`, `LeakageFault`, `ParametricFault` and `CompositeFault` are the
    usual ways to write one.
    """

    fault_type: str
    primitives: tuple[Primitive, ...]
    fault_id: str = ""
    magnitude: float | None = None
    unit: str = ""
    severity: FaultSeverity | None = None
    model_parameters: dict[str, float] = field(default_factory=dict)
    nominal_state: str = ""
    fault_state: str = ""
    tags: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "primitives", tuple(self.primitives))
        if not self.primitives:
            raise ValueError("a fault needs at least one primitive")
        if self.severity is not None and not isinstance(self.severity, FaultSeverity):
            object.__setattr__(self, "severity", FaultSeverity.declared(self.severity))
        if not self.fault_id:
            size = "" if self.magnitude is None else f":{self.magnitude:+g}"
            object.__setattr__(
                self, "fault_id", f"{'+'.join(self.components)}:{self.fault_type}{size}"
            )

    @property
    def components(self) -> tuple[str, ...]:
        """Components the fault acts on, in order of first appearance."""
        return tuple(dict.fromkeys(p.component for p in self.primitives))

    def with_tags(self, tags: dict[str, str]) -> Fault:
        """The same fault with `tags` added to its own; a tag of the same name is replaced."""
        tagged = copy.copy(self)
        object.__setattr__(tagged, "tags", {**self.tags, **tags})
        return tagged

    def apply(self, netlist: Netlist) -> None:
        """Inject the fault into an already realised netlist, primitive by primitive."""
        for primitive in self.primitives:
            primitive.apply(netlist)

    def metadata(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "fault_id": self.fault_id,
            "class": self.class_name,
            "fault_type": self.fault_type,
            "components": list(self.components),
            "primitives": [{"op": _OP_NAMES[type(p)], **asdict(p)} for p in self.primitives],
            "magnitude": None
            if self.magnitude is None
            else {"value": self.magnitude, "unit": self.unit},
            "severity": None if self.severity is None else self.severity.metadata(),
            "model_parameters": dict(self.model_parameters),
            "nominal_state": self.nominal_state,
            "fault_state": self.fault_state,
            "tags": dict(self.tags),
        }

    @classmethod
    def from_metadata(cls, record: dict) -> Fault:
        """Rebuild a fault from its record, as an instance of the class that wrote it."""
        if record["schema_version"] != SCHEMA_VERSION:
            raise ValueError(f"unsupported fault schema version {record['schema_version']}")
        primitives = []
        for item in record["primitives"]:
            item = dict(item)
            primitives.append(_OPS[item.pop("op")](**item))
        magnitude = record["magnitude"] or {"value": None, "unit": ""}
        severity = record["severity"]
        fault = object.__new__(_CLASSES[record["class"]])
        # the record already holds the fields, whatever arguments the class takes
        Fault.__init__(
            fault,
            fault_type=record["fault_type"],
            primitives=tuple(primitives),
            fault_id=record["fault_id"],
            magnitude=magnitude["value"],
            unit=magnitude["unit"],
            severity=None if severity is None else FaultSeverity.from_metadata(severity),
            model_parameters=dict(record["model_parameters"]),
            nominal_state=record["nominal_state"],
            fault_state=record["fault_state"],
            tags=dict(record["tags"]),
        )
        return fault
