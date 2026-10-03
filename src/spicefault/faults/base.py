"""Primitives and the fault built from them (docs/FAULT_MODEL.md, sections 1 and 7)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from ..netlist import RULES, Netlist

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


@dataclass(frozen=True)
class Fault:
    """One fault condition: what is injected, and what is recorded about it.

    `magnitude` is the physical size of the fault, with `unit`; `severity` is an
    optional dimensionless number on the declared `severity_scale`. `model_parameters`
    holds values that belong to the model and not to the defect, such as the
    resistance that stands for an open circuit. `tags` are application labels.
    """

    fault_type: str
    primitives: tuple[Primitive, ...]
    fault_id: str = ""
    magnitude: float | None = None
    unit: str = ""
    severity: float | None = None
    severity_scale: str = ""
    model_parameters: dict[str, float] = field(default_factory=dict)
    nominal_state: str = ""
    fault_state: str = ""
    tags: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "primitives", tuple(self.primitives))
        if not self.primitives:
            raise ValueError("a fault needs at least one primitive")
        if not self.fault_id:
            size = "" if self.magnitude is None else f":{self.magnitude:+g}"
            object.__setattr__(
                self, "fault_id", f"{'+'.join(self.components)}:{self.fault_type}{size}"
            )

    @property
    def components(self) -> tuple[str, ...]:
        """Components the fault acts on, in order of first appearance."""
        return tuple(dict.fromkeys(p.component for p in self.primitives))

    def apply(self, netlist: Netlist) -> None:
        """Inject the fault into an already realised netlist, primitive by primitive."""
        for primitive in self.primitives:
            primitive.apply(netlist)

    def metadata(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "fault_id": self.fault_id,
            "fault_type": self.fault_type,
            "components": list(self.components),
            "primitives": [{"op": _OP_NAMES[type(p)], **asdict(p)} for p in self.primitives],
            "magnitude": None
            if self.magnitude is None
            else {"value": self.magnitude, "unit": self.unit},
            "severity": self.severity,
            "severity_scale": self.severity_scale,
            "model_parameters": dict(self.model_parameters),
            "nominal_state": self.nominal_state,
            "fault_state": self.fault_state,
            "tags": dict(self.tags),
        }

    @classmethod
    def from_metadata(cls, record: dict) -> Fault:
        if record["schema_version"] != SCHEMA_VERSION:
            raise ValueError(f"unsupported fault schema version {record['schema_version']}")
        primitives = []
        for item in record["primitives"]:
            item = dict(item)
            primitives.append(_OPS[item.pop("op")](**item))
        magnitude = record["magnitude"] or {"value": None, "unit": ""}
        return cls(
            fault_type=record["fault_type"],
            primitives=tuple(primitives),
            fault_id=record["fault_id"],
            magnitude=magnitude["value"],
            unit=magnitude["unit"],
            severity=record["severity"],
            severity_scale=record["severity_scale"],
            model_parameters=dict(record["model_parameters"]),
            nominal_state=record["nominal_state"],
            fault_state=record["fault_state"],
            tags=dict(record["tags"]),
        )
