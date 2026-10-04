"""Severity of a fault: a dimensionless number on a declared scale.

There is no universal formula. Each scale below is one definition, named in the
record of the fault together with its parameters, and severities on different
scales are not comparable (docs/FAULT_MODEL.md, section 5).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class FaultSeverity:
    value: float
    scale: str
    parameters: dict[str, float] = field(default_factory=dict)

    @classmethod
    def declared(cls, value: float, scale: str = "declared") -> FaultSeverity:
        """A value given by the user, on a scale the user names."""
        return cls(float(value), scale)

    @classmethod
    def relative_deviation(cls, deviation: float) -> FaultSeverity:
        """s = |delta| for a parameter moved to x (1 + delta). Unbounded above."""
        return cls(abs(float(deviation)), "abs_relative_deviation")

    @classmethod
    def deviation_over_reference(
        cls, fault_value: float, nominal_value: float, reference: float
    ) -> FaultSeverity:
        """s = |x_f - x_0| / x_ref. The reference must be given when x_0 is zero."""
        if reference == 0:
            raise ValueError("the reference of a severity scale cannot be zero")
        return cls(
            abs(float(fault_value) - float(nominal_value)) / abs(float(reference)),
            "abs_deviation_over_reference",
            {"nominal_value": float(nominal_value), "reference": float(reference)},
        )

    @classmethod
    def log_resistance(cls, resistance: float, r_min: float, r_max: float) -> FaultSeverity:
        """s = log(r_max / r) / log(r_max / r_min): 0 at r_max, 1 at r_min."""
        if not 0 < r_min <= resistance <= r_max or r_min == r_max:
            raise ValueError(
                f"need 0 < r_min <= resistance <= r_max, got {r_min}, {resistance}, {r_max}"
            )
        return cls(
            math.log(r_max / resistance) / math.log(r_max / r_min),
            "log_resistance",
            {"r_min": float(r_min), "r_max": float(r_max)},
        )

    @classmethod
    def log_series_resistance(
        cls, resistance: float, r_min: float, r_max: float
    ) -> FaultSeverity:
        """s = log(r / r_min) / log(r_max / r_min): 0 at r_min, 1 at r_max."""
        if not 0 < r_min <= resistance <= r_max or r_min == r_max:
            raise ValueError(
                f"need 0 < r_min <= resistance <= r_max, got {r_min}, {resistance}, {r_max}"
            )
        return cls(
            math.log(resistance / r_min) / math.log(r_max / r_min),
            "log_series_resistance",
            {"r_min": float(r_min), "r_max": float(r_max)},
        )

    def metadata(self) -> dict:
        return {"value": self.value, "scale": self.scale, "parameters": dict(self.parameters)}

    @classmethod
    def from_metadata(cls, record: dict) -> FaultSeverity:
        return cls(record["value"], record["scale"], dict(record["parameters"]))
