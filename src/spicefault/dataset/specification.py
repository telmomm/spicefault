"""Specifications: the limits a circuit must meet, kept apart from what was simulated."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Specification:
    """A limit on one column of a dataset, usually a measurement: `minimum <= value <=
    maximum`, with either bound left open.

    A value that is not finite does not meet the specification: a quantity that could
    not be computed is not known to be within its limits.
    """

    name: str
    minimum: float | None = None
    maximum: float | None = None

    def __post_init__(self):
        if self.minimum is None and self.maximum is None:
            raise ValueError(f"the specification of {self.name!r} needs a minimum or a maximum")
        if None not in (self.minimum, self.maximum) and self.minimum > self.maximum:
            raise ValueError(f"the minimum of the specification {self.name!r} is above its maximum")

    @property
    def column(self) -> str:
        """The label column that says whether each sample meets it."""
        return f"ok_{self.name}"

    def met(self, values) -> np.ndarray:
        """True where a value is finite and within the limits."""
        values = np.asarray(values, dtype=float)
        met = np.isfinite(values)
        with np.errstate(invalid="ignore"):
            if self.minimum is not None:
                met &= values >= self.minimum
            if self.maximum is not None:
                met &= values <= self.maximum
        return met

    def metadata(self) -> dict:
        return {"name": self.name, "minimum": self.minimum, "maximum": self.maximum}

    @classmethod
    def from_metadata(cls, record: dict) -> Specification:
        return cls(record["name"], record["minimum"], record["maximum"])
