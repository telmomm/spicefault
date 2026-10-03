"""Normal variation of the parameters of a circuit, kept apart from faults."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from ..netlist import Netlist
from .distributions import DISTRIBUTIONS, toleranced


class Variation(ABC):
    """Distribution of one parameter of one component in the healthy population."""

    component: str
    parameter: str

    @abstractmethod
    def sample(self, rng: np.random.Generator, nominal: float) -> float:
        """Draw the realised value; `nominal` is the value written in the netlist."""

    @abstractmethod
    def metadata(self) -> dict: ...


@dataclass(frozen=True)
class ToleranceVariation(Variation):
    """Value within nominal * (1 +- tolerance): `uniform`, or `truncnorm` with sigma = tol / 3."""

    component: str
    tolerance: float
    distribution: str = "uniform"
    parameter: str = "value"

    def __post_init__(self):
        if self.distribution not in DISTRIBUTIONS:
            raise ValueError(f"unknown tolerance distribution: {self.distribution}")

    def sample(self, rng: np.random.Generator, nominal: float) -> float:
        return toleranced(nominal, self.tolerance, rng, self.distribution)

    def metadata(self) -> dict:
        return {
            "type": "tolerance",
            "component": self.component,
            "parameter": self.parameter,
            "tolerance": self.tolerance,
            "distribution": self.distribution,
        }


class VariationSet:
    """Variations drawn in a fixed order, so that a random stream defines a circuit."""

    def __init__(self, variations: list[Variation] | tuple[Variation, ...] = ()):
        self.variations = tuple(variations)
        keys = [(v.component.lower(), v.parameter.lower()) for v in self.variations]
        if len(keys) != len(set(keys)):
            raise ValueError("a parameter has more than one variation")

    def __len__(self) -> int:
        return len(self.variations)

    def sample(self, rng: np.random.Generator, netlist: Netlist) -> dict[tuple[str, str], float]:
        """(component, parameter) -> realised value, reading the nominal ones from `netlist`."""
        return {
            (v.component, v.parameter): v.sample(rng, netlist.value(v.component, v.parameter))
            for v in self.variations
        }

    @staticmethod
    def apply(netlist: Netlist, values: dict[tuple[str, str], float]) -> None:
        for (component, parameter), value in values.items():
            netlist.set_parameter(component, parameter, "absolute", value)

    def metadata(self) -> list[dict]:
        return [v.metadata() for v in self.variations]
