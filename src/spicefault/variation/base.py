"""Normal variation of the parameters of a circuit, kept apart from faults."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from ..netlist import Netlist

Target = tuple[str, str]  # (component, parameter)


@dataclass(frozen=True)
class Draw:
    """One realisation of the healthy circuit.

    `values` are written to the netlist. `labels` are only recorded: quantities that
    were drawn but are not netlist parameters, such as the type of a part.
    """

    values: dict[Target, float] = field(default_factory=dict)
    labels: dict[str, object] = field(default_factory=dict)


class Variation(ABC):
    """Distribution of one parameter of one component in the healthy population."""

    component: str
    parameter: str

    def targets(self) -> tuple[Target, ...]:
        return ((self.component, self.parameter),)

    @abstractmethod
    def sample(self, rng: np.random.Generator, nominal: float) -> float:
        """Draw the realised value; `nominal` is the value written in the netlist."""

    def draw(self, rng: np.random.Generator, netlist: Netlist) -> Draw:
        try:
            nominal = netlist.value(self.component, self.parameter)
        except ValueError:  # an expression, not a number
            nominal = float("nan")
        return Draw({(self.component, self.parameter): self.sample(rng, nominal)})

    @abstractmethod
    def scaled(self, factor: float) -> Variation:
        """The same variation with its spread multiplied by `factor`.

        It must use as many random numbers as the original, so that scaling the
        spread of one parameter leaves the draws of the others unchanged.
        """

    @abstractmethod
    def metadata(self) -> dict: ...

    def _where(self) -> dict:
        return {"component": self.component, "parameter": self.parameter}


class VariationSet:
    """Variations drawn in a fixed order, so that a random stream defines a circuit."""

    def __init__(self, variations: Sequence[Variation] = ()):
        self.variations = tuple(variations)
        seen: set[Target] = set()
        for variation in self.variations:
            for component, parameter in variation.targets():
                key = (component.lower(), parameter.lower())
                if key in seen:
                    raise ValueError(f"{component}.{parameter} has more than one variation")
                seen.add(key)

    def __len__(self) -> int:
        return len(self.variations)

    def __iter__(self):
        return iter(self.variations)

    def __add__(self, other: Sequence[Variation]) -> VariationSet:
        return VariationSet([*self.variations, *other])

    def sample(self, rng: np.random.Generator, netlist: Netlist) -> Draw:
        """Draw every variation in order, reading the nominal values from `netlist`."""
        values: dict[Target, float] = {}
        labels: dict[str, object] = {}
        for variation in self.variations:
            draw = variation.draw(rng, netlist)
            values.update(draw.values)
            labels.update(draw.labels)
        return Draw(values, labels)

    @staticmethod
    def apply(netlist: Netlist, values: dict[Target, float]) -> None:
        for (component, parameter), value in values.items():
            netlist.set_parameter(component, parameter, "absolute", value)

    def scaled(self, factor: float, strict: bool = True) -> VariationSet:
        """Every spread multiplied by `factor`; the draws stay aligned with the original.

        Custom and joint variations cannot be scaled: with `strict` they raise,
        otherwise they are kept as they are.
        """
        scaled = []
        for variation in self.variations:
            try:
                scaled.append(variation.scaled(factor))
            except NotImplementedError:
                if strict:
                    raise
                scaled.append(variation)
        return VariationSet(scaled)

    def metadata(self) -> list[dict]:
        return [v.metadata() for v in self.variations]
