"""The distributions a parameter can follow in the healthy population.

Every draw takes its random numbers from the generator it is given and from nothing
else, in a fixed number and order (the truncated ones redraw until in range).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import numpy as np

from ..circuit import Circuit
from ..netlist import Netlist
from .base import Draw, Target, Variation, VariationSet
from .distributions import DISTRIBUTIONS, log_uniform_factor, unit_deviation


def _callable_name(function) -> str:
    """Name of a function, or of the class of a callable object."""
    return getattr(function, "__qualname__", type(function).__qualname__)


def _centre(given: float | None, nominal: float) -> float:
    return nominal if given is None else given


@dataclass(frozen=True)
class FixedVariation(Variation):
    """No spread: the parameter is set to `value` in every sample."""

    component: str
    value: float
    parameter: str = "value"

    def sample(self, rng, nominal):
        return float(self.value)

    def scaled(self, factor):
        return self

    def metadata(self):
        return {"type": "fixed", **self._where(), "value": self.value}


@dataclass(frozen=True)
class ToleranceVariation(Variation):
    """Value within a tolerance band around the nominal value of the netlist.

    Relative (default): nominal * (1 +- tolerance), as for the value of a resistor.
    With `relative=False`: nominal +- tolerance, in the units of the parameter, as for
    an offset voltage that is nominally zero. `distribution` is `uniform`, or
    `truncnorm`: normal with sigma = tolerance / 3, truncated at the band.
    """

    component: str
    tolerance: float
    distribution: str = "uniform"
    parameter: str = "value"
    relative: bool = True

    def __post_init__(self):
        if self.distribution not in DISTRIBUTIONS:
            raise ValueError(f"unknown tolerance distribution: {self.distribution}")

    def sample(self, rng, nominal):
        deviation = self.tolerance * unit_deviation(rng, self.distribution)
        return nominal * (1.0 + deviation) if self.relative else nominal + deviation

    def scaled(self, factor):
        return replace(self, tolerance=self.tolerance * factor)

    def metadata(self):
        return {
            "type": "tolerance",
            **self._where(),
            "tolerance": self.tolerance,
            "distribution": self.distribution,
            "relative": self.relative,
        }


@dataclass(frozen=True)
class UniformVariation(Variation):
    """Uniform between `low` and `high`."""

    component: str
    low: float
    high: float
    parameter: str = "value"

    def __post_init__(self):
        if not self.low <= self.high:
            raise ValueError(f"need low <= high, got {self.low}, {self.high}")

    def sample(self, rng, nominal):
        return float(rng.uniform(self.low, self.high))

    def scaled(self, factor):
        middle, half = 0.5 * (self.low + self.high), 0.5 * (self.high - self.low)
        return replace(self, low=middle - factor * half, high=middle + factor * half)

    def metadata(self):
        return {"type": "uniform", **self._where(), "low": self.low, "high": self.high}


@dataclass(frozen=True)
class NormalVariation(Variation):
    """Normal with standard deviation `sigma` around `mean` (the nominal value if None).

    A normal distribution has no bounds, so a physical parameter can come out
    negative. `truncate` limits the draw to mean +- truncate * sigma, by redrawing.
    """

    component: str
    mean: float | None
    sigma: float
    parameter: str = "value"
    truncate: float | None = None

    def __post_init__(self):
        if self.sigma < 0 or (self.truncate is not None and self.truncate <= 0):
            raise ValueError("sigma must not be negative and truncate must be positive")

    def sample(self, rng, nominal):
        while True:
            z = rng.normal(0.0, 1.0)
            if self.truncate is None or abs(z) <= self.truncate:
                return float(_centre(self.mean, nominal) + self.sigma * z)

    def scaled(self, factor):
        return replace(self, sigma=self.sigma * factor)

    def metadata(self):
        return {
            "type": "normal",
            **self._where(),
            "mean": self.mean,
            "sigma": self.sigma,
            "truncate": self.truncate,
        }


@dataclass(frozen=True)
class LogNormalVariation(Variation):
    """median * exp(N(0, sigma_log)): always positive, with a multiplicative spread.

    `median` defaults to the nominal value. `sigma_log` is the standard deviation of
    the natural logarithm; for small values it is close to the relative spread.
    """

    component: str
    median: float | None
    sigma_log: float
    parameter: str = "value"

    def __post_init__(self):
        if self.sigma_log < 0:
            raise ValueError("sigma_log must not be negative")

    def sample(self, rng, nominal):
        return float(_centre(self.median, nominal) * np.exp(self.sigma_log * rng.normal(0.0, 1.0)))

    def scaled(self, factor):
        return replace(self, sigma_log=self.sigma_log * factor)

    def metadata(self):
        return {
            "type": "lognormal",
            **self._where(),
            "median": self.median,
            "sigma_log": self.sigma_log,
        }


@dataclass(frozen=True)
class LogUniformVariation(Variation):
    """median times a factor drawn log-uniformly between 1 / spread and spread.

    For parameters known to within a factor, not a percentage.
    """

    component: str
    spread: float
    median: float | None = None
    parameter: str = "value"

    def __post_init__(self):
        if self.spread < 1:
            raise ValueError("spread is a factor of at least 1")

    def sample(self, rng, nominal):
        return _centre(self.median, nominal) * log_uniform_factor(rng, self.spread)

    def scaled(self, factor):
        return replace(self, spread=self.spread**factor)

    def metadata(self):
        return {"type": "loguniform", **self._where(), "spread": self.spread, "median": self.median}


@dataclass(frozen=True)
class CustomVariation(Variation):
    """Any distribution of one parameter: `sampler(rng, nominal)` returns the value.

    The function cannot be stored in a record; `description` is what the provenance
    keeps, together with the name of the function.
    """

    component: str
    sampler: Callable[[np.random.Generator, float], float]
    parameter: str = "value"
    description: str = ""

    def sample(self, rng, nominal):
        return float(self.sampler(rng, nominal))

    def scaled(self, factor):
        raise NotImplementedError(f"custom variation of {self.component} cannot be scaled")

    def metadata(self):
        return {
            "type": "custom",
            **self._where(),
            "sampler": _callable_name(self.sampler),
            "description": self.description,
        }


@dataclass(frozen=True)
class JointVariation(Variation):
    """Several parameters drawn together, when they are not independent.

    For example the parameters of a part whose type is drawn first, or a netlist
    parameter computed from two drawn quantities. `sampler(rng, netlist)` returns the
    values of `parameters`, as {(component, parameter): value}, or a `Draw` to also
    record labels. `name` identifies it in the provenance.
    """

    name: str
    parameters: tuple[Target, ...]
    sampler: Callable[[np.random.Generator, Netlist], Draw | dict]
    description: str = ""

    def __post_init__(self):
        object.__setattr__(self, "parameters", tuple(tuple(t) for t in self.parameters))

    def targets(self):
        return self.parameters

    def sample(self, rng, nominal):
        raise TypeError("a joint variation is drawn as a whole")

    def draw(self, rng, netlist):
        drawn = self.sampler(rng, netlist)
        if not isinstance(drawn, Draw):
            drawn = Draw(dict(drawn))
        if set(drawn.values) != set(self.parameters):
            raise ValueError(f"joint variation {self.name!r} must return its declared parameters")
        return drawn

    def scaled(self, factor):
        raise NotImplementedError(f"joint variation {self.name!r} cannot be scaled")

    def metadata(self):
        return {
            "type": "joint",
            "name": self.name,
            "parameters": [{"component": c, "parameter": p} for c, p in self.parameters],
            "sampler": _callable_name(self.sampler),
            "description": self.description,
        }


def tolerances(
    circuit: Circuit,
    by_kind: dict[str, float],
    distribution: str = "uniform",
    overrides: dict[str, float] | None = None,
    components: Sequence[str] | None = None,
) -> VariationSet:
    """Relative tolerances for the values of a circuit, in netlist order.

    `by_kind` maps element letters to tolerances, e.g. `{"R": 0.01, "C": 0.05}`;
    `overrides` gives the tolerance of individual components; `components` restricts
    the set to those named.
    """
    overrides = {name.lower(): tol for name, tol in (overrides or {}).items()}
    wanted = None if components is None else {name.lower() for name in components}
    found = []
    for component in circuit.components():
        name = component.name.lower()
        if wanted is not None and name not in wanted:
            continue
        tolerance = overrides.get(name, by_kind.get(component.kind))
        if tolerance is not None and "value" in component.parameters:
            found.append(ToleranceVariation(component.name, tolerance, distribution))
    return VariationSet(found)
