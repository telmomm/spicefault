"""The distributions a parameter can follow in the healthy population.

Every draw takes its random numbers from the generator it is given and from nothing
else, in a fixed number and order (the truncated ones redraw until in range).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
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


@dataclass(frozen=True)
class CatalogueVariation(Variation):
    """A part that is one of several types: the type is drawn first, then its parameters.

    `options` maps each type to the medians of the parameters of `component`, for
    example `{"gel": {"r": 2e3, "c": 50e-9}, "steel": {"r": 2e5, "c": 5e-9}}`; every
    type gives the same parameters. The type is drawn with probabilities proportional
    to `weights` (equal by default), and each parameter is its median times a factor
    drawn log-uniformly between 1 / `spread` and `spread`, as in
    `LogUniformVariation`. The drawn type is recorded under `label`, by default
    `<component>_kind`, and becomes a column of the dataset.

    It uses the same random numbers whatever type is drawn, and it is rebuilt from
    its record, since it holds no function.
    """

    component: str
    options: Mapping[str, Mapping[str, float]]
    weights: Sequence[float] | None = None
    spread: float = 1.0
    label: str = ""

    def __post_init__(self):
        options = {str(kind): dict(medians) for kind, medians in self.options.items()}
        if not options:
            raise ValueError("a catalogue needs at least one type")
        parameters = [tuple(medians) for medians in options.values()]
        if not parameters[0] or any(set(p) != set(parameters[0]) for p in parameters):
            raise ValueError("every type of a catalogue gives the same parameters")
        if self.spread < 1:
            raise ValueError("spread is a factor of at least 1")
        if self.weights is not None:
            weights = [float(w) for w in self.weights]
            if len(weights) != len(options) or min(weights) < 0 or not sum(weights) > 0:
                raise ValueError("weights: one per type, not negative, not all zero")
            object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "options", options)
        object.__setattr__(self, "label", self.label or f"{self.component}_kind")

    @property
    def parameters(self) -> tuple[str, ...]:
        return tuple(next(iter(self.options.values())))

    def targets(self):
        return tuple((self.component, parameter) for parameter in self.parameters)

    def sample(self, rng, nominal):
        raise TypeError("a catalogue variation is drawn as a whole")

    def draw(self, rng, netlist):
        kinds = list(self.options)
        weights = np.asarray(self.weights or [1.0] * len(kinds), dtype=float)
        edges = np.cumsum(weights) / weights.sum()
        index = min(int(np.searchsorted(edges, rng.random(), side="right")), len(kinds) - 1)
        medians = self.options[kinds[index]]
        values = {
            (self.component, parameter): medians[parameter] * log_uniform_factor(rng, self.spread)
            for parameter in self.parameters
        }
        return Draw(values, {self.label: kinds[index]})

    def scaled(self, factor):
        return replace(self, spread=self.spread**factor)

    def metadata(self):
        return {
            "type": "catalogue",
            "component": self.component,
            "options": {kind: dict(medians) for kind, medians in self.options.items()},
            "weights": None if self.weights is None else list(self.weights),
            "spread": self.spread,
            "label": self.label,
        }


_REBUILDABLE = {
    "catalogue": CatalogueVariation,
    "fixed": FixedVariation,
    "tolerance": ToleranceVariation,
    "uniform": UniformVariation,
    "normal": NormalVariation,
    "lognormal": LogNormalVariation,
    "loguniform": LogUniformVariation,
}


def variation_from_metadata(record: dict) -> Variation:
    """Rebuild a variation from its record. Custom and joint ones hold a function,
    which a record cannot carry.
    """
    record = dict(record)
    kind = record.pop("type")
    if kind not in _REBUILDABLE:
        name = record.get("name") or record.get("component")
        raise NotImplementedError(
            f"the {kind} variation {name!r} holds a function and cannot be rebuilt from "
            "its record; pass the variations explicitly"
        )
    return _REBUILDABLE[kind](**record)


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


def instance_tolerances(
    circuit: Circuit,
    model: str,
    parameters: Mapping[str, float | tuple[float, str]],
    distribution: str = "uniform",
) -> VariationSet:
    """The same tolerances for the parameters of every instance of a subcircuit or model.

    `model` is the subcircuit of the X instances, or the model of the devices, to
    vary (`Component.model`). `parameters` maps each instance parameter to its
    tolerance: a number for a relative one, or `(tolerance, "relative")` or
    `(tolerance, "absolute")`, the second for a parameter that is nominally zero:

        instance_tolerances(circuit, "opamp", {"vos": (0.5e-3, "absolute"), "aol": 0.5})

    The variations come in netlist order, instance by instance. A parameter must be
    written on every instance, since only what the netlist states can be varied.
    """
    kinds = {"relative": True, "absolute": False}
    wanted = []
    for parameter, tolerance in parameters.items():
        tolerance, kind = tolerance if isinstance(tolerance, tuple) else (tolerance, "relative")
        if kind not in kinds:
            raise ValueError(f"{parameter}: a tolerance is relative or absolute, not {kind!r}")
        wanted.append((parameter, float(tolerance), kinds[kind]))
    instances = [c for c in circuit.components() if c.model.lower() == model.lower()]
    if not instances:
        raise ValueError(f"the circuit has no instance of {model!r}")
    found = []
    for component in instances:
        written = {name.lower() for name in component.parameters}
        for parameter, tolerance, relative in wanted:
            if parameter.lower() not in written:
                raise ValueError(f"{component.name} does not state the parameter {parameter!r}")
            found.append(
                ToleranceVariation(component.name, tolerance, distribution, parameter, relative)
            )
    return VariationSet(found)
