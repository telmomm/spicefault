"""Fault model: a fault is a list of primitive netlist transformations with its metadata."""

from .base import Fault, InsertParallel, InsertSeries, Primitive, SetParameter
from .faultset import FaultSet
from .severity import FaultSeverity
from .types import (
    CompositeFault,
    LeakageFault,
    OpenCircuit,
    ParametricFault,
    SeriesResistanceFault,
    ShortCircuit,
)
from .universe import (
    FaultRule,
    FaultUniverse,
    leakage_rule,
    open_rule,
    parametric_rule,
    series_rule,
    short_rule,
)

__all__ = [
    "CompositeFault",
    "Fault",
    "FaultRule",
    "FaultSet",
    "FaultSeverity",
    "FaultUniverse",
    "InsertParallel",
    "InsertSeries",
    "LeakageFault",
    "OpenCircuit",
    "ParametricFault",
    "Primitive",
    "SetParameter",
    "SeriesResistanceFault",
    "ShortCircuit",
    "leakage_rule",
    "open_rule",
    "parametric_rule",
    "short_rule",
    "series_rule",
]
