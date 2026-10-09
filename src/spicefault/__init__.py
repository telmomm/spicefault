"""Reproducible SPICE-based fault injection and reliability assessment of electronic circuits."""

__version__ = "0.4.0"

from .circuit import Circuit, Component  # noqa: E402
from .conditions import OperatingCondition  # noqa: E402
from .dataset import Dataset, Specification, split_by_magnitude, split_by_replica  # noqa: E402
from .experiments import (  # noqa: E402
    Campaign,
    Design,
    Experiment,
    ExperimentResult,
    FaultCampaign,
    corners,
)
from .faults import Fault  # noqa: E402
from .measurements import Instrument, Measurement, Reading, Waveform  # noqa: E402
from .simulation import (  # noqa: E402
    SimulationConfig,
    SimulationResult,
    SimulationStatus,
    Simulator,
)
from .variation import VariationSet  # noqa: E402

__all__ = [
    "Campaign",
    "Circuit",
    "Component",
    "Dataset",
    "Design",
    "Experiment",
    "ExperimentResult",
    "Fault",
    "FaultCampaign",
    "Instrument",
    "Measurement",
    "OperatingCondition",
    "Reading",
    "SimulationConfig",
    "SimulationResult",
    "SimulationStatus",
    "Simulator",
    "Specification",
    "corners",
    "split_by_magnitude",
    "split_by_replica",
    "VariationSet",
    "Waveform",
    "__version__",
]
