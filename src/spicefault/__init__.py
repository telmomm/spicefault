"""Reproducible SPICE-based fault injection and reliability assessment of electronic circuits."""

__version__ = "0.1.0"

from .circuit import Circuit, Component  # noqa: E402
from .conditions import OperatingCondition  # noqa: E402
from .dataset import Dataset, split_by_magnitude, split_by_replica  # noqa: E402
from .experiments import Experiment, ExperimentResult, FaultCampaign  # noqa: E402
from .faults import Fault  # noqa: E402
from .measurements import Measurement, Waveform  # noqa: E402
from .simulation import (  # noqa: E402
    SimulationConfig,
    SimulationResult,
    SimulationStatus,
    Simulator,
)
from .variation import VariationSet  # noqa: E402

__all__ = [
    "Circuit",
    "Component",
    "Dataset",
    "Experiment",
    "ExperimentResult",
    "Fault",
    "FaultCampaign",
    "Measurement",
    "OperatingCondition",
    "SimulationConfig",
    "SimulationResult",
    "SimulationStatus",
    "Simulator",
    "split_by_magnitude",
    "split_by_replica",
    "VariationSet",
    "Waveform",
    "__version__",
]
