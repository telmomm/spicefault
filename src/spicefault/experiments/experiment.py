"""The experiment: a circuit, its normal variation, faults and operating conditions,
simulated as one deterministic set of samples.
"""

from __future__ import annotations

import pickle
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import pandas as pd

from .. import __version__
from ..circuit import Circuit
from ..conditions import OperatingCondition
from ..faults import Fault, FaultSet
from ..measurements import Measurement, Waveform
from ..simulation import SimulationConfig, SimulationResult, Simulator
from ..variation import VariationSet
from .seeding import SCHEMES, sample_stream, seed_key


@dataclass(frozen=True)
class Sample:
    """Definition of one simulation. `fault_index` 0 is the fault-free circuit."""

    sample_id: int
    fault_index: int
    replica: int
    condition_index: int


@dataclass(frozen=True)
class Realisation:
    """A sample turned into a netlist, with the values drawn for it."""

    netlist: str
    parameters: dict[tuple[str, str], float]
    labels: dict[str, object]
    seed_key: tuple[int, ...]


@dataclass(frozen=True)
class SampleResult:
    """A simulated sample. Its draws come from `sample_stream(seed, *seed_key)`."""

    sample: Sample
    fault_id: str
    condition: str
    seed_key: tuple[int, ...]
    parameters: dict[tuple[str, str], float]
    labels: dict[str, object]
    result: SimulationResult
    measurements: dict[str, float] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    metadata: dict
    samples: list[SampleResult]

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)

    def status_counts(self) -> dict[str, int]:
        return dict(Counter(s.result.status.value for s in self.samples))

    def to_frame(self) -> pd.DataFrame:
        """One row per sample: its definition, its status and the realised parameters."""
        rows = []
        for s in self.samples:
            row = {
                "sample_id": s.sample.sample_id,
                "fault_index": s.sample.fault_index,
                "fault_id": s.fault_id,
                "replica": s.sample.replica,
                "condition": s.condition,
                "seed_key": "/".join(map(str, s.seed_key)),
                "status": s.result.status.value,
                "message": s.result.message,
                **s.labels,
            }
            row.update({f"p_{c}_{p}": value for (c, p), value in s.parameters.items()})
            row.update(s.measurements)
            rows.append(row)
        return pd.DataFrame(rows)


HEALTHY_ID = "healthy"


@dataclass
class Experiment:
    """What is simulated, under which conditions, with which faults and which uncertainty.

    Each fault is simulated `samples` times and the fault-free circuit
    `healthy_samples` times (`samples` by default), under every operating condition.
    The values of a sample are drawn from a stream that depends only on the seed and
    on the key of the sample, which `seeding` defines (see `seeding.seed_key`): by
    default (fault index, replica). The result therefore does not depend on the number
    of workers. The operating condition is never part of the key, so one drawn circuit
    is observed under all the conditions.
    """

    circuit: Circuit
    simulator: Simulator = field(default_factory=Simulator)
    config: SimulationConfig = field(default_factory=SimulationConfig)
    faults: Sequence[Fault] = ()
    variations: VariationSet = field(default_factory=VariationSet)
    conditions: Sequence[OperatingCondition] = (OperatingCondition(),)
    samples: int = 1
    healthy_samples: int | None = None
    seed: int = 0
    seeding: str = "positional"
    measurements: Sequence[Measurement] = ()
    waveform: Waveform | None = None

    def __post_init__(self):
        if self.seeding not in SCHEMES:
            raise ValueError(f"unknown seeding scheme {self.seeding!r}; expected one of {SCHEMES}")
        if isinstance(self.conditions, OperatingCondition):
            self.conditions = (self.conditions,)
        if not isinstance(self.variations, VariationSet):
            self.variations = VariationSet(self.variations)
        self.faults, self.conditions = tuple(self.faults), tuple(self.conditions)
        self.measurements = tuple(self.measurements)
        if not self.conditions:
            raise ValueError("an experiment needs at least one operating condition")
        for label, names in (
            ("fault", [HEALTHY_ID, *(f.fault_id for f in self.faults)]),
            ("operating condition", [c.name for c in self.conditions]),
            ("measurement", [m.name for m in self.measurements]),
        ):
            repeated = [name for name, n in Counter(names).items() if n > 1]
            if repeated:
                raise ValueError(f"repeated {label} identifiers: {repeated}")

    # --- definition -------------------------------------------------------------------

    def fault(self, sample: Sample) -> Fault | None:
        return self.faults[sample.fault_index - 1] if sample.fault_index else None

    def fault_id(self, sample: Sample) -> str:
        fault = self.fault(sample)
        return fault.fault_id if fault else HEALTHY_ID

    def seed_key(self, sample: Sample) -> tuple[int, ...]:
        return seed_key(self.seeding, sample.fault_index, self.fault_id(sample), sample.replica)

    def plan(self) -> list[Sample]:
        """Every sample, in a fixed order: by fault (healthy first), replica and condition."""
        n_healthy = self.samples if self.healthy_samples is None else self.healthy_samples
        plan = []
        for fault_index in range(len(self.faults) + 1):
            for replica in range(n_healthy if fault_index == 0 else self.samples):
                for condition_index in range(len(self.conditions)):
                    plan.append(Sample(len(plan), fault_index, replica, condition_index))
        return plan

    def realise(self, sample: Sample) -> Realisation:
        """Draw the circuit, inject the fault, then set the operating condition."""
        netlist = self.circuit.netlist()
        key = self.seed_key(sample)
        draw = self.variations.sample(sample_stream(self.seed, *key), netlist)
        self.variations.apply(netlist, draw.values)
        fault = self.fault(sample)
        if fault is not None:
            fault.apply(netlist)
        self.conditions[sample.condition_index].apply(netlist)
        return Realisation(str(netlist), draw.values, draw.labels, key)

    def faults_overlapping_tolerance(self) -> int:
        """How many parametric fault conditions lie partly inside the tolerance band."""
        report = FaultSet(self.faults).tolerance_overlap(self.variations, self.circuit)
        return int((report["inside_fraction"] > 0).sum())

    def metadata(self) -> dict:
        """Everything needed to regenerate the samples, as JSON-serialisable data."""
        n_healthy = self.samples if self.healthy_samples is None else self.healthy_samples
        return {
            "spicefault_version": __version__,
            **self.simulator.metadata(),
            **self.circuit.metadata(),
            "seed": self.seed,
            "seeding": self.seeding,
            "samples": self.samples,
            "healthy_samples": n_healthy,
            "simulation": self.config.metadata(),
            "faults": [f.metadata() for f in self.faults],
            "variations": self.variations.metadata(),
            "conditions": [c.metadata() for c in self.conditions],
            "measurements": [m.metadata() for m in self.measurements],
            "waveform": None if self.waveform is None else self.waveform.metadata(),
        }

    # --- execution --------------------------------------------------------------------

    def check_picklable(self) -> None:
        """Worker processes receive a copy of the experiment, so it must be picklable."""
        try:
            pickle.dumps(self)
        except Exception as exc:
            raise TypeError(
                "the experiment cannot be sent to worker processes. Functions given to "
                "custom or joint variations, custom measurements and custom backends must "
                "be defined at module level (not lambdas or functions defined inside another "
                f"function), or run with workers=1. Cause: {type(exc).__name__}: {exc}"
            ) from exc

    def measure(self, result: SimulationResult) -> dict[str, float]:
        """The measurements of a successful simulation, by name."""
        return {m.name: m(result) for m in self.measurements}

    def run_sample(self, sample: Sample) -> SampleResult:
        realised = self.realise(sample)
        result = self.simulator.run(realised.netlist, self.config)
        return SampleResult(
            sample=sample,
            fault_id=self.fault_id(sample),
            condition=self.conditions[sample.condition_index].name,
            seed_key=realised.seed_key,
            parameters=realised.parameters,
            labels=realised.labels,
            result=result,
            measurements=self.measure(result) if result.ok else {},
        )

    def run(self, workers: int = 1) -> ExperimentResult:
        """Simulate every sample and keep the results in memory.

        A simulation that fails is recorded with its status; it does not stop the run.
        """
        plan = self.plan()
        if workers == 1:
            results = [self.run_sample(sample) for sample in plan]
        else:
            self.check_picklable()
            with ProcessPoolExecutor(
                max_workers=workers, initializer=_init_worker, initargs=(self,)
            ) as pool:
                results = list(pool.map(_run_sample, plan, chunksize=16))
        return ExperimentResult(self.metadata(), results)


_EXPERIMENT: Experiment | None = None


def _init_worker(experiment: Experiment) -> None:
    global _EXPERIMENT
    _EXPERIMENT = experiment


def _run_sample(sample: Sample) -> SampleResult:
    return _EXPERIMENT.run_sample(sample)
