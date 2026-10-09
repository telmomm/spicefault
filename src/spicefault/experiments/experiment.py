"""The experiment: a circuit, its normal variation, faults and operating conditions,
simulated as one deterministic set of samples.
"""

from __future__ import annotations

import pickle
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace

import pandas as pd

from .. import __version__
from ..circuit import Circuit
from ..conditions import OperatingCondition
from ..faults import Fault, FaultSet
from ..measurements import Measurement, Waveform
from ..simulation import SimulationConfig, SimulationResult, SimulationStatus, Simulator
from ..variation import Variation, VariationSet
from . import sampling as designs
from .design import Design
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

    `sampling` chooses how the circuits of a population are drawn: `random`, each from
    its own independent stream; `lhs`, a Latin hypercube; or `sobol`, a scrambled Sobol
    sequence (see `experiments.sampling`). The last two decide the points of each
    population jointly, through the quantile function of every variation, and their
    samples are not independent.

    With a `design`, the circuits are not drawn: each row of the design is a circuit
    with the parameter values it gives, simulated without and with each fault, under
    every condition. There are then no variations and no sample counts to give.
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
    design: Design | None = None
    sampling: str = "random"

    def __post_init__(self):
        self._designs: dict[tuple, object] = {}
        if self.sampling not in designs.METHODS:
            raise ValueError(
                f"unknown sampling method {self.sampling!r}; expected one of {designs.METHODS}"
            )
        if self.config == SimulationConfig() and self.circuit.imported_analyses:
            self.config = SimulationConfig(analyses=self.circuit.imported_analyses)
        if self.seeding not in SCHEMES:
            raise ValueError(f"unknown seeding scheme {self.seeding!r}; expected one of {SCHEMES}")
        if isinstance(self.conditions, OperatingCondition):
            self.conditions = (self.conditions,)
        if not isinstance(self.variations, VariationSet):
            self.variations = VariationSet(self.variations)
        self.faults, self.conditions = tuple(self.faults), tuple(self.conditions)
        self.measurements = tuple(self.measurements)
        if self.design is not None:
            if len(self.variations):
                raise ValueError(
                    "a design gives the parameter values: it cannot be combined with variations"
                )
            if self.samples not in (1, len(self.design)) or self.healthy_samples not in (
                None, len(self.design),
            ):  # fmt: skip
                raise ValueError("a design has one sample per row: do not give sample counts")
            self.samples, self.healthy_samples = len(self.design), None
        if self.sampling != "random":
            if self.design is not None:
                raise ValueError("a design gives the parameter values: it is not sampled")
            for variation in self.variations:
                if type(variation).quantile is Variation.quantile:
                    raise ValueError(
                        f"{self.sampling} sampling turns uniform numbers into values through the "
                        f"quantile function of each variation, and the "
                        f"{type(variation).__name__} of "
                        f"{getattr(variation, 'component', None) or variation.name} has none"
                    )
            if self.sampling == "sobol":
                designs.record("sobol")  # fails here, not in a worker, if SciPy is missing
        if not self.conditions:
            raise ValueError("an experiment needs at least one operating condition")
        for label, names in (
            ("fault", [HEALTHY_ID, *(f.fault_id for f in self.faults)]),
            ("operating condition", [c.name for c in self.conditions]),
            ("measurement", [name for m in self.measurements for name in m.columns]),
            *(
                (
                    f"measurement of condition {c.name}",
                    [name for m in c.measurements for name in m.columns],
                )
                for c in self.conditions
                if c.measurements is not None
            ),
        ):
            repeated = [name for name, n in Counter(names).items() if n > 1]
            if repeated:
                raise ValueError(f"repeated {label} identifiers: {repeated}")
        points = {w.n_points for w in map(self.waveform_for, range(len(self.conditions))) if w}
        if len(points) > 1:
            raise ValueError(
                "the waveforms of the operating conditions are stored in one array and must "
                f"have the same number of points, not {sorted(points)}"
            )

    # --- definition -------------------------------------------------------------------

    def fault(self, sample: Sample) -> Fault | None:
        return self.faults[sample.fault_index - 1] if sample.fault_index else None

    def fault_id(self, sample: Sample) -> str:
        fault = self.fault(sample)
        return fault.fault_id if fault else HEALTHY_ID

    def seed_key(self, sample: Sample) -> tuple[int, ...]:
        return seed_key(self.seeding, sample.fault_index, self.fault_id(sample), sample.replica)

    def config_for(self, condition_index: int) -> SimulationConfig:
        condition = self.conditions[condition_index]
        return condition.config or self.config

    def measurements_for(self, condition_index: int) -> tuple[Measurement, ...]:
        condition = self.conditions[condition_index]
        return condition.measurements if condition.measurements is not None else self.measurements

    def waveform_for(self, condition_index: int) -> Waveform | None:
        """The waveform stored for a condition: its own, or that of the experiment."""
        own = self.conditions[condition_index].waveform
        return self.waveform if own is None else own or None

    @property
    def waveform_points(self) -> int | None:
        """Points of the stored waveforms; None if no condition stores one."""
        for index in range(len(self.conditions)):
            waveform = self.waveform_for(index)
            if waveform is not None:
                return waveform.n_points
        return None

    @property
    def measurement_columns(self) -> tuple[str, ...]:
        """The measurement columns of the table: those of every condition, once each."""
        return tuple(dict.fromkeys(
            column
            for index in range(len(self.conditions))
            for column in self.columns_for(index)
        ))

    def columns_for(self, condition_index: int) -> tuple[str, ...]:
        """The measurement columns that a condition fills."""
        return tuple(
            column
            for measurement in self.measurements_for(condition_index)
            for column in measurement.columns
        )

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
        if self.design is not None:
            values, labels = self.design.row(sample.replica)
        elif self.sampling == "random":
            draw = self.variations.sample(sample_stream(self.seed, *key), netlist)
            values, labels = draw.values, draw.labels
        else:
            row, labels = self._points(sample)[sample.replica], {}
            values = {
                variation.targets()[0]: variation.quantile(float(u), variation.nominal(netlist))
                for variation, u in zip(self.variations, row, strict=True)
            }
        self.variations.apply(netlist, values)
        fault = self.fault(sample)
        if fault is not None:
            fault.apply(netlist)
        self.conditions[sample.condition_index].apply(netlist)
        return Realisation(str(netlist), values, labels, key)

    def _points(self, sample: Sample):
        """The design of the population of a sample: one row per replica, one column per
        variation. A population is the fault-free circuits, or those with one fault.
        """
        population = self.seed_key(sample)[:-1]
        size = self.samples
        if sample.fault_index == 0 and self.healthy_samples is not None:
            size = self.healthy_samples
        found = self._designs.get((population, size))
        if found is None:
            found = designs.uniforms(
                self.sampling, self.seed, population, size, len(self.variations)
            )
            self._designs[population, size] = found
        return found

    def parameter_targets(self) -> tuple[tuple[str, str], ...]:
        """The parameters whose value every sample records: those drawn or designed."""
        if self.design is not None:
            return tuple(self.design.targets)
        return tuple(target for variation in self.variations for target in variation.targets())

    def faults_overlapping_tolerance(self) -> int:
        """How many parametric fault conditions lie partly inside the tolerance band."""
        report = FaultSet(self.faults).tolerance_overlap(self.variations, self.circuit)
        return int((report["inside_fraction"] > 0).sum())

    def metadata(self) -> dict:
        """Everything needed to regenerate the samples, as JSON-serialisable data."""
        n_healthy = self.samples if self.healthy_samples is None else self.healthy_samples
        # only when there is one, so that the record of a drawn experiment does not change
        design = {} if self.design is None else {"design": self.design.metadata()}
        if self.sampling != "random":
            design["sampling"] = designs.record(self.sampling)
        return {
            **design,
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

    @classmethod
    def from_metadata(cls, metadata: dict, netlist: str, **overrides) -> Experiment:
        """Rebuild an experiment from its record and the netlist of its circuit.

        What a record cannot carry must be passed again as keyword arguments: custom
        or joint variations (`variations`), custom measurements (`measurements`) and
        a backend other than the built-in ones (`simulator`). The netlist must be the
        one the record was made with.
        """
        circuit = Circuit(
            netlist,
            metadata["circuit"],
            imported_analyses=metadata.get("imported_analyses", ()),
        )
        if circuit.metadata()["netlist_sha256"] != metadata["netlist_sha256"]:
            raise ValueError("the netlist is not the one this experiment was defined with")
        waveform = metadata["waveform"]
        parts = {
            "simulator": lambda: Simulator(metadata["simulator"]),
            "config": lambda: SimulationConfig.from_metadata(metadata["simulation"]),
            "faults": lambda: [Fault.from_metadata(f) for f in metadata["faults"]],
            "variations": lambda: VariationSet.from_metadata(metadata["variations"]),
            "conditions": lambda: [
                OperatingCondition.from_metadata(c) for c in metadata["conditions"]
            ],
            "measurements": lambda: [
                Measurement.from_metadata(m) for m in metadata["measurements"]
            ],
            "waveform": lambda: None if waveform is None else Waveform.from_metadata(waveform),
            "design": lambda: (
                Design.from_metadata(metadata["design"]) if metadata.get("design") else None
            ),
        }
        built = {name: overrides.get(name) or make() for name, make in parts.items()}
        return cls(
            circuit,
            samples=metadata["samples"],
            healthy_samples=None if built["design"] is not None else metadata["healthy_samples"],
            sampling=metadata.get("sampling", {}).get("method", "random"),
            seed=metadata["seed"],
            seeding=metadata["seeding"],
            **built,
        )

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

    def measure(
        self, result: SimulationResult, condition_index: int | None = None
    ) -> dict[str, float]:
        """The measurements of a successful simulation, by name."""
        measurements = self.measurements if condition_index is None else self.measurements_for(
            condition_index
        )
        values: dict[str, float] = {}
        for measurement in measurements:
            values.update(measurement.values(result))
        return values

    def _measured(
        self, result: SimulationResult, condition_index: int
    ) -> tuple[SimulationResult, dict[str, float]]:
        """The result and its measurements; a result that cannot be measured is invalid."""
        if not result.ok:
            return result, {}
        try:
            measurements = dict.fromkeys(self.measurement_columns, float("nan"))
            measurements.update(self.measure(result, condition_index))
        except Exception as exc:  # the simulation ran but its output cannot be used
            measurements = {}
            result = replace(
                result,
                status=SimulationStatus.INVALID_OUTPUT,
                message=f"measurement failed: {type(exc).__name__}: {exc}",
            )
        return result, measurements

    def nominal(self, fault: Fault | str | None = None) -> dict[str, SampleResult]:
        """The nominal circuit, simulated and measured under each operating condition.

        No variation is drawn: the circuit has the values of its netlist. It is the
        reference of an experiment, such as the nominal gain against which a gain
        error is defined. With `fault`, a fault or the identifier of one of the
        experiment, it is the nominal circuit with that fault. See `evaluate`, of
        which this is the case with no values given.
        """
        return self.evaluate({}, fault)

    def evaluate(
        self, parameters: dict[tuple[str, str], float], fault: Fault | str | None = None
    ) -> dict[str, SampleResult]:
        """The circuit with the given parameter values, simulated and measured under
        each operating condition.

        `parameters` maps (component, parameter) to a value; what is not given keeps
        the value of the netlist. Nothing is drawn. With `fault`, a fault or the
        identifier of one of the experiment, it is injected after the values are set,
        as in a sample. This is what an external algorithm needs to drive the circuit:
        an optimiser, or a sensitivity method that supplies its own points.

        Returns {condition name: SampleResult}, with the measurements of the condition
        and the complete simulation result. These are not samples of the plan:
        `sample_id` and `replica` are -1. As in a run, a simulation that fails is
        returned with its status, not raised.
        """
        if isinstance(fault, str):
            fault = FaultSet(self.faults)[fault]
        if fault is None:
            fault_index = 0
        else:
            fault_index = self.faults.index(fault) + 1 if fault in self.faults else -1
        values = {(str(c), str(p)): float(value) for (c, p), value in parameters.items()}
        results = {}
        for condition_index, condition in enumerate(self.conditions):
            netlist = self.circuit.netlist()
            for (component, parameter), value in values.items():
                try:
                    netlist.set_parameter(component, parameter, "absolute", value)
                except (KeyError, IndexError) as exc:
                    raise KeyError(
                        f"the circuit has no parameter {component}.{parameter}"
                    ) from exc
            if fault is not None:
                fault.apply(netlist)
            condition.apply(netlist)
            result = self.simulator.run(str(netlist), self.config_for(condition_index))
            result, measurements = self._measured(result, condition_index)
            own = self.columns_for(condition_index)  # not the columns of the other conditions
            measurements = {name: measurements[name] for name in own if name in measurements}
            results[condition.name] = SampleResult(
                sample=Sample(-1, fault_index, -1, condition_index),
                fault_id=fault.fault_id if fault else HEALTHY_ID,
                condition=condition.name,
                seed_key=(),
                parameters=dict(values),
                labels={},
                result=result,
                measurements=measurements,
            )
        return results

    def run_sample(self, sample: Sample) -> SampleResult:
        realised = self.realise(sample)
        result = self.simulator.run(realised.netlist, self.config_for(sample.condition_index))
        result, measurements = self._measured(result, sample.condition_index)
        return SampleResult(
            sample=sample,
            fault_id=self.fault_id(sample),
            condition=self.conditions[sample.condition_index].name,
            seed_key=realised.seed_key,
            parameters=realised.parameters,
            labels=realised.labels,
            result=result,
            measurements=measurements,
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
