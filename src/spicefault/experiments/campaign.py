"""Fault campaign: an experiment simulated to disk, in chunks, resumable, with every
simulation accounted for.
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..dataset.manifest import MANIFEST
from ..dataset.store import SAMPLES, load_dataset, load_metadata
from ..simulation import SimulationStatus
from .engine import run_campaign
from .experiment import HEALTHY_ID, Experiment, Sample
from .seeding import sample_stream

# columns that describe a sample, before its labels, parameters and measurements
DEFINITION_COLUMNS = [
    "sample_id", "fault_index", "fault_id", "fault_type", "fault_location", "fault_magnitude",
    "fault_severity", "replica", "condition", "seed_key", "status", "message", "sim_ok",
    "elapsed_s",
]  # fmt: skip
CIRCUIT_FILE = "circuit.cir"


def simulate_sample(sample: Sample, experiment: Experiment) -> tuple[dict, np.ndarray | None]:
    """One row of the dataset, and the waveform if the experiment stores one.

    Nothing that happens here stops the campaign: a simulation that fails, an output
    that cannot be measured, or an error while building the netlist is recorded in
    the row with its status. `elapsed_s` is the only column that is not reproducible.
    """
    fault = experiment.fault(sample)
    magnitude = None if fault is None else fault.magnitude
    severity = None if fault is None or fault.severity is None else fault.severity.value
    row: dict = {
        "fault_index": sample.fault_index,
        "fault_id": fault.fault_id if fault else HEALTHY_ID,
        "fault_type": fault.fault_type if fault else HEALTHY_ID,
        "fault_location": "+".join(fault.components) if fault else "",
        "fault_magnitude": math.nan if magnitude is None else magnitude,
        "fault_severity": math.nan if severity is None else severity,
        "replica": sample.replica,
        "condition": experiment.conditions[sample.condition_index].name,
        "seed_key": "/".join(map(str, experiment.seed_key(sample))),
        "status": SimulationStatus.FAILED.value,
        "message": "",
        "sim_ok": False,
        "elapsed_s": 0.0,
    }
    measurements = dict.fromkeys(experiment.measurement_columns, math.nan)
    try:
        realised = experiment.realise(sample)
    except Exception as exc:  # a definition that does not fit the circuit
        row["message"] = f"could not build the netlist: {type(exc).__name__}: {exc}"
        return {**row, **measurements}, None
    row.update(realised.labels)
    row.update({f"p_{c}_{p}": value for (c, p), value in realised.parameters.items()})

    result = experiment.simulator.run(
        realised.netlist, experiment.config_for(sample.condition_index)
    )
    row.update(status=result.status.value, message=result.message, elapsed_s=result.elapsed)
    waveform = None
    if result.ok:
        try:
            measurements.update(experiment.measure(result, sample.condition_index))
            if experiment.waveform is not None:
                waveform = experiment.waveform(result)
            active = experiment.columns_for(sample.condition_index)
            if not all(math.isfinite(measurements[column]) for column in active):
                raise ValueError("a measurement is not finite")
            row["sim_ok"] = True
        except Exception as exc:  # the simulation ran but its output cannot be used
            measurements = dict.fromkeys(experiment.measurement_columns, math.nan)
            waveform = None
            row["status"] = SimulationStatus.INVALID_OUTPUT.value
            row["message"] = f"measurement failed: {type(exc).__name__}: {exc}"
    return {**row, **measurements}, waveform


def _simulate_with_tags(sample: Sample, context) -> tuple[dict, np.ndarray | None]:
    experiment, tag_columns = context
    row, waveform = simulate_sample(sample, experiment)
    fault = experiment.fault(sample)
    row.update({column: "" if fault is None else fault.tags.get(column, "")
                for column in tag_columns})  # fmt: skip
    return row, waveform


@dataclass
class ValidationReport:
    """Problems found before running. Errors prevent the run; warnings do not."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def __str__(self) -> str:
        lines = [f"error: {e}" for e in self.errors] + [f"warning: {w}" for w in self.warnings]
        return "\n".join(lines) or "no problems found"


class FaultCampaign:
    """A systematic set of simulations: every fault, `samples_per_fault` times, under
    every operating condition, plus the fault-free circuit.

    It takes the arguments of `Experiment` (with `samples_per_fault` for `samples`), or
    an experiment already built through `from_experiment`. The results go to
    `out_dir` as a dataset: `samples.parquet`, `waveforms.npy` if a waveform is
    declared, and `manifest.json`.
    """

    def __init__(
        self,
        circuit,
        faults=(),
        *,
        out_dir: str | Path,
        samples_per_fault: int = 1,
        tag_columns=(),
        metadata: dict | None = None,
        **kwargs,
    ):
        self.experiment = Experiment(circuit, faults=faults, samples=samples_per_fault, **kwargs)
        self.out_dir = Path(out_dir)
        self.tag_columns = tuple(tag_columns)
        self.user_metadata = dict(metadata or {})
        json.dumps(self.user_metadata)
        self._validate_tag_columns()

    @classmethod
    def from_experiment(
        cls,
        experiment: Experiment,
        out_dir: str | Path,
        *,
        tag_columns=(),
        metadata: dict | None = None,
    ) -> FaultCampaign:
        campaign = object.__new__(cls)
        campaign.experiment, campaign.out_dir = experiment, Path(out_dir)
        campaign.tag_columns = tuple(tag_columns)
        campaign.user_metadata = dict(metadata or {})
        json.dumps(campaign.user_metadata)
        campaign._validate_tag_columns()
        return campaign

    def _validate_tag_columns(self) -> None:
        if len(set(self.tag_columns)) != len(self.tag_columns) or any(
            not isinstance(column, str) or not column for column in self.tag_columns
        ):
            raise ValueError("tag_columns must be unique, non-empty strings")
        parameters = {
            f"p_{component}_{parameter}"
            for variation in self.experiment.variations
            for component, parameter in variation.targets()
        }
        protected = {*DEFINITION_COLUMNS, *parameters, *self.experiment.measurement_columns}
        overlap = protected.intersection(self.tag_columns)
        if overlap:
            raise ValueError(f"tag columns conflict with reserved columns: {sorted(overlap)}")

    def _metadata(self) -> dict:
        record = self.experiment.metadata()
        if self.tag_columns:
            record["fault_tag_columns"] = list(self.tag_columns)
        return record

    # --- before running -----------------------------------------------------------------

    def validate(self, simulate: bool = True) -> ValidationReport:
        """Check that the definitions fit the circuit, without running the campaign.

        Every variation is drawn once, and every fault and operating condition is
        applied to the circuit. With `simulate`, the nominal fault-free circuit is
        also simulated under each condition and measured: a campaign whose healthy
        circuit does not simulate is not worth starting.
        """
        e = self.experiment
        report = ValidationReport()
        if not e.measurement_columns and e.waveform is None:
            report.warnings.append("no measurement and no waveform: only the status is recorded")
        try:
            draw = e.variations.sample(sample_stream(e.seed), e.circuit.netlist())
            e.variations.apply(e.circuit.netlist(), draw.values)
        except Exception as exc:
            report.errors.append(f"variations: {type(exc).__name__}: {exc}")
        for fault in e.faults:
            try:
                fault.apply(e.circuit.netlist())
            except Exception as exc:
                report.errors.append(f"fault {fault.fault_id}: {type(exc).__name__}: {exc}")
        for condition_index, condition in enumerate(e.conditions):
            try:
                netlist = e.circuit.netlist()
                condition.apply(netlist)
            except Exception as exc:
                report.errors.append(f"condition {condition.name}: {type(exc).__name__}: {exc}")
                continue
            if not simulate:
                continue
            result = e.simulator.run(str(netlist), e.config_for(condition_index))
            if not result.ok:
                report.errors.append(
                    f"nominal circuit, condition {condition.name}: "
                    f"{result.status.value}: {result.message}"
                )
                continue
            checks = [(m.name, m.values) for m in e.measurements_for(condition_index)]
            if e.waveform is not None:
                checks.append(("waveform", e.waveform))
            for label, read in checks:
                try:
                    found = read(result)
                    values = list(found.values()) if isinstance(found, dict) else found
                    if not np.all(np.isfinite(values)):
                        raise ValueError("not finite")
                except Exception as exc:
                    report.errors.append(
                        f"measurement {label}, condition {condition.name}: "
                        f"{type(exc).__name__}: {exc}"
                    )
        overlap = e.faults_overlapping_tolerance()
        if overlap:
            report.warnings.append(
                f"{overlap} parametric fault conditions lie partly inside the tolerance band "
                "(see FaultSet.tolerance_overlap)"
            )
        return report

    # --- running ------------------------------------------------------------------------

    def run(
        self,
        workers: int = 1,
        resume: bool = True,
        chunk: int = 2000,
        progress: bool = True,
        validate: bool = True,
    ) -> Path:
        """Simulate every pending sample and write the dataset; returns its folder.

        An interrupted run continues after the last complete chunk when launched again
        with the same definitions; with `resume=False` it starts from the beginning.
        A finished campaign is not run again. Individual failures are recorded and do
        not stop the run.
        """
        e = self.experiment
        if (self.out_dir / MANIFEST).exists():
            if load_metadata(self.out_dir) != json.loads(json.dumps(self._metadata())):
                raise FileExistsError(
                    f"{self.out_dir} holds the dataset of another campaign; choose another folder"
                )
            return self.out_dir
        if not resume and (self.out_dir / "parts").exists():
            shutil.rmtree(self.out_dir / "parts")
        e.check_picklable()  # the pool is used even with one worker
        if validate:
            report = self.validate()
            if not report.ok:
                raise ValueError(f"the campaign is not valid:\n{report}")
        return run_campaign(
            e.plan(),
            _simulate_with_tags,
            (e, self.tag_columns),
            self.out_dir,
            config=self._metadata(),
            n_points=None if e.waveform is None else e.waveform.n_points,
            workers=workers,
            chunk=chunk,
            progress=progress,
            summary=self._summary,
            files={CIRCUIT_FILE: e.circuit.to_netlist()},
            user_metadata=self.user_metadata,
        )

    def _summary(self, df: pd.DataFrame) -> dict:
        """Counts for the manifest, and the role of each column of the table."""
        e = self.experiment
        status = df["status"].value_counts().to_dict()
        targets = [t for v in e.variations for t in v.targets()]
        measurements = list(e.measurement_columns)
        parameters = {f"p_{c}_{p}": [c, p] for c, p in targets}
        known = {*DEFINITION_COLUMNS, *parameters, *measurements}
        return {
            "n_completed": int(df["sim_ok"].sum()),
            "n_failed": int((~df["sim_ok"]).sum()),
            "status_counts": {k: int(v) for k, v in sorted(status.items())},
            "columns": {
                "definition": DEFINITION_COLUMNS,
                "labels": [c for c in df.columns if c not in known],
                "parameters": parameters,
                "measurements": measurements,
            },
        }

    # --- after, or while, running -------------------------------------------------------

    def _rows(self) -> pd.DataFrame:
        """Rows simulated so far: the dataset, or the chunks of a run in progress."""
        if (self.out_dir / SAMPLES).exists():
            return pd.read_parquet(self.out_dir / SAMPLES)
        parts = sorted((self.out_dir / "parts").glob("part_*.parquet"))
        if not parts:
            return pd.DataFrame(columns=DEFINITION_COLUMNS)
        return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)

    def status(self) -> dict[str, int]:
        """Samples completed (simulated successfully), failed and still pending."""
        rows = self._rows()
        total = len(self.experiment.plan())
        completed = int(rows["sim_ok"].sum()) if len(rows) else 0
        return {
            "total": total,
            "completed": completed,
            "failed": len(rows) - completed,
            "pending": total - len(rows),
        }

    def summary(self) -> pd.DataFrame:
        """One row per fault: samples simulated, successful, and the count of each status."""
        rows = self._rows()
        order = [HEALTHY_ID, *(f.fault_id for f in self.experiment.faults)]
        counts = pd.crosstab(rows["fault_id"], rows["status"]).reindex(order, fill_value=0)
        counts.insert(0, "samples", counts.sum(axis=1))
        counts.insert(1, "success_rate", rows.groupby("fault_id")["sim_ok"].mean().reindex(order))
        return counts.rename_axis(columns=None)

    def failures(self) -> pd.DataFrame:
        """The samples that did not give a usable result, with status and message."""
        rows = self._rows()
        failed = ~rows["sim_ok"].astype(bool)
        return rows.loc[failed, DEFINITION_COLUMNS[:-2]].reset_index(drop=True)

    def dataset(self):
        """The finished campaign as a `Dataset`."""
        from ..dataset import Dataset

        return Dataset(self.out_dir)

    def load(self, drop_failed: bool = True):
        """(samples, waveforms, manifest) of the finished campaign."""
        return load_dataset(self.out_dir, drop_failed)

