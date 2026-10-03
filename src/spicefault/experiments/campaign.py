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

from ..dataset.store import MANIFEST, SAMPLES, load_dataset
from ..simulation import SimulationStatus
from .engine import run_campaign
from .experiment import HEALTHY_ID, Experiment, Sample
from .seeding import sample_stream

# columns that describe a sample, before its labels, parameters and measurements
DEFINITION_COLUMNS = [
    "sample_id", "fault_index", "fault_id", "fault_type", "replica", "condition", "seed_key",
    "status", "message", "sim_ok", "elapsed_s",
]  # fmt: skip


def simulate_sample(sample: Sample, experiment: Experiment) -> tuple[dict, np.ndarray | None]:
    """One row of the dataset, and the waveform if the experiment stores one.

    Nothing that happens here stops the campaign: a simulation that fails, an output
    that cannot be measured, or an error while building the netlist is recorded in
    the row with its status. `elapsed_s` is the only column that is not reproducible.
    """
    fault = experiment.fault(sample)
    row: dict = {
        "fault_index": sample.fault_index,
        "fault_id": fault.fault_id if fault else HEALTHY_ID,
        "fault_type": fault.fault_type if fault else HEALTHY_ID,
        "replica": sample.replica,
        "condition": experiment.conditions[sample.condition_index].name,
        "seed_key": "/".join(map(str, experiment.seed_key(sample))),
        "status": SimulationStatus.FAILED.value,
        "message": "",
        "sim_ok": False,
        "elapsed_s": 0.0,
    }
    measurements = dict.fromkeys((m.name for m in experiment.measurements), math.nan)
    try:
        realised = experiment.realise(sample)
    except Exception as exc:  # a definition that does not fit the circuit
        row["message"] = f"could not build the netlist: {type(exc).__name__}: {exc}"
        return {**row, **measurements}, None
    row.update(realised.labels)
    row.update({f"p_{c}_{p}": value for (c, p), value in realised.parameters.items()})

    result = experiment.simulator.run(realised.netlist, experiment.config)
    row.update(status=result.status.value, message=result.message, elapsed_s=result.elapsed)
    waveform = None
    if result.ok:
        try:
            measurements = experiment.measure(result)
            if experiment.waveform is not None:
                waveform = experiment.waveform(result)
            if not all(math.isfinite(v) for v in measurements.values()):
                raise ValueError("a measurement is not finite")
            row["sim_ok"] = True
        except Exception as exc:  # the simulation ran but its output cannot be used
            measurements = dict.fromkeys(measurements, math.nan)
            waveform = None
            row["status"] = SimulationStatus.INVALID_OUTPUT.value
            row["message"] = f"measurement failed: {type(exc).__name__}: {exc}"
    return {**row, **measurements}, waveform


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
        self, circuit, faults=(), *, out_dir: str | Path, samples_per_fault: int = 1, **kwargs
    ):
        self.experiment = Experiment(circuit, faults=faults, samples=samples_per_fault, **kwargs)
        self.out_dir = Path(out_dir)

    @classmethod
    def from_experiment(cls, experiment: Experiment, out_dir: str | Path) -> FaultCampaign:
        campaign = object.__new__(cls)
        campaign.experiment, campaign.out_dir = experiment, Path(out_dir)
        return campaign

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
        if not e.measurements and e.waveform is None:
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
        for condition in e.conditions:
            try:
                netlist = e.circuit.netlist()
                condition.apply(netlist)
            except Exception as exc:
                report.errors.append(f"condition {condition.name}: {type(exc).__name__}: {exc}")
                continue
            if not simulate:
                continue
            result = e.simulator.run(str(netlist), e.config)
            if not result.ok:
                report.errors.append(
                    f"nominal circuit, condition {condition.name}: "
                    f"{result.status.value}: {result.message}"
                )
                continue
            for item in (*e.measurements, *([e.waveform] if e.waveform else [])):
                label = getattr(item, "name", "waveform")
                try:
                    if not np.all(np.isfinite(item(result))):
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
            if json.loads((self.out_dir / MANIFEST).read_text())["config"] != e.metadata():
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
            simulate_sample,
            e,
            self.out_dir,
            config=e.metadata(),
            n_points=None if e.waveform is None else e.waveform.n_points,
            workers=workers,
            chunk=chunk,
            progress=progress,
            summary=_counts,
        )

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

    def load(self, drop_failed: bool = True):
        """(samples, waveforms, manifest) of the finished campaign."""
        return load_dataset(self.out_dir, drop_failed)


def _counts(df: pd.DataFrame) -> dict:
    status = df["status"].value_counts().to_dict()
    return {
        "n_completed": int(df["sim_ok"].sum()),
        "n_failed": int((~df["sim_ok"]).sum()),
        "status_counts": {k: int(v) for k, v in sorted(status.items())},
    }
