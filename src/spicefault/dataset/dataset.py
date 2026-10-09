"""The dataset of a campaign as an object: traceable, verifiable and reproducible."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..circuit import Circuit
from ..conditions import OperatingCondition
from ..faults import FaultSet
from ..variation import VariationSet
from .manifest import Manifest, file_record
from .specification import Specification
from .store import SAMPLES, WAVEFORMS, load_metadata, read_waveforms

CIRCUIT_FILE = "circuit.cir"
HEALTHY_ID = "healthy"


@dataclass(frozen=True)
class Provenance:
    """Everything that identifies one sample and what produced it."""

    sample_id: int
    seed: int
    seeding: str
    seed_key: str
    circuit: str
    netlist_sha256: str
    fault_id: str
    fault_type: str
    fault_location: str
    fault_magnitude: float | None
    fault_severity: float | None
    condition: str
    status: str
    message: str
    simulator: str
    simulator_version: str
    spicefault_version: str
    parameters: dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)


class Dataset:
    """A dataset folder written by `FaultCampaign`.

    `samples` has one row per simulation, failed ones included; `waveforms`, if any,
    is aligned with it row by row, with NaN where a simulation failed or its
    operating condition stores no waveform. The file holds no row for those
    conditions: `waveforms` is then a `StoredWaveforms`, which reads like the full
    array and gives the array of the file as `waveforms.stored`. `metadata` is the
    definition of the experiment and `manifest` the record of the run.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.manifest = Manifest.read(self.path)
        self.metadata = load_metadata(self.path)
        self.samples = pd.read_parquet(self.path / SAMPLES)
        try:
            self.waveforms = read_waveforms(
                self.path, self.samples, self.manifest.summary, mmap=True
            )
        except ValueError:  # the file does not have the rows the manifest says: see verify
            self.waveforms = None
        columns = self.manifest.summary.get("columns", {})
        self.features: list[str] = list(columns.get("measurements", []))
        self.labels: list[str] = list(columns.get("labels", []))
        self.parameters: dict[str, tuple[str, str]] = {
            column: tuple(target) for column, target in columns.get("parameters", {}).items()
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __repr__(self) -> str:
        return (
            f"Dataset({str(self.path)!r}: {len(self)} samples, "
            f"{len(self.metadata['faults'])} faults, {len(self.features)} measurements)"
        )

    @property
    def ok(self) -> np.ndarray:
        """True for the samples whose simulation gave a usable result."""
        return self.samples["sim_ok"].to_numpy(dtype=bool)

    @property
    def circuit(self) -> Circuit:
        return Circuit(
            (self.path / CIRCUIT_FILE).read_text(),
            self.metadata["circuit"],
            imported_analyses=self.metadata.get("imported_analyses", ()),
        )

    @property
    def faults(self) -> FaultSet:
        return FaultSet.from_metadata(self.metadata["faults"])

    # --- integrity ----------------------------------------------------------------------

    def verify(self) -> list[str]:
        """Problems with the files or their consistency; an empty list if there are none.

        The files must match the fingerprints of the manifest, the table must have one
        row per planned sample, and failed samples must carry no result.
        """
        problems = self.manifest.verify(self.path)
        problems.extend(self._verify_table())
        problems.extend(self._verify_circuit_and_includes())
        problems.extend(self._verify_results())
        return problems

    def _verify_table(self) -> list[str]:
        problems = []
        df = self.samples
        if len(df) != self.manifest.n_samples:
            problems.append(f"{len(df)} rows, the manifest says {self.manifest.n_samples}")
        if list(df["sample_id"]) != list(range(len(df))):
            problems.append("sample_id is not the row number")
        return problems

    def _verify_circuit_and_includes(self) -> list[str]:
        problems = []
        netlist = (self.path / CIRCUIT_FILE).read_bytes()
        if hashlib.sha256(netlist).hexdigest() != self.metadata["netlist_sha256"]:
            problems.append(f"{CIRCUIT_FILE} is not the netlist of the experiment")
        for include in self.metadata.get("includes", []):
            path = Path(include["path"])
            if not path.is_file():
                problems.append(f"included file is missing: {path}")
            elif hashlib.sha256(path.read_bytes()).hexdigest() != include["sha256"]:
                problems.append(f"included file differs from the experiment: {path}")
        return problems

    def _verify_results(self) -> list[str]:
        problems = []
        df = self.samples
        defined = {HEALTHY_ID, *(f["fault_id"] for f in self.metadata["faults"])}
        unknown = set(df["fault_id"]) - defined
        if unknown:
            problems.append(f"faults not defined in the metadata: {sorted(unknown)}")
        ok = self.ok
        if self.features:
            values = df[self.features].to_numpy(float)
            expected = self._expected_measurements()
            if (expected & ~np.isfinite(values))[ok].any():
                problems.append("a successful sample has a measurement that is not finite")
            if (~expected & ~np.isnan(values))[ok].any():
                problems.append(
                    "a sample has a measurement that its operating condition does not declare"
                )
            if (~np.isnan(values))[~ok].any():
                problems.append("a failed sample has a measurement")
        if self.waveforms is None and (self.path / WAVEFORMS).exists():
            problems.append("waveforms and samples have different lengths")
        if self.waveforms is not None:
            if len(self.waveforms) != len(df):
                problems.append("waveforms and samples have different lengths")
            else:
                stored = ok & self._stores_waveform()
                missing = np.isnan(self.waveforms[stored]).any()
                if missing or not np.isnan(self.waveforms[~stored]).all():
                    problems.append("waveforms do not match the status of the samples")
        return problems

    def _measured(self) -> dict[str, list[str]]:
        """Operating condition -> the measurement columns it fills.

        A condition that declares its own measurements fills only those; the columns
        of the other conditions hold NaN in its rows.
        """
        default = self.metadata.get("measurements", [])
        return {
            condition["name"]: [
                column
                for measurement in condition.get("measurements", default)
                for column in measurement.get("names", [measurement["name"]])
            ]
            for condition in self.metadata.get("conditions", [])
        }

    def _stores_waveform(self) -> np.ndarray:
        """True for the rows whose operating condition stores a waveform."""
        default = self.metadata.get("waveform")
        stores = {
            condition["name"]: bool(condition.get("waveform", default))
            for condition in self.metadata.get("conditions", [])
        }
        condition = self.samples["condition"].to_numpy()
        return np.array([stores.get(name, True) for name in condition], dtype=bool)

    def _expected_measurements(self) -> np.ndarray:
        """[samples, features]: True where the condition of the row fills the column."""
        measured = self._measured()
        condition = self.samples["condition"].to_numpy()
        expected = np.ones((len(self.samples), len(self.features)), dtype=bool)
        for name, columns in measured.items():
            rows = condition == name
            expected[rows] = [feature in columns for feature in self.features]
        return expected

    def update_columns(self, frame: pd.DataFrame, note: str = "") -> None:
        """Add or replace derived label columns without invalidating dataset integrity."""
        self._check_derived(frame)
        self._store_columns(frame, {"type": "derived_columns", "note": note})

    def _check_derived(self, frame: pd.DataFrame) -> None:
        if frame.empty or len(frame.columns) == 0:
            raise ValueError("frame must contain at least one derived column")
        if not frame.index.equals(self.samples.index):
            raise ValueError("frame rows must have the same index and order as dataset.samples")
        if not frame.columns.is_unique:
            raise ValueError("frame column names must be unique")
        reserved = {
            *self.manifest.summary.get("columns", {}).get("definition", []),
            *self.parameters,
            *self.features,
        }
        protected = reserved.intersection(frame.columns)
        if protected:
            raise ValueError(f"cannot update definition, parameter or measurement columns: "
                             f"{sorted(protected)}")
        existing = set(self.samples.columns) - set(self.labels)
        conflicts = existing.intersection(frame.columns)
        if conflicts:
            raise ValueError(f"columns are not derived labels: {sorted(conflicts)}")

    def _store_columns(self, frame: pd.DataFrame, entry: dict, drop: Sequence[str] = ()) -> None:
        """Write label columns to the table, with its fingerprint and an entry in the history."""
        updated = self.samples.drop(columns=list(drop))
        for column in frame.columns:
            updated[column] = frame[column].to_numpy()
        path = self.path / SAMPLES
        temporary = path.with_suffix(".tmp")
        updated.to_parquet(temporary, index=False)
        temporary.replace(path)
        self.samples = updated
        kept = [label for label in self.labels if label not in drop]
        self.labels = list(dict.fromkeys([*kept, *frame.columns]))
        columns = self.manifest.summary.setdefault("columns", {})
        columns["labels"] = list(self.labels)
        self.manifest.summary.setdefault("history", []).append(
            {
                "type": entry["type"],
                "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "columns": list(frame.columns),
                **{key: value for key, value in entry.items() if key != "type"},
            }
        )
        self.manifest.files[SAMPLES] = file_record(path)
        self.manifest.write(self.path)

    # --- specifications -----------------------------------------------------------------

    @property
    def specifications(self) -> list[Specification]:
        """The specifications the dataset was last labelled with."""
        records = self.manifest.summary.get("specifications", [])
        return [Specification.from_metadata(record) for record in records]

    def label(self, specifications: Sequence[Specification], note: str = "") -> pd.DataFrame:
        """Label every sample as meeting its specifications or not, and keep the limits.

        It writes `ok_<name>` for each specification, `compliant` (all of them are
        met) and `violated` (the names of those that are not, separated by commas),
        stores the limits in the manifest and appends to its history. Calling it again
        with other limits relabels the dataset; nothing is simulated.

        A specification is a property of the drawn circuit, so the rows of one circuit
        under its operating conditions get the same labels: the value is read in the
        conditions that measure it. A value that could not be computed, also because
        the simulation failed, is a violation. Returns the labels written.
        """
        specifications = list(specifications)
        names = [specification.name for specification in specifications]
        if not names or len(set(names)) != len(names):
            raise ValueError("label needs specifications with different names")
        unknown = [name for name in names if name not in self.samples]
        if unknown:
            raise KeyError(f"no column for the specifications {unknown}")

        circuit = self._case_index()
        n_circuits = int(circuit.max()) + 1 if len(circuit) else 0
        measured = self._measured()
        condition = self.samples["condition"].to_numpy()
        labels = {}
        for specification in specifications:
            met = specification.met(self.samples[specification.name])
            measures = np.ones(len(self), dtype=bool)
            if specification.name in self.features:
                # a condition the metadata does not describe is taken to measure it
                name = specification.name
                measures = np.array([name in measured.get(c, (name,)) for c in condition])
            violated = np.bincount(circuit, measures & ~met, n_circuits) > 0
            evaluated = np.bincount(circuit, measures, n_circuits) > 0
            labels[specification.column] = (evaluated & ~violated)[circuit]
        frame = pd.DataFrame(labels, index=self.samples.index)
        frame["compliant"] = frame.all(axis=1)
        failed = ~frame[[s.column for s in specifications]].to_numpy()
        frame["violated"] = [",".join(n for n, bad in zip(names, row, strict=True) if bad)
                             for row in failed]  # fmt: skip

        self._check_derived(frame)
        records = [specification.metadata() for specification in specifications]
        previous = {s.column for s in self.specifications}
        self.manifest.summary["specifications"] = records
        self._store_columns(
            frame,
            {"type": "specifications", "specifications": records, "note": note},
            drop=sorted(previous - set(frame.columns)),
        )
        return frame

    def _case_index(self) -> np.ndarray:
        """For each row, the number of its drawn circuit: one per (fault, replica)."""
        keys = self.samples[["fault_index", "replica"]]
        return keys.groupby(["fault_index", "replica"], sort=False).ngroup().to_numpy()

    # --- one row per drawn circuit ------------------------------------------------------

    def cases(self, waveform_from: str | None = None) -> tuple[pd.DataFrame, np.ndarray | None]:
        """(cases, waveforms): one row per drawn circuit, across its operating conditions.

        The table of samples has one row per simulation, so a circuit drawn once and
        simulated under several conditions is several rows, each with the measurements
        of its condition. Here they are joined: the definition of the circuit, its
        labels and its realised parameters once, then the measurements of every
        condition side by side. A measurement that several conditions make gets one
        column per condition, `<name>_<condition>`. `sim_ok` is true only if every
        condition succeeded, and `cases.attrs["features"]` lists the measurement
        columns. Labels are read from the row of the first condition.

        `waveforms` are those of the condition `waveform_from`, aligned with the cases;
        it may be omitted when a single condition stores them. None if the dataset
        has no waveforms.
        """
        cases, _, waveforms = self._cases(waveform_from, self.waveforms is not None)
        return cases, waveforms

    def _cases(self, waveform_from: str | None, with_waveforms: bool):
        df, measured = self.samples, self._measured()
        names = list(measured)
        if not names:
            raise ValueError("the metadata of this dataset does not describe its conditions")
        keys = ["fault_index", "replica"]
        per_row = {*self.features, "sample_id", "condition", "status", "message", "sim_ok",
                   "elapsed_s"}  # fmt: skip
        rows = {name: df[(df["condition"] == name).to_numpy()] for name in names}
        cases = rows[names[0]][[c for c in df.columns if c not in per_row]].reset_index(drop=True)
        makers = pd.Series([column for name in names for column in measured[name]]).value_counts()
        ok = np.ones(len(cases), dtype=bool)
        values, features = {}, []
        for name in names:
            if not np.array_equal(rows[name][keys].to_numpy(), cases[keys].to_numpy()):
                raise ValueError(f"condition {name!r} does not have the circuits of the others")
            ok &= rows[name]["sim_ok"].to_numpy(dtype=bool)
            for column in measured[name]:
                target = column if makers[column] == 1 else f"{column}_{name}"
                values[target] = rows[name][column].to_numpy()
                features.append(target)
        clash = set(values).intersection(cases.columns) or len(set(features)) != len(features)
        if clash:
            raise ValueError(f"case columns are not unique: {clash}")
        cases.insert(list(cases.columns).index("seed_key") + 1, "sim_ok", ok)
        cases = pd.concat([cases, pd.DataFrame(values, index=cases.index)], axis=1)
        cases.attrs["features"] = features

        waveforms = None
        if with_waveforms:
            if self.waveforms is None:
                raise ValueError("this dataset has no waveforms")
            stores = self._stores_waveform()
            storing = [name for name in names if stores[(df["condition"] == name).to_numpy()].any()]
            if waveform_from is None and len(storing) != 1:
                raise ValueError(
                    f"the conditions {storing} store a waveform; choose one with waveform_from"
                )
            chosen = storing[0] if waveform_from is None else waveform_from
            if chosen not in storing:
                raise ValueError(f"condition {chosen!r} stores no waveform; those that do: "
                                 f"{storing}")  # fmt: skip
            waveforms = np.asarray(self.waveforms[(df["condition"] == chosen).to_numpy()])
        return cases, features, waveforms

    # --- traceability -------------------------------------------------------------------

    def _row(self, sample_id: int) -> pd.Series:
        row = self.samples.iloc[sample_id]
        if row["sample_id"] != sample_id:
            raise ValueError("sample_id is not the row number: the table was altered")
        return row

    def provenance(self, sample_id: int) -> Provenance:
        row, meta = self._row(sample_id), self.metadata

        def number(value) -> float | None:
            return None if pd.isna(value) else float(value)

        return Provenance(
            sample_id=int(sample_id),
            seed=meta["seed"],
            seeding=meta["seeding"],
            seed_key=row["seed_key"],
            circuit=meta["circuit"],
            netlist_sha256=meta["netlist_sha256"],
            fault_id=row["fault_id"],
            fault_type=row["fault_type"],
            fault_location=row["fault_location"],
            fault_magnitude=number(row["fault_magnitude"]),
            fault_severity=number(row["fault_severity"]),
            condition=row["condition"],
            status=row["status"],
            message=row["message"],
            simulator=meta["simulator"],
            simulator_version=meta["simulator_version"],
            spicefault_version=meta["spicefault_version"],
            parameters={c: float(row[c]) for c in self.parameters if not pd.isna(row[c])},
        )

    def netlist(self, sample_id: int) -> str:
        """The netlist that was simulated for a sample, rebuilt from what is stored.

        The realised values are read from the table, not drawn again, so this works
        whatever variations the experiment used.
        """
        row = self._row(sample_id)
        values = {target: row[column] for column, target in self.parameters.items()}
        if any(pd.isna(v) for v in values.values()):
            raise ValueError(f"sample {sample_id} has no realised values: {row['message']}")
        netlist = self.circuit.netlist()
        VariationSet.apply(netlist, {target: float(v) for target, v in values.items()})
        if row["fault_id"] != HEALTHY_ID:
            self.faults[row["fault_id"]].apply(netlist)
        # only the settings and the temperature change the netlist: the measurements of
        # the condition are not rebuilt, since a custom one holds a function
        conditions = {c["name"]: c for c in self.metadata["conditions"]}
        record = conditions[row["condition"]]
        OperatingCondition.from_metadata(
            {key: record[key] for key in ("name", "temperature", "settings")}
        ).apply(netlist)
        return str(netlist)

    # --- reproducibility ----------------------------------------------------------------

    def experiment(self, **overrides):
        """The experiment that produced the dataset, rebuilt from its metadata.

        Custom or joint variations, custom measurements and custom backends cannot be
        stored and must be passed again (`variations=`, `measurements=`, `simulator=`).
        """
        from ..experiments import Experiment

        return Experiment.from_metadata(
            self.metadata, (self.path / CIRCUIT_FILE).read_text(), **overrides
        )

    def reproduce(
        self,
        sample_ids: Sequence[int] | None = None,
        n: int = 20,
        experiment=None,
        seed: int = 0,
    ) -> pd.DataFrame:
        """Simulate samples again and compare them with what is stored.

        By default `n` samples chosen at random. One row per sample:
        - `definition`: same fault, condition and seed key;
        - `parameters`: the redrawn values are exactly the stored ones;
        - `status`: same simulation status;
        - `max_abs_diff`, `max_rel_diff`: largest difference over the measurements;
        - `waveform_abs_diff`: largest difference over the waveform, if stored.
        On the platform and simulator version that wrote the dataset the differences
        are expected to be zero; elsewhere they measure the dependence on both.
        """
        from ..experiments import simulate_sample

        experiment = experiment or self.experiment()
        plan = experiment.plan()
        if len(plan) != len(self):
            raise ValueError("the experiment does not have the samples of this dataset")
        if sample_ids is None:
            rng = np.random.default_rng(seed)
            sample_ids = np.sort(rng.choice(len(self), min(n, len(self)), replace=False))
        rows = []
        for sample_id in map(int, sample_ids):
            stored = self._row(sample_id)
            new, waveform = simulate_sample(plan[sample_id], experiment)
            ours = np.array([new[f] for f in self.features], dtype=float)
            theirs = stored[self.features].to_numpy(dtype=float)
            diff = np.abs(ours - theirs)
            both_missing = np.isnan(ours) & np.isnan(theirs)
            diff = np.where(both_missing, 0.0, diff)
            waveform_diff = np.nan
            if self.waveforms is not None and waveform is not None:
                waveform_diff = float(np.abs(waveform - self.waveforms[sample_id]).max())
            rows.append(
                {
                    "sample_id": sample_id,
                    "definition": all(
                        new[c] == stored[c] for c in ("fault_id", "condition", "seed_key")
                    ),
                    "parameters": all(
                        new.get(c) == stored[c] or (pd.isna(new.get(c)) and pd.isna(stored[c]))
                        for c in self.parameters
                    ),
                    "status": new["status"] == stored["status"],
                    "max_abs_diff": float(diff.max()) if len(diff) else 0.0,
                    "max_rel_diff": float(
                        (diff / np.maximum(np.abs(theirs), 1e-300))[~both_missing].max()
                    )
                    if (~both_missing).any()
                    else 0.0,
                    "waveform_abs_diff": waveform_diff,
                }
            )
        return pd.DataFrame(rows)

    # --- uses ---------------------------------------------------------------------------

    def to_ml(
        self,
        target: str | Callable[[pd.DataFrame], np.ndarray] = "fault_id",
        features: Sequence[str] | None = None,
        waveforms: bool = False,
        drop_failed: bool = True,
        by_case: bool = False,
        waveform_from: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """(X, y) for a statistical or machine-learning model.

        X is the table of measurements (or of `features`), or the waveforms with
        `waveforms`, for the samples of the conditions that store one. y is a column
        of the samples, by default the fault identifier (`fault_type` and
        `fault_location` are the usual alternatives), or the result
        of a function of the samples. The library does not depend on any ML framework.

        With `by_case` there is one row per drawn circuit instead of one per
        simulation, with the measurements of all its operating conditions and the
        waveform of the condition `waveform_from` (see `cases`).
        """
        if by_case:
            samples, case_features, stored = self._cases(waveform_from, waveforms)
            keep = samples["sim_ok"].to_numpy(dtype=bool) if drop_failed else slice(None)
            samples = samples[keep]
            if waveforms:
                x = stored[keep]
            else:
                x = samples[list(features or case_features)].to_numpy(dtype=float)
            y = target(samples) if callable(target) else samples[target].to_numpy()
            return x, np.asarray(y)
        keep = self.ok if drop_failed else np.ones(len(self), dtype=bool)
        if waveforms:
            if self.waveforms is None:
                raise ValueError("this dataset has no waveforms")
            keep = keep & self._stores_waveform()
        samples = self.samples[keep]
        if waveforms:
            x = np.asarray(self.waveforms[keep])
        else:
            x = samples[list(features or self.features)].to_numpy(dtype=float)
        y = target(samples) if callable(target) else samples[target].to_numpy()
        return x, np.asarray(y)

    def analysis(self, features: Sequence[str] | None = None, by_case: bool = False, **kwargs):
        """A `ReliabilityAnalysis` of the dataset.

        With `by_case` it is made on one row per drawn circuit (see `cases`), which a
        dataset whose operating conditions measure different things needs.
        """
        from ..reliability import ReliabilityAnalysis

        samples, default = self.samples, self.features
        if by_case:
            samples, default, _ = self._cases(None, False)
        return ReliabilityAnalysis(
            samples, features or default, faults=self.metadata["faults"], **kwargs
        )
