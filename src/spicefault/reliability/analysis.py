"""Reliability analysis of the fault responses of a campaign.

Every figure here is conditional on the features used, on the measurement model
already applied to them, on the normal variation of the campaign and on its
operating condition (docs/RELIABILITY_METRICS.md). Metric identifiers (M1, M2, ...)
refer to that document.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..dataset.store import load_dataset, load_metadata
from .detection import Detector, LimitTest
from .statistics import INTERVALS, auc, bootstrap_interval, robust_spread
from .structure import ambiguity_groups, component_groups, confusable_components, pairwise_shift


@dataclass(frozen=True)
class FaultResponse:
    """The simulated response of one fault condition: the unit of the analysis.

    `features` holds the successful simulations, one row each; `n_total` counts the
    failed ones too. `record` is the record of the fault, when the analysis has it.
    """

    fault_id: str
    features: pd.DataFrame
    n_total: int
    record: dict | None = None

    @property
    def n_ok(self) -> int:
        return len(self.features)

    @property
    def n_failed(self) -> int:
        return self.n_total - self.n_ok

    def centre(self) -> pd.Series:
        return self.features.median()

    def spread(self) -> pd.Series:
        return pd.Series(robust_spread(self.features.to_numpy()), index=self.features.columns)


@dataclass(frozen=True)
class Detectability:
    """M1. `table` has one row per fault; the false-alarm rate was measured on
    `n_evaluation` healthy samples, the thresholds were set on `n_calibration`.
    """

    table: pd.DataFrame
    alpha: float
    false_alarm: float
    false_alarm_interval: tuple[float, float]
    n_calibration: int
    n_evaluation: int
    held_out: bool


@dataclass(frozen=True)
class Ambiguity:
    """M9b, at one threshold of d'."""

    threshold: float
    undetectable: list[str]  # conditions linked directly to healthy
    groups: list[list[str]]  # connected conditions (transitive: an upper bound)
    confusable: dict[str, list[str]] | None  # per component, without transitivity
    component_groups: list[list[str]] | None


class ReliabilityAnalysis:
    """Metrics computed from the samples of one campaign, at one operating condition.

    `samples` has one row per simulation, with the fault identifier in column `fault`
    (`healthy` for the fault-free ones), the measurements in the `features` columns
    and, optionally, a boolean column `ok` (False for failed simulations) and a
    boolean column `compliant` (the circuit meets its specifications).

    - `noise_floor`: standard deviation of the measurement noise of each feature; a
      lower bound for spreads.
    - `healthy_split`: fraction of the healthy samples used to set thresholds; the
      rest measure the false-alarm rate. None uses all of them for both, which biases
      the false-alarm rate downwards.
    - `faults`: fault records (`Fault.metadata()`), for the metrics that need the
      component or the magnitude of each fault.
    """

    def __init__(
        self,
        samples: pd.DataFrame,
        features: Sequence[str],
        *,
        fault: str = "fault_id",
        healthy: str = "healthy",
        ok: str | None = "sim_ok",
        compliant: str | None = None,
        noise_floor: Mapping[str, float] | None = None,
        healthy_split: float | None = 0.5,
        faults: Sequence[dict] | None = None,
        confidence: float = 0.95,
        interval: str = "wilson",
        seed: int = 0,
    ):
        self.features = list(features)
        self.fault_column, self.healthy_id, self.compliant_column = fault, healthy, compliant
        self.confidence, self.seed, self.healthy_split = confidence, seed, healthy_split
        self._interval = INTERVALS[interval]
        self._settings = {
            "fault": fault, "healthy": healthy, "ok": ok, "compliant": compliant,
            "noise_floor": noise_floor, "healthy_split": healthy_split, "faults": faults,
            "confidence": confidence, "interval": interval, "seed": seed,
        }  # fmt: skip
        self.noise_floor = None if noise_floor is None else pd.Series(noise_floor, dtype=float)
        self.records = {r["fault_id"]: r for r in faults or ()}

        self._source = samples
        valid = samples[ok].astype(bool) if ok and ok in samples else pd.Series(True, samples.index)
        self.n_total = samples[fault].value_counts(sort=False)
        self.samples = samples[valid.to_numpy()].reset_index(drop=True)
        if not np.isfinite(self.samples[self.features].to_numpy(dtype=float)).all():
            raise ValueError("a successful sample has a feature that is not finite")
        ids = list(dict.fromkeys(samples[fault]))
        if healthy not in ids:
            raise ValueError(f"no sample of the healthy condition {healthy!r}")
        self.fault_ids = [i for i in ids if i != healthy]

        is_healthy = (self.samples[fault] == healthy).to_numpy()
        healthy_rows = np.flatnonzero(is_healthy)
        if healthy_split is None:
            self._calibration = self._evaluation = healthy_rows
        else:
            order = np.random.default_rng(seed).permutation(healthy_rows)
            n_cal = round(healthy_split * len(order))
            self._calibration, self._evaluation = np.sort(order[:n_cal]), np.sort(order[n_cal:])
            if not len(self._calibration) or not len(self._evaluation):
                raise ValueError("too few healthy samples to split")
        self._x = self.samples[self.features].to_numpy(dtype=float)
        self._rows = self.samples.groupby(fault, sort=False).indices

    @classmethod
    def from_dataset(
        cls, path: str | Path, features: Sequence[str] | None = None, **kwargs
    ) -> ReliabilityAnalysis:
        """Analysis of a dataset written by `FaultCampaign`. The features default to its
        measurements, and the fault records are taken from its manifest.
        """
        samples, _, _ = load_dataset(path, drop_failed=False)
        config = load_metadata(path)
        features = features or [m["name"] for m in config["measurements"]]
        return cls(samples, features, faults=config["faults"], **kwargs)

    def by(self, column: str) -> dict[object, ReliabilityAnalysis]:
        """One analysis per value of `column`, e.g. per operating condition (M11).

        Each has its own healthy reference, drawn from the samples of that value.
        """
        return {
            value: ReliabilityAnalysis(part, self.features, **self._settings)
            for value, part in self._source.groupby(column, sort=False)
        }

    # --- the populations ----------------------------------------------------------------

    def counts(self) -> pd.DataFrame:
        """Samples requested, simulated successfully and failed, per condition."""
        n_ok = self.samples[self.fault_column].value_counts(sort=False)
        table = pd.DataFrame({"n": self.n_total})
        table["n_ok"] = n_ok.reindex(table.index, fill_value=0)
        table["n_failed"] = table["n"] - table["n_ok"]
        return table.loc[[self.healthy_id, *self.fault_ids]]

    def response(self, fault_id: str) -> FaultResponse:
        rows = self._rows.get(fault_id, np.array([], dtype=int))
        return FaultResponse(
            fault_id,
            self.samples.loc[rows, self.features].reset_index(drop=True),
            int(self.n_total[fault_id]),
            self.records.get(fault_id),
        )

    def dictionary(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """(centre, spread) of every feature per condition: median and IQR / 1.349."""
        grouped = self.samples.groupby(self.fault_column, sort=False)[self.features]
        spread = (grouped.quantile(0.75) - grouped.quantile(0.25)) / 1.349
        return grouped.median(), spread

    # --- M1: detection probability at a fixed false-alarm rate --------------------------

    def flags(self, alpha: float = 0.01, detector: Detector | None = None) -> np.ndarray:
        """Decision of the detector for every successful sample, in the order of `samples`."""
        detector = detector or LimitTest(alpha)
        return np.asarray(detector.fit(self._x[self._calibration]).flag(self._x), dtype=bool)

    def detectability(self, alpha: float = 0.01, detector: Detector | None = None) -> Detectability:
        """P(detected | fault), with the detector's thresholds set on healthy samples.

        Per fault: `p_detect` over the successful simulations, with its interval, and
        `bound_low`, `bound_high`: the same proportion over all the simulations
        requested, counting every failed one as undetected and as detected. Failed
        simulations are not missing at random, so the bounds are what can be claimed
        when there are any.
        """
        flagged = self.flags(alpha, detector)
        rows = []
        for fault_id in self.fault_ids:
            idx = self._rows.get(fault_id, np.array([], dtype=int))
            k, n, total = int(flagged[idx].sum()), len(idx), int(self.n_total[fault_id])
            low, high = self._interval(k, n, self.confidence)
            rows.append(
                {
                    "fault_id": fault_id,
                    "n": total,
                    "n_ok": n,
                    "detected": k,
                    "p_detect": k / n if n else np.nan,
                    "ci_low": low,
                    "ci_high": high,
                    "bound_low": k / total,
                    "bound_high": (k + total - n) / total,
                }
            )
        k, n = int(flagged[self._evaluation].sum()), len(self._evaluation)
        return Detectability(
            pd.DataFrame(rows).set_index("fault_id"),
            alpha,
            k / n,
            self._interval(k, n, self.confidence),
            len(self._calibration),
            n,
            self.healthy_split is not None,
        )

    # --- M2 and M3: views that do not depend on a threshold ----------------------------

    def standardised_shift(self, per_feature: bool = False) -> pd.DataFrame:
        """d of every fault against healthy: |median shift| / pooled robust spread.

        Returns d' (the largest d over the features) and the feature that gives it, or
        the whole fault x feature table with `per_feature`. Univariate and
        conservative: a map of where the information is, not a detection probability.
        """
        centre, spread = self.dictionary()
        var = spread.to_numpy() ** 2
        if self.noise_floor is not None:
            var = np.maximum(var, self.noise_floor.reindex(self.features).to_numpy() ** 2)
        var = pd.DataFrame(var, index=centre.index, columns=self.features)
        pooled = np.sqrt(0.5 * (var + var.loc[self.healthy_id]))
        d = ((centre - centre.loc[self.healthy_id]).abs() / np.maximum(pooled, 1e-300))
        d = d.reindex(self.fault_ids)
        if per_feature:
            return d
        return pd.DataFrame({"d_prime": d.max(axis=1), "feature": d.idxmax(axis=1)})

    def auc(self) -> pd.DataFrame:
        """Area under the ROC curve of the best single feature, per fault.

        `auc` is max over features of max(AUC, 1 - AUC). Choosing the best of
        `n_features` is optimistic, more so with many features and few samples.
        """
        healthy = self._x[self._rows[self.healthy_id]]
        rows = []
        for fault_id in self.fault_ids:
            idx = self._rows.get(fault_id)
            if idx is None:
                rows.append({"fault_id": fault_id, "auc": np.nan, "feature": None})
                continue
            values = [auc(self._x[idx, j], healthy[:, j]) for j in range(len(self.features))]
            best = int(np.argmax([max(a, 1 - a) for a in values]))
            rows.append(
                {
                    "fault_id": fault_id,
                    "auc": max(values[best], 1 - values[best]),
                    "feature": self.features[best],
                }
            )
        table = pd.DataFrame(rows).set_index("fault_id")
        table["n_features"] = len(self.features)
        return table

    # --- M5 and M6: functional failure and diagnostic coverage -------------------------

    def _failed(self) -> np.ndarray:
        if self.compliant_column is None:
            raise ValueError("this metric needs the `compliant` column of the application")
        return ~self.samples[self.compliant_column].to_numpy(dtype=bool)

    def failure_probability(self) -> pd.DataFrame:
        """P(the circuit does not meet its specifications | condition), healthy included.

        For the healthy condition, 1 minus this probability is the yield. This is a
        conditional probability, not a failure rate in time: a campaign says nothing
        about how often a fault occurs.
        """
        failed = self._failed()
        rows = []
        for fault_id in [self.healthy_id, *self.fault_ids]:
            idx = self._rows.get(fault_id, np.array([], dtype=int))
            k, n = int(failed[idx].sum()), len(idx)
            low, high = self._interval(k, n, self.confidence)
            rows.append(
                {
                    "fault_id": fault_id,
                    "n_ok": n,
                    "failures": k,
                    "p_failure": k / n if n else np.nan,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
        return pd.DataFrame(rows).set_index("fault_id")

    def diagnostic_coverage(
        self,
        alpha: float = 0.01,
        weights: Mapping[str, float] | None = None,
        detector: Detector | None = None,
        n_boot: int = 1000,
    ) -> dict:
        """Fraction of the failures that are detected, over the fault conditions.

        DC = sum_k w_k P(detected and failed | f_k) / sum_k w_k P(failed | f_k), and
        the escape rate is 1 - DC. With equal weights (the default) this is a statement
        about the fault list, not about field behaviour; weights proportional to the
        occurrence of each fault, from a cited source, make it one.

        The false-reject rate is P(detected | compliant), over the healthy samples not
        used for the thresholds and over the fault samples that still comply.
        """
        failed, flagged = self._failed(), self.flags(alpha, detector)
        # a fault without any successful simulation cannot enter the estimate; it is counted
        present = [f for f in self.fault_ids if f in self._rows]
        w = np.array([1.0 if weights is None else float(weights[f]) for f in present])
        groups = [self._rows[f] for f in present]
        if not groups:
            raise ValueError("no fault has a successful simulation")

        def coverage(rng: np.random.Generator | None = None) -> float:
            caught = failing = 0.0
            for weight, idx in zip(w, groups, strict=True):
                if rng is not None:
                    idx = rng.choice(idx, len(idx))
                caught += weight * np.mean(flagged[idx] & failed[idx])
                failing += weight * np.mean(failed[idx])
            return caught / failing if failing else np.nan

        dc = coverage()
        low, high = bootstrap_interval(coverage, n_boot, self.confidence, self.seed)
        in_faults = np.concatenate(groups)
        healthy = self._evaluation[~failed[self._evaluation]]
        compliant = np.concatenate([healthy, in_faults[~failed[in_faults]]])
        k, n = int(flagged[compliant].sum()), len(compliant)
        return {
            "diagnostic_coverage": dc,
            "interval": (low, high),
            "escape_rate": 1.0 - dc,
            "false_reject_rate": k / n if n else np.nan,
            "false_reject_interval": self._interval(k, n, self.confidence),
            "n_failed_samples": int(failed[in_faults].sum()),
            "n_compliant_samples": n,
            "n_compliant_healthy": len(healthy),
            "n_faults": len(present),
            "n_faults_without_results": len(self.fault_ids) - len(present),
            "weights": "equal" if weights is None else "given",
            "alpha": alpha,
        }

    # --- M7b: response to the magnitude of a fault --------------------------------------

    def severity_response(self, alpha: float = 0.01, detector: Detector | None = None):
        """Detection probability of every graded fault against its magnitude.

        One row per fault that has a magnitude, with the components and type that
        define its family. Needs the fault records.
        """
        if not self.records:
            raise ValueError("this metric needs the fault records (`faults`)")
        table = self.detectability(alpha, detector).table
        rows = []
        for fault_id in self.fault_ids:
            record = self.records.get(fault_id)
            if record is None or record["magnitude"] is None:
                continue
            severity = record["severity"]
            rows.append(
                {
                    "fault_id": fault_id,
                    "fault_type": record["fault_type"],
                    "components": "+".join(record["components"]),
                    "magnitude": record["magnitude"]["value"],
                    "unit": record["magnitude"]["unit"],
                    "severity": None if severity is None else severity["value"],
                    **table.loc[fault_id, ["p_detect", "ci_low", "ci_high", "n_ok"]].to_dict(),
                }
            )
        return pd.DataFrame(rows)

    def minimum_detectable(self, alpha: float = 0.01, beta: float = 0.1, by: str = "severity"):
        """Smallest simulated severity (or magnitude) detected with probability >= 1 - beta.

        Per family (fault type and components) and direction (sign of the magnitude).
        `monotone` is False when a larger simulated value falls below 1 - beta, in
        which case the smallest value does not describe the family. The result cannot
        be finer than the grid of simulated values, which is given.
        """
        response = self.severity_response(alpha).dropna(subset=[by])
        response["direction"] = np.where(response["magnitude"] < 0, "-", "+")
        rows = []
        for (fault_type, components, direction), family in response.groupby(
            ["fault_type", "components", "direction"], sort=False
        ):
            family = family.assign(size=family[by].abs()).sort_values("size")
            detected = (family["p_detect"] >= 1.0 - beta).to_numpy()
            first = int(np.argmax(detected)) if detected.any() else None
            rows.append(
                {
                    "fault_type": fault_type,
                    "components": components,
                    "direction": direction,
                    "minimum_detectable": np.nan if first is None else family["size"].iloc[first],
                    "monotone": bool(first is not None and detected[first:].all()),
                    "grid": family["size"].tolist(),
                }
            )
        return pd.DataFrame(rows)

    # --- M9: separability ---------------------------------------------------------------

    def separation(self) -> pd.DataFrame:
        """Pairwise d' between all conditions, healthy included (M9a)."""
        centre, spread = self.dictionary()
        return pairwise_shift(centre, spread, self.noise_floor)

    def ambiguity(
        self, threshold: float = 3.0, component_of: Mapping[str, str] | None = None
    ) -> Ambiguity:
        """Which conditions, and which components, cannot be told apart (M9b).

        `component_of` maps each fault to the component to locate; by default the
        components of its record, joined by "+".
        """
        d = self.separation()
        close_to_healthy = d.loc[self.healthy_id].drop(self.healthy_id) < threshold
        if component_of is None and self.records:
            component_of = {f: "+".join(r["components"]) for f, r in self.records.items()}
        confusable = groups = None
        if component_of is not None:
            faults_only = d.drop(index=self.healthy_id, columns=self.healthy_id)
            confusable = confusable_components(faults_only, dict(component_of), threshold)
            groups = component_groups(faults_only, dict(component_of), threshold)
        return Ambiguity(
            threshold,
            list(close_to_healthy.index[close_to_healthy]),
            ambiguity_groups(d, threshold),
            confusable,
            groups,
        )


def detectability_across(
    analyses: Mapping[object, ReliabilityAnalysis],
    alpha: float = 0.01,
    beta: float = 0.1,
    detector: Detector | None = None,
) -> pd.DataFrame:
    """Detection probability of every fault in several analyses, side by side.

    The analyses are campaigns that differ in one thing: the tolerance scale (M8,
    robustness) or the operating condition (M11). Each is judged against its own
    healthy reference. Columns: one per analysis, then
    - `worst_case`: the smallest detection probability;
    - `range`: largest minus smallest;
    - `lost`: the analyses where the fault falls below 1 - beta although it is
      detected in the first one.
    """
    labels = list(analyses)
    table = pd.DataFrame(
        {k: analyses[k].detectability(alpha, detector).table["p_detect"] for k in labels}
    )
    values = table[labels]
    detected = values >= 1.0 - beta
    table["worst_case"] = values.min(axis=1)
    table["range"] = values.max(axis=1) - values.min(axis=1)
    table["lost"] = [
        [label for label in labels[1:] if not row[label]] if row[labels[0]] else []
        for _, row in detected.iterrows()
    ]
    return table


def robustness(
    analyses: Mapping[float, ReliabilityAnalysis],
    alpha: float = 0.01,
    beta: float = 0.1,
    detector: Detector | None = None,
) -> pd.DataFrame:
    """Detectability against the tolerance scale (M8).

    `analyses` maps each tolerance scale (1 is the declared design) to the analysis of
    the campaign simulated at that scale. Columns: the detection probability at each
    scale, in increasing order, then
    - `loss`: detection probability at the smallest scale minus at the largest;
    - `critical_tolerance`: the largest simulated scale at which the fault is still
      detected with probability >= 1 - beta, NaN if at none.
    """
    scales = sorted(analyses)
    table = detectability_across({k: analyses[k] for k in scales}, alpha, beta, detector)
    values = table[scales]
    table = values.copy()
    table["loss"] = values[scales[0]] - values[scales[-1]]
    detected = (values >= 1.0 - beta).to_numpy()
    table["critical_tolerance"] = [
        max((k for k, hit in zip(scales, row, strict=True) if hit), default=np.nan)
        for row in detected
    ]
    return table
