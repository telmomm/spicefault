"""Coverage metrics of a set of tests, with bounds that hold for all faults at once.

From the failure rate, the severity and the classes of each fault (`fault_table`) and
the detection probability of each fault (`detection.detect`):

- DC_w, the risk-weighted diagnostic coverage;
- DC and SFF, the diagnostic coverage and safe failure fraction of IEC 61508-2 Annex C;
- SPFM and LFM, the single-point and latent fault metrics of ISO 26262-5 Annex C;
- the SIL (IEC 61508-2, Route 1H) and the ASIL (ISO 26262-5) that a value reaches.

Every metric is non-decreasing in each detection probability, so plugging in per-fault
bounds that hold simultaneously gives bounds on the metric that hold too.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

from ...reliability import clopper_pearson_interval
from .detection import Detection

IEC_CLASSES = ("safe", "dangerous")
ISO_CLASSES = ("safe", "single_point", "multiple_point")
METRICS = ("dc_w", "dc", "sff", "spfm", "lfm")


def fault_table(
    fault_ids: Sequence[str],
    rate: float | pd.Series = 1.0,
    severity: float | pd.Series = 1.0,
    iec: str | pd.Series = "dangerous",
    iso: str | pd.Series = "single_point",
) -> pd.DataFrame:
    """Failure rate (any unit, e.g. FIT), severity and classes of each fault.

    `iec` is safe or dangerous (IEC 61508); `iso` is safe, single_point (violates the
    safety goal on its own unless detected) or multiple_point (only together with
    another fault, so latent unless detected) (ISO 26262). Each is one value for every
    fault or a Series indexed by fault. The defaults weigh every fault equally and
    count it as dangerous and single-point.
    """
    table = pd.DataFrame(index=pd.Index(list(fault_ids), name="fault_id"))
    for column, value in (("rate", rate), ("severity", severity), ("iec", iec), ("iso", iso)):
        table[column] = value.reindex(table.index) if isinstance(value, pd.Series) else value
    numbers = table[["rate", "severity"]]
    if numbers.isna().any().any() or (numbers < 0).any().any():
        raise ValueError("every fault needs a non-negative rate and severity")
    for column, allowed in (("iec", IEC_CLASSES), ("iso", ISO_CLASSES)):
        wrong = set(table[column]) - set(allowed)
        if wrong:
            raise ValueError(f"unknown {column} classes {sorted(wrong)}; expected {allowed}")
    return table


def _ratio(num, den):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def metrics_array(P: np.ndarray, faults: pd.DataFrame) -> dict[str, np.ndarray]:
    """The metrics for detection probabilities `P`, an array [draws, faults] whose columns
    are the rows of `faults`, in that order.
    """
    rate = faults.rate.to_numpy(float)
    weight = rate * faults.severity.to_numpy(float)
    dangerous, safe = (faults.iec == "dangerous").to_numpy(), (faults.iec == "safe").to_numpy()
    single = (faults.iso == "single_point").to_numpy()
    multiple = (faults.iso == "multiple_point").to_numpy()
    lam_d, lam_s = rate[dangerous].sum(), rate[safe].sum()
    lam_dd = (P[:, dangerous] * rate[dangerous]).sum(axis=1)
    residual = ((1 - P[:, single]) * rate[single]).sum(axis=1)
    latent = ((1 - P[:, multiple]) * rate[multiple]).sum(axis=1)
    nan = np.full(len(P), np.nan)
    return {
        "dc_w": _ratio((P * weight).sum(axis=1), weight.sum()),
        "dc": _ratio(lam_dd, lam_d),
        "sff": _ratio(lam_s + lam_dd, lam_s + lam_d),
        "spfm": 1 - _ratio(residual, rate.sum()) if single.any() else nan,
        "lfm": 1 - _ratio(latent, rate.sum() - residual) if multiple.any() else nan,
    }


def metric_values(p: pd.Series, faults: pd.DataFrame) -> dict[str, float]:
    """The metrics for detection probabilities `p` (indexed by fault).

    - dc_w = Σ λ_k w_k P_k / Σ λ_k w_k, with λ_k the rate and w_k the severity
    - dc = λ_DD / λ_D, with λ_DD = Σ over dangerous faults of λ_k P_k
    - sff = (λ_S + λ_DD) / (λ_S + λ_D)
    - spfm = 1 − Σ over single-point faults of λ_k (1 − P_k) / Σ λ_k
    - lfm = 1 − Σ over multiple-point faults of λ_k (1 − P_k) /
      (Σ λ_k − Σ over single-point faults of λ_k (1 − P_k))

    NaN when a metric has no fault of its kind.
    """
    found = metrics_array(p.to_numpy(float)[None, :], faults.loc[p.index])
    return {m: float(v[0]) for m, v in found.items()}


def coverage(
    detection: Detection,
    faults: pd.DataFrame,
    confidence: float = 0.95,
    method: str = "bonferroni",
    draws: int = 20000,
    seed: int = 0,
) -> pd.DataFrame:
    """Each metric with a one-sided lower and upper bound at `confidence`.

    - `bonferroni`: per-fault one-sided Clopper–Pearson bounds at (1 − confidence) / K,
      for the K faults, plugged into the metric. They hold for all faults at once, so
      the bounds of the metric hold at least at `confidence`. Conservative.
    - `jeffreys`: quantiles of the metric over `draws` draws of each P_k from its
      Jeffreys posterior, Beta(x + 1/2, n − x + 1/2). Less conservative, Bayesian.
    """
    per_fault = detection.per_fault
    missing = set(per_fault.index) - set(faults.index)
    if missing:
        raise KeyError(f"faults without a row in the fault table: {sorted(missing)[:5]}")
    k, n = per_fault.detected, per_fault.n
    estimate = metric_values(per_fault.p, faults)
    if method == "bonferroni":
        two_sided = 1 - 2 * (1 - confidence) / len(per_fault)
        bounds = [clopper_pearson_interval(ki, ni, two_sided) for ki, ni in zip(k, n, strict=True)]
        low = metric_values(pd.Series([b[0] for b in bounds], index=per_fault.index), faults)
        high = metric_values(pd.Series([b[1] for b in bounds], index=per_fault.index), faults)
    elif method == "jeffreys":
        rng = np.random.default_rng(seed)
        P = rng.beta(k.to_numpy() + 0.5, (n - k).to_numpy() + 0.5, size=(draws, len(per_fault)))
        found = metrics_array(P, faults.loc[per_fault.index])
        low = {m: float(np.quantile(v, 1 - confidence)) for m, v in found.items()}
        high = {m: float(np.quantile(v, confidence)) for m, v in found.items()}
    else:
        raise ValueError(f"unknown method {method!r}; expected bonferroni or jeffreys")
    return pd.DataFrame(
        [
            {"metric": m, "estimate": estimate[m], "lower": low[m], "upper": high[m]}
            for m in METRICS
        ]
    ).set_index("metric")


# --- targets ---------------------------------------------------------------------------

#: (SPFM, LFM) targets by ASIL, ISO 26262-5 Table E.1. Taken from ISO/DIS 26262-5: to be
#: checked against the edition in force.
ASIL_TARGETS = {"B": (0.90, 0.60), "C": (0.97, 0.80), "D": (0.99, 0.90)}
_SFF_BANDS = (0.60, 0.90, 0.99)
#: Maximum SIL by SFF band (< 60, 60–90, 90–99, ≥ 99 %) and hardware fault tolerance 0, 1
#: and 2, for type A and type B elements, IEC 61508-2 Tables 2 and 3 (Route 1H). Taken
#: from prEN IEC 61508-2: to be checked against the edition in force.
SIL_ROUTE_1H = {
    "A": ((1, 2, 3), (2, 3, 4), (3, 4, 4), (3, 4, 4)),
    "B": ((0, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4, 4)),
}


def asil_reached(spfm: float, lfm: float | None = None) -> str:
    """The highest ASIL whose SPFM (and LFM, when given) targets the values reach;
    QM-A if none. The targets are in `ASIL_TARGETS`, taken from a draft.
    """
    reached = "QM-A"
    for asil, (spfm_target, lfm_target) in ASIL_TARGETS.items():
        lfm_ok = lfm is None or (not math.isnan(lfm) and lfm > lfm_target)
        if spfm > spfm_target and lfm_ok:
            reached = asil
    return reached


def sil_reached(sff: float, hft: int = 0, element_type: str = "B") -> int:
    """The maximum SIL that an SFF allows for a subsystem (Route 1H); 0 if not allowed.
    The table is `SIL_ROUTE_1H`, taken from a draft.
    """
    band = sum(sff >= edge for edge in _SFF_BANDS)
    return SIL_ROUTE_1H[element_type][band][hft]
