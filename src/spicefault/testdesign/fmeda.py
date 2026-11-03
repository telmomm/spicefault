"""FMEDA of a self-test plan: one row per failure mode of a component, with the diagnostic
coverage computed by simulation and its bounds, and the totals of IEC 61508 and
ISO 26262.

From the fault table of a circuit (one row per fault: component, kind of part, fault
type, rate, severity class and classes, as `faultprior.prior` gives them with the fault
universe) and the detection of a plan (circuits and alarms per fault):

- `faults`: one row per fault with its failure mode, estimate and bounds;
- `rows`: one row per component, failure mode, severity and class, with its rate, its
  coverage and the rates of the standards (λ_S, λ_DD, λ_DU, λ_SPF, λ_RF, λ_MPF);
- `metrics` and `summary`: DC_w, DC, SFF, SPFM and LFM from the rows;
- `write`: the workbook (Cover, FMEDA, Faults, Summary, Assumptions), with the rates of
  the standards and the claimed metrics as live formulas, and `template` the same
  workbook empty, to be filled by hand with table values. Both need the extra
  `spicefault[fmeda]`.

A failure mode is that of MIL-HDBK-338B (`faultprior.modes`). A mode whose faults differ
in severity or class is split into one row for each (the parameter change of a resistor
can be safe at +10 % and single-point at −50 %), so every metric, DC_w included, is a
sum over the rows.

The bounds of a fault are one-sided Clopper–Pearson bounds. With `claim="mode"` they are
at (1 − confidence) / K for the K faults, or those of `metrics.adaptive.certify`, so they
hold for every fault at once, and the totals computed with the lower bounds are the
`bonferroni` bound of `metrics.coverage` or the lower bound of `certify`: the sheet is
the claim. With `claim="metric"` they are at 1 − confidence each and only describe the
row: the claim is made on the metric by `certify_sampled`, and the sheet claims the
estimates.

Modes of a component that the fault universe does not model are kept as rows with no
faults, class `not assessed` and DC 0, and are left out of the totals; `summary` also
gives the metrics with them counted as single-point and undetected.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from ..reliability import clopper_pearson_interval
from . import faultprior
from .faultprior import LINEAR
from .metrics.coverage import asil_reached, sil_reached

__all__ = [
    "ASSUMPTIONS",
    "CLAIMS",
    "FAULT_COLUMNS",
    "LAMBDAS",
    "METRICS",
    "NOT_ASSESSED",
    "faults",
    "metrics",
    "rows",
    "summary",
    "template",
    "write",
]

CLAIMS = ("mode", "metric")
NOT_ASSESSED = "not assessed"
METRICS = ("dc_w", "dc", "sff", "spfm", "lfm")
FAULT_COLUMNS = (
    "component", "kind", "fault_type", "rate", "severity_class", "iec", "iso",
    "component_rate",
)  # fmt: skip
LAMBDAS = (
    "lambda_s", "lambda_dd", "lambda_du", "lambda_spf", "lambda_rf", "lambda_mpf_d",
    "lambda_mpf_l",
)  # fmt: skip


def _mode_table(mode_table: pd.DataFrame | None) -> pd.DataFrame:
    table = faultprior.modes() if mode_table is None else mode_table
    return table.assign(fault_type=table.fault_type.fillna(""))


def _mode_of(kind: str, fault_type: str, table: pd.DataFrame) -> tuple[str, float]:
    """Name and probability of the failure mode that a fault type models for a kind of
    part; several modes of one fault type are joined."""
    category = faultprior.PARTS[kind][1]
    found = table[(table.category == category) & (table.fault_type == fault_type)]
    if found.empty:
        raise KeyError(f"no failure mode of {category!r} is modelled by {fault_type!r}")
    return " / ".join(found["mode"]), float(found.alpha.sum())


def _tests(tests: pd.Series | Mapping[str, str | Iterable[str]], index: pd.Index) -> pd.Series:
    """The tests that alarm on each fault, as one string of test names."""
    joined = {
        f: t if isinstance(t, str) else " ".join(sorted(t)) for f, t in dict(tests).items()
    }
    return pd.Series(joined, dtype=object).reindex(index).fillna("")


def faults(
    table: pd.DataFrame,
    per_fault: pd.DataFrame,
    confidence: float = 0.95,
    claim: str = "mode",
    tests: pd.Series | Mapping[str, str | Iterable[str]] | None = None,
    mode_table: pd.DataFrame | None = None,
    weights: Sequence[float] = LINEAR,
) -> pd.DataFrame:
    """One row per fault, indexed by fault, with its failure mode, the circuits `n`,
    the alarms `detected`, the estimate and the one-sided bounds of its detection.

    `table` has the columns of `FAULT_COLUMNS` and is indexed by fault: the fault table
    of `faultprior.prior` joined with the component and fault type of the fault
    universe, and, if known, its `effect` and `deviation`. The severity weight is the
    column `weight`, or `severity` (that of `faultprior.prior`), or otherwise
    `weights[severity_class]`. `per_fault` is that of a `Detection` or a
    `Certification` (n and detected). With `claim="mode"`, its `lower` and `upper`, when
    given, are kept: those of `certify` hold for every fault at every stage. `tests`
    gives, per fault, the tests that alarm on it, as a string or a set of names (the
    `alarming` of a sampler made with `record=True`). A fault never simulated has no
    estimate or bounds.
    """
    if claim not in CLAIMS:
        raise ValueError(f"unknown claim {claim!r}; expected {CLAIMS}")
    missing = set(per_fault.index) - set(table.index)
    if missing:
        raise KeyError(f"faults without a row in the fault table: {sorted(missing)[:5]}")
    modes = _mode_table(mode_table)
    out = table.loc[per_fault.index, list(FAULT_COLUMNS)].copy()
    for column in ("deviation", "effect"):
        given = table[column].reindex(out.index).astype(object) if column in table else None
        out[column] = given.where(given.notna(), "") if given is not None else ""
    if "weight" in table:
        out["weight"] = table.weight.reindex(out.index)
    elif "severity" in table:
        out["weight"] = table.severity.reindex(out.index)
    else:
        out["weight"] = out.severity_class.map(lambda s: weights[int(s)])
    named = [_mode_of(k, t, modes) for k, t in zip(out.kind, out.fault_type, strict=True)]
    out["mode"] = [m for m, _ in named]
    out["mode_share"] = [a for _, a in named]
    out["n"] = per_fault.n.astype(int)
    out["detected"] = per_fault.detected.astype(int)
    alpha = (1 - confidence) / len(per_fault) if claim == "mode" else 1 - confidence
    bounds = [clopper_pearson_interval(k, n, 1 - 2 * alpha) if n else (np.nan, np.nan)
              for k, n in zip(out.detected, out.n, strict=True)]  # fmt: skip
    out["estimate"] = np.where(out.n > 0, out.detected / out.n.where(out.n > 0, 1), np.nan)
    out["lower"] = [b[0] for b in bounds]
    out["upper"] = [b[1] for b in bounds]
    if claim == "mode" and {"lower", "upper"} <= set(per_fault.columns):
        out["lower"], out["upper"] = per_fault.lower, per_fault.upper
    out["tests"] = _tests(tests, out.index) if tests is not None else ""
    return out


def _lambdas(rows: pd.DataFrame, dc: pd.Series) -> pd.DataFrame:
    """Rates of IEC 61508 and ISO 26262 of each row for the coverage `dc`."""
    lam = rows["lambda"]
    dangerous, safe = rows.iec == "dangerous", rows.iec == "safe"
    single, multiple = rows.iso == "single_point", rows.iso == "multiple_point"
    return pd.DataFrame({
        "lambda_s": lam.where(safe, 0.0),
        "lambda_dd": (lam * dc).where(dangerous, 0.0),
        "lambda_du": (lam * (1 - dc)).where(dangerous, 0.0),
        "lambda_spf": lam.where(single & (dc == 0), 0.0),
        "lambda_rf": (lam * (1 - dc)).where(single & (dc > 0), 0.0),
        "lambda_mpf_d": (lam * dc).where(multiple, 0.0),
        "lambda_mpf_l": (lam * (1 - dc)).where(multiple, 0.0),
    })  # fmt: skip


def rows(
    fault_rows: pd.DataFrame, claim: str = "mode", mode_table: pd.DataFrame | None = None
) -> pd.DataFrame:
    """One row per component, failure mode, severity class and classes, from the rows of
    `faults`, plus one row per mode of a component that no fault models.

    The coverage of a row is the mean over its simulated faults, weighted by their
    rates, of their estimates and bounds; `dc_claimed` is the lower bound with
    `claim="mode"` and the estimate with `claim="metric"`, and 0 for a row with no
    simulated fault. The λ columns are computed with `dc_claimed`.
    """
    if claim not in CLAIMS:
        raise ValueError(f"unknown claim {claim!r}; expected {CLAIMS}")
    modes = _mode_table(mode_table)
    keys = ["component", "kind", "mode", "severity_class", "iec", "iso"]
    out = []
    for key, g in fault_rows.groupby(keys, sort=False, dropna=False):
        row = dict(zip(keys, key, strict=True))
        row["weight"] = g.weight.iloc[0]
        lam = g.rate.sum()
        simulated = g[g.n > 0]
        w = simulated.rate if simulated.rate.sum() > 0 else pd.Series(1.0, index=simulated.index)
        mean = {c: float(np.average(simulated[c], weights=w)) if len(simulated) else np.nan
                for c in ("estimate", "lower", "upper")}  # fmt: skip
        part = g.component_rate.iloc[0] * g.mode_share.iloc[0]
        tests = sorted({t for ts in g.tests for t in str(ts).split()})
        out.append({
            **row, "component_rate": g.component_rate.iloc[0],
            "mode_share": g.mode_share.iloc[0], "class_share": lam / part if part else np.nan,
            "lambda": lam, "effect": "; ".join(dict.fromkeys(g.effect.astype(str))),
            "safety_mechanism": " ".join(tests), "dc_basis": "computed",
            "faults": len(g), "circuits": int(g.n.sum()), **mean,
        })  # fmt: skip
    found = pd.DataFrame(out)
    # modes that no fault of the component models
    extra = []
    for (component, kind), g in found.groupby(["component", "kind"], sort=False):
        category = faultprior.PARTS[kind][1]
        modelled = set(fault_rows.fault_type[fault_rows.component == component])
        for m in modes[modes.category == category].itertuples():
            if m.fault_type in modelled:
                continue
            extra.append({
                "component": component, "kind": kind, "mode": m.mode,
                "severity_class": np.nan, "weight": np.nan,
                "iec": NOT_ASSESSED, "iso": NOT_ASSESSED,
                "component_rate": g.component_rate.iloc[0], "mode_share": m.alpha,
                "class_share": 1.0, "lambda": g.component_rate.iloc[0] * m.alpha,
                "effect": "not modelled", "safety_mechanism": "", "dc_basis": "none",
                "faults": 0, "circuits": 0, "estimate": np.nan, "lower": np.nan,
                "upper": np.nan,
            })  # fmt: skip
    result = pd.concat([found, pd.DataFrame(extra)], ignore_index=True) if extra else found
    order = {c: i for i, c in enumerate(dict.fromkeys(result.component))}
    result = result.sort_values(
        ["component", "mode", "severity_class"],
        key=lambda s: s.map(order) if s.name == "component" else s,
        kind="stable",
    ).reset_index(drop=True)
    claimed = result.lower if claim == "mode" else result.estimate
    result["dc_claimed"] = claimed.fillna(0.0).where(result.iec != NOT_ASSESSED, 0.0)
    return pd.concat([result, _lambdas(result, result.dc_claimed)], axis=1)


def metrics(
    rows: pd.DataFrame,
    dc: str = "dc_claimed",
    weights: Sequence[float] = LINEAR,
    not_assessed: str | None = None,
) -> dict[str, float]:
    """DC_w, DC, SFF, SPFM and LFM of the rows with the coverage in column `dc`.

    Rows not assessed are left out, or counted with DC 0 as the ISO 26262 class
    `not_assessed` (`single_point` for the worst case), dangerous, and with the largest
    of `weights`. NaN when a metric has no row of its kind.
    """
    r = rows.copy()
    na = r.iec == NOT_ASSESSED
    if not_assessed is None:
        r, na = r[~na], na[~na]
    else:
        r.loc[na, "iso"], r.loc[na, "iec"] = not_assessed, "dangerous"
        r.loc[na, "weight"] = max(weights)
    p = r[dc].fillna(0.0).where(~na, 0.0)
    lam = r["lambda"]
    rates = _lambdas(r, p)
    weight = lam * r.weight
    total = lam.sum()
    residual = rates.lambda_spf.sum() + rates.lambda_rf.sum()
    lam_d = rates.lambda_dd.sum() + rates.lambda_du.sum()
    lam_s = rates.lambda_s.sum()

    def ratio(a, b):
        return float(a / b) if b > 0 else np.nan

    single, multiple = (r.iso == "single_point").any(), (r.iso == "multiple_point").any()
    return {
        "dc_w": ratio((weight * p).sum(), weight.sum()),
        "dc": ratio(rates.lambda_dd.sum(), lam_d),
        "sff": ratio(lam_s + rates.lambda_dd.sum(), lam_s + lam_d),
        "spfm": 1 - ratio(residual, total) if single else np.nan,
        "lfm": 1 - ratio(rates.lambda_mpf_l.sum(), total - residual) if multiple else np.nan,
    }


def summary(
    rows: pd.DataFrame,
    target: Mapping[str, float] | None = None,
    weights: Sequence[float] = LINEAR,
) -> pd.DataFrame:
    """Each metric with the claimed value, the estimate and the bounds of the rows, the
    worst case of the rows not assessed (single point, DC 0), the target and whether the
    claimed value meets it."""
    columns = {"claimed": "dc_claimed", "estimate": "estimate", "lower": "lower",
               "upper": "upper"}  # fmt: skip
    out = pd.DataFrame({name: metrics(rows, c, weights) for name, c in columns.items()})
    worst = metrics(rows, "dc_claimed", weights, "single_point")
    out["not assessed as single point"] = pd.Series(worst)
    out["target"] = pd.Series(dict(target or {}), dtype=float).reindex(out.index)
    out["met"] = (out.claimed >= out.target).where(out.target.notna())
    return out.rename_axis("metric").loc[list(METRICS)]


# --- workbook ---------------------------------------------------------------------------

NAVY = "0B2D4A"
FIT, SHARE, PCT = "0.000", "0.0%", "0.00%"
LABELS = {"single_point": "single point", "multiple_point": "multiple point"}
NAMES = {"dc_w": "DC_w", "dc": "DC", "sff": "SFF", "spfm": "SPFM", "lfm": "LFM"}
# header, column of `rows` or formula of row {r}, number format; the letters of the
# formulas are those of this order (G is λ, J the weight, K and L the classes, T the DC)
FMEDA_SHEET = (
    ("Ref", "component", None),
    ("Part", "kind", None),
    ("λ part (FIT)", "component_rate", FIT),
    ("Failure mode", "mode", None),
    ("Mode share", "mode_share", SHARE),
    ("Class share", "class_share", SHARE),
    ("λ (FIT)", "=C{r}*E{r}*F{r}", FIT),
    ("Effect", "effect", None),
    ("Severity", "severity_class", "0"),
    ("Weight", "weight", "0.##"),
    ("IEC 61508", "iec", None),
    ("ISO 26262", "iso", None),
    ("Safety mechanism", "safety_mechanism", None),
    ("DC basis", "dc_basis", None),
    ("Faults", "faults", "0"),
    ("Circuits", "circuits", "0"),
    ("DC estimate", "estimate", PCT),
    ("DC lower", "lower", PCT),
    ("DC upper", "upper", PCT),
    ("DC claimed", "dc_claimed", PCT),
    ("λ_S", '=IF(K{r}="safe",G{r},0)', FIT),
    ("λ_DD", '=IF(K{r}="dangerous",G{r}*T{r},0)', FIT),
    ("λ_DU", '=IF(K{r}="dangerous",G{r}*(1-T{r}),0)', FIT),
    ("λ_SPF", '=IF(AND(L{r}="single point",T{r}=0),G{r},0)', FIT),
    ("λ_RF", '=IF(AND(L{r}="single point",T{r}>0),G{r}*(1-T{r}),0)', FIT),
    ("λ_MPF,D", '=IF(L{r}="multiple point",G{r}*T{r},0)', FIT),
    ("λ_MPF,L", '=IF(L{r}="multiple point",G{r}*(1-T{r}),0)', FIT),
)
FAULTS_SHEET = (
    ("Fault", "fault_id", None),
    ("FMEDA row", "row", "0"),
    ("Ref", "component", None),
    ("Failure mode", "mode", None),
    ("Fault type", "fault_type", None),
    ("Deviation", "deviation", None),
    ("Effect", "effect", None),
    ("Severity", "severity_class", "0"),
    ("IEC 61508", "iec", None),
    ("ISO 26262", "iso", None),
    ("λ (FIT)", "rate", FIT),
    ("Circuits", "n", "0"),
    ("Alarms", "detected", "0"),
    ("Estimate", "estimate", PCT),
    ("Lower", "lower", PCT),
    ("Upper", "upper", PCT),
    ("Tests that alarm", "tests", None),
)
ASSUMPTIONS = {
    "Failure rates": "Parts count method of MIL-HDBK-217F Notice 2, Appendix A, in FIT; "
    "pessimistic for current parts. Ratio metrics (DC, SFF, SPFM, LFM, DC_w) do not depend "
    "on their scale; PMHF and PFH need IEC 61709 or SN 29500.",
    "Failure modes": "Mode probabilities of MIL-HDBK-338B, Table 7.8-1. Modes that the fault "
    "universe does not model are rows 'not assessed', with DC 0, left out of the totals.",
    "Coverage of a row": "Mean of the detection of its faults weighted by their rates, with "
    "one-sided Clopper–Pearson bounds per fault.",
}
TEMPLATE_COVER = (
    "Case", "Safety goal", "Standard and target", "Safety mechanisms", "DC basis",
    "Confidence", "Environment and source of the rates", "Prepared by",
)  # fmt: skip


def _openpyxl():
    try:
        import openpyxl
    except ImportError as error:
        raise ImportError(
            'fmeda.write and fmeda.template need openpyxl: pip install "spicefault[fmeda]"'
        ) from error
    return openpyxl


def _letter(i: int) -> str:
    return _openpyxl().utils.get_column_letter(i)


def _cell(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, np.generic):
        value = value.item()
    return LABELS.get(value, value) if isinstance(value, str) else value


def _header(cell) -> None:
    styles = _openpyxl().styles
    cell.font = styles.Font(bold=True, color="FFFFFF")
    cell.fill = styles.PatternFill("solid", fgColor=NAVY)


def _table(ws, columns, records: Sequence[Mapping], blank: int = 0,
           widths: Mapping[str, int] | None = None) -> None:  # fmt: skip
    """Header and rows of a sheet; a column whose source starts with "=" is a formula of
    the row; `blank` rows more with only the formulas, for a template."""
    for j, (header, _, _) in enumerate(columns, start=1):
        _header(ws.cell(1, j, header))
        width = (widths or {}).get(header, max(10, len(header) + 2))
        ws.column_dimensions[_letter(j)].width = width
    for i, rec in enumerate(list(records) + [{}] * blank, start=2):
        for j, (_, source, fmt) in enumerate(columns, start=1):
            value = source.format(r=i) if source.startswith("=") else _cell(rec.get(source))
            c = ws.cell(i, j, value)
            if fmt:
                c.number_format = fmt
    ws.freeze_panes = "B2"
    ws.auto_filter.ref = f"A1:{_letter(len(columns))}{max(2, len(records) + blank + 1)}"


def _pairs(ws, title: str, items: Mapping[str, object]) -> None:
    styles = _openpyxl().styles
    ws.cell(1, 1, title).font = styles.Font(bold=True, size=14)
    for i, (key, value) in enumerate(items.items(), start=3):
        ws.cell(i, 1, key).font = styles.Font(bold=True)
        ws.cell(i, 2, _cell(value)).alignment = styles.Alignment(wrap_text=True, vertical="top")
    ws.column_dimensions["A"].width, ws.column_dimensions["B"].width = 30, 100


def _summary_sheet(ws, last: int, summary: pd.DataFrame | None, levels: Mapping[str, object],
                   sampled: pd.DataFrame | None) -> None:  # fmt: skip
    """Metrics as formulas over the FMEDA sheet (rows 2 to `last`), and the values that
    only the simulation gives (estimate, bounds, worst case of the rows not assessed)."""
    bold = _openpyxl().styles.Font(bold=True)
    col = {h: _letter(j) for j, (h, _, _) in enumerate(FMEDA_SHEET, start=1)}

    def rng(h):
        return f"FMEDA!${col[h]}$2:${col[h]}${last}"

    def s(h):
        return f"SUM({rng(h)})"

    lam = f'SUMIF({rng("IEC 61508")},"<>{NOT_ASSESSED}",{rng("λ (FIT)")})'
    residual = f'({s("λ_SPF")}+{s("λ_RF")})'
    formulas = {
        "dc_w": f'=IFERROR(SUMPRODUCT({rng("λ (FIT)")},{rng("Weight")},{rng("DC claimed")})'
        f'/SUMPRODUCT({rng("λ (FIT)")},{rng("Weight")}),"")',
        "dc": f'=IFERROR({s("λ_DD")}/({s("λ_DD")}+{s("λ_DU")}),"")',
        "sff": f'=IFERROR(({s("λ_S")}+{s("λ_DD")})/({s("λ_S")}+{s("λ_DD")}+{s("λ_DU")}),"")',
        "spfm": f'=IF(COUNTIF({rng("ISO 26262")},"single point")=0,"",1-{residual}/{lam})',
        "lfm": f'=IF(COUNTIF({rng("ISO 26262")},"multiple point")=0,"",'
        f'1-{s("λ_MPF,L")}/({lam}-{residual}))',
    }
    headers = ("Metric", "Claimed", "Estimate", "Lower", "Upper",
               "Not assessed as single point", "Target", "Met")  # fmt: skip
    for j, h in enumerate(headers, start=1):
        _header(ws.cell(1, j, h))
        ws.column_dimensions[_letter(j)].width = max(12, len(h) + 2)
    ws.column_dimensions["A"].width = 44
    given = ("estimate", "lower", "upper", "not assessed as single point", "target")
    for i, m in enumerate(METRICS, start=2):
        ws.cell(i, 1, NAMES[m])
        ws.cell(i, 2, formulas[m]).number_format = PCT
        if summary is not None:
            for j, c in enumerate(given, start=3):
                ws.cell(i, j, _cell(summary.loc[m, c])).number_format = PCT
        ws.cell(i, 8, f'=IF(OR(G{i}="",B{i}=""),"",IF(B{i}>=G{i},"yes","no"))')
    r = len(METRICS) + 3
    ws.cell(r, 1, "λ assessed (FIT)")
    ws.cell(r, 2, f"={lam}").number_format = FIT
    ws.cell(r + 1, 1, "λ not assessed (FIT)")
    unassessed = f'=SUMIF({rng("IEC 61508")},"{NOT_ASSESSED}",{rng("λ (FIT)")})'
    ws.cell(r + 1, 2, unassessed).number_format = FIT
    ws.cell(r + 2, 1, "Share of λ not assessed")
    ws.cell(r + 2, 2, f'=IFERROR(B{r + 1}/(B{r}+B{r + 1}),"")').number_format = PCT
    for k, (key, value) in enumerate(levels.items(), start=r + 3):
        ws.cell(k, 1, key)
        ws.cell(k, 2, _cell(value))
    if sampled is not None and len(sampled):
        k = r + 4 + len(levels)
        ws.cell(k, 1, "Claim at metric level (certify_sampled)").font = bold
        for j, h in enumerate(sampled.columns, start=1):
            ws.cell(k + 1, j, h).font = bold
        for i, rec in enumerate(sampled.itertuples(index=False), start=k + 2):
            for j, v in enumerate(rec, start=1):
                c = ws.cell(i, j, _cell(v))
                if isinstance(v, float) and 0 <= v <= 1:
                    c.number_format = PCT


def _workbook(cover: Mapping[str, object], rows_: pd.DataFrame | None,
              faults_: pd.DataFrame | None, summary_: pd.DataFrame | None,
              levels: Mapping[str, object], sampled: pd.DataFrame | None,
              assumptions: Mapping[str, object], blank: int = 0):  # fmt: skip
    wb = _openpyxl().Workbook()
    _pairs(wb.active, "FMEDA", cover)
    wb.active.title = "Cover"
    records = [] if rows_ is None else rows_.to_dict("records")
    _table(wb.create_sheet("FMEDA"), FMEDA_SHEET, records, blank,
           {"Effect": 40, "Safety mechanism": 40, "Failure mode": 18})  # fmt: skip
    if faults_ is not None:
        widths = {"Fault": 26, "Effect": 40, "Tests that alarm": 40, "Failure mode": 18}
        records_ = faults_.reset_index().to_dict("records")
        _table(wb.create_sheet("Faults"), FAULTS_SHEET, records_, widths=widths)
    last = len(records) + blank + 1
    _summary_sheet(wb.create_sheet("Summary"), last, summary_, levels, sampled)
    _pairs(wb.create_sheet("Assumptions"), "Assumptions", assumptions)
    return wb


def _save(wb, path: str | Path) -> Path:
    """Save with fixed dates, in the document properties and in the zip, so that the same
    content gives the same file."""
    import io
    import re
    import zipfile
    from datetime import datetime

    wb.properties.created = wb.properties.modified = datetime(2000, 1, 1)
    buffer = io.BytesIO()
    wb.save(buffer)
    path = Path(path)
    fixed = rb"\g<1>2000-01-01T00:00:00Z"
    with zipfile.ZipFile(buffer) as src, zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as dst:
        for item in src.infolist():
            data = src.read(item.filename)
            if item.filename == "docProps/core.xml":
                data = re.sub(rb"(<dcterms:modified[^>]*>)[^<]*", fixed, data)
            info = zipfile.ZipInfo(item.filename, (1980, 1, 1, 0, 0, 0))
            dst.writestr(info, data, zipfile.ZIP_DEFLATED)
    return path


def _row_numbers(rows_: pd.DataFrame, faults_: pd.DataFrame) -> pd.Series:
    """The FMEDA sheet row (counting the header as 1) of each fault."""
    keys = ["component", "mode", "severity_class", "iec", "iso"]
    numbered = rows_[keys].assign(row=np.arange(2, len(rows_) + 2))
    found = faults_.reset_index().merge(numbered, on=keys, how="left").set_index("fault_id")
    return found.row


def write(
    path: str | Path,
    rows_: pd.DataFrame,
    faults_: pd.DataFrame,
    summary_: pd.DataFrame,
    cover: Mapping[str, object],
    sampled: pd.DataFrame | None = None,
    assumptions: Mapping[str, object] | None = None,
) -> Path:
    """Write the workbook of an FMEDA computed by `rows`, `faults` and `summary`.

    `cover` gives the case, safety goal, target, plan, claim and confidence as items;
    `sampled` the claim at metric level (one row per metric) when the claim is made by
    `certify_sampled`; `assumptions` items added to `ASSUMPTIONS`. The SIL (Route 1H,
    type B, HFT 0) and the ASIL reached are computed from the claimed values. The same
    content gives the same file.
    """
    claimed = summary_.claimed
    levels = {}
    if not np.isnan(claimed["sff"]):
        key = "SIL reached by SFF (IEC 61508-2 Route 1H, type B, HFT 0)"
        levels[key] = sil_reached(claimed["sff"])
    if not np.isnan(claimed["spfm"]):
        lfm = None if np.isnan(claimed["lfm"]) else claimed["lfm"]
        key = "ASIL reached by SPFM" + (" and LFM" if lfm is not None else "")
        levels[key] = asil_reached(claimed["spfm"], lfm)
    numbered = faults_.assign(row=_row_numbers(rows_, faults_))
    notes = {**ASSUMPTIONS, **(assumptions or {})}
    return _save(_workbook(cover, rows_, numbered, summary_, levels, sampled, notes), path)


def template(path: str | Path, blank: int = 200) -> Path:
    """Write the empty workbook: the FMEDA sheet with `blank` rows that carry only the
    formulas of the rates of the standards, and the Summary over them. A table FMEDA
    fills Ref to Class share, Effect to DC basis and DC claimed by hand."""
    notes = ("Failure rates", "Failure modes", "Coverage of a row", "Severity rules",
             "Not modelled")  # fmt: skip
    cover = dict.fromkeys(TEMPLATE_COVER)
    return _save(_workbook(cover, None, None, None, {}, None, dict.fromkeys(notes), blank), path)
