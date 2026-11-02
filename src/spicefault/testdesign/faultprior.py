"""Prior of each fault: failure rate, severity weight and classes of the standards.

It turns a fault universe (one row per fault: component, kind of part, fault type and
deviation) into the fault table that `metrics.fault_table` takes:

- rate: the failure rate of the component (generic rate of its category in the
  environment, times its quality factor), split over its failure modes by their mode
  probabilities, and within a mode equally over the faults that model it (for example
  the four deviation levels of a parametric fault);
- severity: a weight w_k from the severity class of the fault (0 to 3);
- iec and iso: the classes of IEC 61508 and ISO 26262, from the severity and the effect.

Rates and mode probabilities are read from two tables, shipped with the package and
replaceable by passing another path to `rates` or `modes`:

- `rates.csv`: generic failure rates of the parts count method, in failures per 10^6
  hours, for the ground benign (gb), ground fixed (gf) and ground mobile (gm)
  environments, with the quality factor pi_Q; from MIL-HDBK-217F Notice 2 (1995),
  Appendix A.
- `modes.csv`: failure mode probabilities alpha of each kind of part; from
  MIL-HDBK-338B (1998), Table 7.8-1. A mode with no `fault_type` is not modelled by
  the fault universe.

Both are U.S. government works. Their absolute rates are pessimistic for current parts:
ratio metrics (DC, SFF, SPFM, LFM, DC_w) do not depend on the scale of the rates, but
absolute ones such as PMHF and PFH need the tables of IEC 61709 or SN 29500, which
cannot be redistributed and can be passed as a rate table of the same form.

Modes that the fault universe does not model (a fuse that fails to open, a FET stuck
low) are left out, and `modelled_share` says how much of each component's rate the
faults represent.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

__all__ = [
    "ENVIRONMENTS",
    "LINEAR",
    "MODES",
    "PARTS",
    "RATES",
    "classes",
    "component_rate",
    "fault_rates",
    "mode_shares",
    "modes",
    "prior",
    "rates",
]

DATA = Path(__file__).parent / "data"
RATES, MODES = DATA / "rates.csv", DATA / "modes.csv"
ENVIRONMENTS = ("gb", "gf", "gm")

# kind of part in a fault universe -> (rate category, mode category)
PARTS = {
    "resistor": ("resistor", "resistor"),
    "capacitor": ("capacitor", "capacitor"),
    "zener": ("zener", "zener"),
    "switch": ("fet", "fet"),
    "fuse": ("fuse", "fuse"),
    "ntc": ("thermistor", "transducer"),
    "strain_gauge": ("resistor", "transducer"),
}

# severity weight of each class 0..3
LINEAR = (0.0, 1.0, 2.0, 3.0)


def rates(path: str | Path = RATES) -> pd.DataFrame:
    """The rate table, indexed by category."""
    return pd.read_csv(path).set_index("category")


def modes(path: str | Path = MODES) -> pd.DataFrame:
    """The failure mode table; an empty `fault_type` marks a mode not modelled."""
    return pd.read_csv(path, keep_default_na=False)


def component_rate(category: str, environment: str, table: pd.DataFrame | None = None) -> float:
    """Failure rate of one component in FIT (failures per 10^9 hours): the generic rate
    of its category, in failures per 10^6 hours, times its quality factor."""
    if environment not in ENVIRONMENTS:
        raise ValueError(f"environment {environment!r} not in {ENVIRONMENTS}")
    row = (rates() if table is None else table).loc[category]
    return float(row[environment] * row.pi_q * 1000)


def mode_shares(category: str, table: pd.DataFrame | None = None) -> dict[str, float]:
    """Probability of each fault type for a mode category; modes not modelled are left
    out, so the shares add up to at most one."""
    table = modes() if table is None else table
    rows = table[(table.category == category) & (table.fault_type != "")]
    if rows.empty and category not in set(table.category):
        raise KeyError(f"no failure modes for {category!r}")
    return rows.groupby("fault_type").alpha.sum().to_dict()


def fault_rates(
    faults: pd.DataFrame,
    environment: str,
    parts: Mapping[str, str] | None = None,
    rate_table: pd.DataFrame | None = None,
    mode_table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Rate of each fault in FIT, indexed by `fault_id`.

    `faults` has the columns fault_id, component, part and fault_type; `parts` maps a
    component to a kind of part (a key of `PARTS`) where its `part` is not specific
    enough (a sensor that is an NTC or a strain gauge). The result also gives, per
    fault, the rate of its component and the share of that rate its faults model.
    """
    rate_table = rates() if rate_table is None else rate_table
    mode_table = modes() if mode_table is None else mode_table
    parts = dict(parts or {})
    kind = pd.Series(
        [parts.get(c, p) for c, p in zip(faults.component, faults.part, strict=True)],
        index=faults.index,
    )
    unknown = sorted(set(kind) - set(PARTS))
    if unknown:
        raise KeyError(f"kinds of part with no category: {unknown}")
    out = faults[["fault_id", "component", "fault_type"]].assign(kind=kind.to_numpy())
    # how many faults of each component model each fault type
    count = out.groupby(["component", "fault_type"]).fault_id.transform("size")
    rows = []
    for (_, k), group in out.groupby(["component", "kind"], sort=False):
        rate_category, mode_category = PARTS[k]
        lam = component_rate(rate_category, environment, rate_table)
        shares = mode_shares(mode_category, mode_table)
        modelled = sum(shares.get(t, 0.0) for t in set(group.fault_type))
        for i, f in group.iterrows():
            rows.append({"fault_id": f.fault_id, "kind": k, "component_rate": lam,
                         "rate": lam * shares.get(f.fault_type, 0.0) / count[i],
                         "modelled_share": modelled})  # fmt: skip
    return pd.DataFrame(rows).set_index("fault_id").loc[faults.fault_id]


def classes(
    severity: pd.Series, effect: pd.Series, latent: Sequence[str] = ("self-test only",)
) -> pd.DataFrame:
    """IEC 61508 and ISO 26262 classes from the severity class and the effect of each
    fault.

    Severity 3 violates the safety goal: dangerous, and single-point unless detected.
    A fault whose only effect is one of `latent` (on the self-test itself) is latent:
    multiple-point. The rest are safe.
    """
    dangerous = severity == 3
    iec = dangerous.map({True: "dangerous", False: "safe"})
    iso = pd.Series("safe", index=severity.index)
    iso[effect.isin(latent)] = "multiple_point"
    iso[dangerous] = "single_point"
    return pd.DataFrame({"iec": iec, "iso": iso})


def prior(
    faults: pd.DataFrame,
    environment: str,
    weights: Sequence[float] = LINEAR,
    parts: Mapping[str, str] | None = None,
    rate_table: pd.DataFrame | None = None,
    mode_table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """The fault table of `metrics.fault_table` (rate, severity, iec, iso), indexed by
    fault.

    `faults` is as for `fault_rates`, with two more columns: `severity`, the class 0 to
    3, and `effect`. The `severity` of the result is the weight `weights[class]`, and
    `severity_class` keeps the class; `kind`, `component_rate` and `modelled_share` come
    from `fault_rates`.
    """
    r = fault_rates(faults, environment, parts, rate_table, mode_table)
    table = faults.set_index("fault_id")
    c = classes(table.severity, table.effect)
    return pd.DataFrame({
        "rate": r.rate,
        "severity": table.severity.map(lambda s: weights[int(s)]),
        "iec": c.iec,
        "iso": c.iso,
        "severity_class": table.severity,
        "kind": r.kind,
        "component_rate": r.component_rate,
        "modelled_share": r.modelled_share,
    })  # fmt: skip
