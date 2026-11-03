# Test design: FMEDA export

The people who certify a product read a spreadsheet FMEDA: one row per failure mode of
a component, with its rate, its class and the diagnostic coverage (DC) of the safety
mechanism that covers it, and totals that give the SFF of IEC 61508 or the SPFM and LFM
of ISO 26262. The DC of each row is usually taken from tables (IEC 61508-2 Annex A,
ISO 26262-5 Annex D: low 60 %, medium 90 %, high 99 %). `testdesign.fmeda` fills the
same sheet with the DC that the simulation gives for a plan, with its bounds, and keeps
the faults behind every row. The workbook needs `pip install "spicefault[fmeda]"`.

## From faults to rows

```python
from spicefault.testdesign import faultprior, fmeda

priors = faultprior.prior(records, "gf")
table = priors.join(records.set_index("fault_id")[["component", "fault_type", "effect"]])
faults = fmeda.faults(table, claim.per_fault, confidence=0.95, claim="mode",
                      tests=sampler.alarming)
rows = fmeda.rows(faults, claim="mode")
summary = fmeda.summary(rows, target={"spfm": 0.97})
```

`table` is the fault table of `faultprior.prior` with the component and fault type of
each fault, and its effect and deviation if known; `per_fault` is that of a `Detection`
or a `Certification`. `faults` gives each fault its failure mode of MIL-HDBK-338B, its
estimate and one-sided Clopper–Pearson bounds. `rows` groups them into one row per
component, failure mode, severity and class, with:

- λ = λ part × mode share × class share, in FIT;
- the effect, and the tests of the plan that alarm on its faults as safety mechanism;
- the faults and circuits behind it, and its DC: the mean of the detection of its
  faults weighted by their rates, with the same mean of their bounds;
- `dc_claimed`, and from it λ_S, λ_DD and λ_DU of IEC 61508 and λ_SPF, λ_RF, λ_MPF,D
  and λ_MPF,L of ISO 26262.

A mode is split by severity and class because one class does not fit it: the parameter
change of a resistor can be safe at +10 % and single-point at −50 %. With one row per
mode, severity and class, every metric, DC_w included, is a sum over the rows.

## Which claim the sheet carries

| `claim` | Bounds of a row | DC claimed | The claim |
|---|---|---|---|
| `"mode"` | at (1 − c)/K for the K faults, or those of `certify`, simultaneous | the lower bound | the totals of the sheet |
| `"metric"` | at 1 − c each, not simultaneous | the estimate | the lower bound of `certify_sampled`, in the Summary |

Every metric is linear in the rows, so with `claim="mode"` the totals computed with the
lower bound of each row are the `bonferroni` bound of `metrics.coverage`, or the lower
bound of `certify` when its `per_fault` is given: the number the sheet shows is the
number claimed, and an assessor can audit it row by row. A claim per row costs many
more circuits than a claim on the metric: in CERTA, 4 600 against 267 on a battery cell
chain and 9 925 against 50 on an ECG front end. Both are useful, `certify_sampled` to
decide the target and `certify` for an assessor who reads rows.

## The tests that alarm

```python
with formulation.ConditionSampler(..., record=True) as sampler:
    claim = adaptive.certify(sampler, fault_table, 0.97, "spfm")
sampler.alarming   # {fault: {tests that left their limits on any of its replicas}}
```

`SpiceSampler` and `ConditionSampler` take `record=True` to keep, per fault, the tests
that left their limits; it needs a detector with limits per test, as `GaussianLimits`
and `reliability.LimitTest` (`metrics.outside`).

## Modes not modelled

A mode that no fault models (a FET stuck low, a fuse that fails to open) is a row
`not assessed`, with its rate, no faults and DC 0. It is left out of the totals, as in
`metrics`, and the Summary gives its share of the rate and every metric with those
modes counted as single-point and undetected. On the battery cell chain of CERTA they
are 12.7 % of the rate, and counted so the SPFM falls from 0.976 to 0.852, below
ASIL C: they are rows of the sheet so that they are seen.

## The workbook

```python
fmeda.write("fmeda.xlsx", rows, faults, summary, cover={"Case": "...", ...},
            sampled=sampled_claims, assumptions={"Severity rules": "..."})
fmeda.template("fmeda_template.xlsx")
```

| Sheet | Content |
|---|---|
| Cover | The items of `cover`: case, safety goal, standard and target, plan, claim, confidence |
| FMEDA | The rows; λ and the λ of the standards are formulas of the row |
| Faults | One row per fault, with the FMEDA row it falls in and the tests that alarm |
| Summary | The metrics as formulas over the FMEDA sheet, with the estimate, bounds, worst case of the modes not assessed, target and decision; the SIL and ASIL reached; the claim of `certify_sampled` if given |
| Assumptions | Sources of the rates and modes, how a row's coverage is computed, and the items of `assumptions` |

The totals are live formulas, so changing a rate, a class or a DC moves them; outside
Excel, the workbook recalculated by `formulas` gives the values of `summary` to 1e-12.
The same content gives the same bytes. `template` writes the same workbook empty, with
the formulas on `blank` rows, for an FMEDA filled by hand with table values: it gives
the SPFM, LFM and SFF of ISO 26262-5 and IEC 61508-2. The SIL and ASIL reached come from
`metrics.sil_reached` and `metrics.asil_reached`, whose tables were taken from drafts of
the standards.

See the [example](../examples/fmeda.md), which takes the plan of the cheapest certified
plan to a workbook.
