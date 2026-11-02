# Test design: fault priors

The coverage metrics of IEC 61508 and ISO 26262 weight each fault by its failure rate,
and DC_w also by its severity. If every study splits the rates of its components into
modes and deviation levels its own way, their coverages cannot be compared.
`testdesign.faultprior` does it one way, from two public tables.

```python
from spicefault.testdesign import faultprior, metrics

priors = faultprior.prior(records, "gf", parts={"S1": "ntc"})
table = metrics.fault_table(priors.index, priors.rate, priors.severity, priors.iec, priors.iso)
```

`records` has one row per fault: `fault_id`, `component`, `part` (resistor, capacitor,
zener, switch, fuse, ntc, strain_gauge), `fault_type` (open_circuit, short_circuit,
parametric), and for `prior` also `severity` (class 0 to 3) and `effect`. `parts` gives
the kind of a component whose `part` is not specific enough.

## Rates

`component_rate(category, environment)` is the generic rate of the parts count method
times the quality factor π_Q, in FIT (failures per 10⁹ hours), for ground benign (gb),
fixed (gf) or mobile (gm). `mode_shares(category)` gives the probability of each fault
type. `fault_rates` splits the rate of each component over its modes and, within a
mode, equally over the faults that model it, such as the deviation levels of a
parametric fault. Modes the fault universe does not model (a fuse that fails to open)
are left out; `modelled_share` says how much of the component's rate the faults carry.

| Table | Source |
|---|---|
| `rates.csv` | MIL-HDBK-217F Notice 2 (1995), Appendix A, parts count |
| `modes.csv` | MIL-HDBK-338B (1998), Table 7.8-1 |

Both are U.S. government works and ship with the package; `rates(path)` and
`modes(path)` read others of the same form. These absolute rates are pessimistic for
current parts. Ratio metrics (DC, SFF, SPFM, LFM, DC_w) do not depend on their scale;
absolute ones such as PMHF and PFH need IEC 61709 or SN 29500, which cannot be
redistributed and can be passed as a rate table.

## Severity and classes

`classes(severity, effect)` gives the class of IEC 61508 (dangerous for severity 3,
else safe) and of ISO 26262 (single-point for severity 3, multiple-point when the only
effect is on the self-test, else safe). `prior` turns the severity class into the
weight `weights[class]`, linear by default, and keeps the class in `severity_class`.
