# Fault model

Phase 0 deliverable 2 of 4. Status: implemented in `spicefault.faults` (Phase 3), except where a section says otherwise. The open decisions at the end still stand.

This document defines what a fault is in `spicefault`, which fault types the first release contains, how each one changes the netlist, and what is recorded about it. Notation follows [SCIENTIFIC_SCOPE.md](SCIENTIFIC_SCOPE.md) §2.

## 1. Definition

A circuit $C$ is a set of components. Each component has a designator, a kind, an ordered list of terminals connected to nodes, and named parameters with nominal values $x_0$.

A **fault** is a triple (type, location, parameters) that defines an injection operator $I_f$ acting on a realised circuit $(C, \theta)$. The operator is a finite list of the three primitive netlist transformations below. Nothing else may change the netlist.

| Primitive | Effect |
|---|---|
| `set_parameter(component, parameter, rule)` | Replace a parameter value. `rule` is `absolute(v)`: $x \leftarrow v$; `relative(δ)`: $x \leftarrow x_\theta (1 + δ)$; `scale(k)`: $x \leftarrow k\,x_\theta$; or `divide(k)`: $x \leftarrow x_\theta / k$ |
| `insert_series(component, terminal, R)` | Disconnect the terminal from its node and reconnect it through a resistance $R$ |
| `insert_parallel(component, terminal_a, terminal_b, R)` | Add a resistance $R$ between the nodes of two terminals |

Consequences:

- Every fault can be audited by listing its primitives, and two faults are identical if their primitive lists are.
- The topology of the circuit is never edited by free text.
- $x_\theta$ is the realised value after normal variation. `relative`, `scale` and `divide` compound with the Monte Carlo draw; `absolute` overwrites it. Both behaviours exist in the ECG code and both must be available (§6).
- `divide(k)` is not redundant with `scale(1/k)`: the two can differ in the last bit in floating point, and exact equivalence with existing results needs the operation that was actually used.

## 2. Order of operations

For every sample: draw $\theta$; build the realised circuit; apply $I_f$; apply the operating condition $u$; simulate. The fault is applied after the draw so that a faulty component still carries its own manufacturing deviation where the rule is relative, and all other components keep their spread.

## 3. Fault types of the first release

| Type | Physical interpretation | Primitives | Model parameters | Applies to |
|---|---|---|---|---|
| `OpenCircuit` | Broken connection: cracked joint, lifted lead, open track, open component | `insert_series` at one terminal | `r_open` | Any terminal of any component |
| `ShortCircuit` | Solder bridge, dielectric breakdown, conductive contamination | `insert_parallel` across two terminals | `r_short` | Any terminal pair |
| `LeakageFault` | Finite parasitic conduction: capacitor insulation loss, contaminated board, degraded junction | `insert_parallel` with a graded resistance | `r_leak` (the magnitude) | Mainly capacitors and high-impedance nodes |
| `ParametricFault` | A parameter outside its tolerance band: wrong value fitted, aged or stressed component, degraded device parameter | `set_parameter` | rule and value (the magnitude) | Any numeric parameter, including subcircuit parameters |
| `DriftFault` | Parametric deviation produced by a declared drift law over time or stress | `set_parameter`, value computed from the law | law, its coefficients, and the time or stress value | Parameters with a documented law |
| `CompositeFault` | One physical mechanism that changes several electrical quantities | Any list of primitives | Those of its parts | Declared case by case |

Notes on each type.

**Open and short are modelled with finite resistances.** An ideal open leaves nodes floating and makes the DC operating point singular. `r_open` and `r_short` are therefore model parameters, recorded with every sample, and results may depend on them. The ECG study uses `r_open = 1 GΩ` and `r_short = 1 Ω`, and an earlier version used 10 MΩ for opens. The effect of `r_open` relative to the impedances of the circuit should be checked once per circuit: an open in series with a 10 MΩ bias resistor is not well represented by 10 MΩ.

**Open on a two-terminal component** is electrically the same at either terminal. For components with more terminals the terminal must be named, and each terminal is a different fault.

**Leakage and short are the same primitive at different magnitudes.** They are kept as two types because the physical interpretation and the usual severity axis differ: a short is a hard fault at a fixed low resistance, leakage is graded over decades.

**DriftFault is a ParametricFault with a provenance.** It is not implemented (open decision 1). Its only addition is that the deviation is computed from a law, such as $x(t) = x_0 (1 + a\,t^{b})$, whose form, coefficients and source are recorded. It should be implemented only when at least one law with a citable source is available for the case study; otherwise it adds a name without physical content. See open decision 1.

**CompositeFault is not a multiple fault.** It represents one defect with several electrical consequences. The ECG example is electrolytic capacitor degradation, which lowers capacitance and raises series resistance together. Several independent defects at once (`MultipleFault`) remain out of scope.

Fault types deferred: `IntermittentFault`, `StuckAtFault`, `BridgingFault` between arbitrary nodes, `MultipleFault`. A bridge between two nodes that do not belong to the same component is not expressible with the primitives above, which are component-centred; adding it needs a fourth primitive `insert_between(node_a, node_b, R)`.

## 4. Fault and normal variation

A parametric deviation is a fault only relative to a declared tolerance. Let $t$ be the tolerance of a parameter, so that healthy values lie in $x_0 (1 \pm t)$.

- A relative fault of size $δ$ applied to a realised value gives a total deviation from nominal between $(1-t)(1+δ) - 1$ and $(1+t)(1+δ) - 1$.
- If that interval overlaps $\pm t$, some faulty samples are healthy by definition and no method can detect them.

This already happens in the ECG catalogue: capacitors have a 5 % tolerance and the smallest parametric faults are ±5 % and ±10 %. For `δ = +0.05` the total deviation ranges from −0.25 % to +10.25 %. With the uniform tolerance of that study, half of the ±5 % capacitor faults and 1 in 22 of the +10 % ones lie inside the healthy band; the −10 % ones never do, and neither does any resistor fault (1 % tolerance). This is deliberate there, because small deviations are what separates percentage severity from functional severity, but it must be visible.

Rule for the framework: catalogue validation computes this overlap for every parametric fault condition and reports it. It does not reject the condition. This is `FaultSet.tolerance_overlap`, which gives the deviation interval and the fraction of the fault population inside the band, for uniform and truncated-normal tolerances.

## 5. Magnitude and severity

Two different quantities, both optional except where noted.

**Magnitude** is the physical parameter of the fault, with a unit and a sign: a relative deviation, a resistance, an offset voltage. Graded faults must have one.

**Severity** $s$ is a dimensionless number that orders the conditions of one fault type on one component. It is defined by a declared scale; there is no universal formula, and severities of different fault types are not comparable.

| Fault type | Proposed default scale | Remarks |
|---|---|---|
| `ParametricFault`, relative | $s = \lvert δ \rvert$ | Unbounded above; sign kept in the magnitude |
| `ParametricFault`, absolute | $s = \lvert x_f - x_0 \rvert / x_\text{ref}$, with $x_\text{ref}$ declared | For a parameter with nominal value zero, such as offset voltage, $x_\text{ref}$ must be given: the tolerance limit is the natural choice |
| `LeakageFault` | $s = \log(r_\text{max} / r_\text{leak}) / \log(r_\text{max} / r_\text{min})$ on a declared range | Logarithmic, since the effect spans decades |
| `OpenCircuit`, `ShortCircuit` | None by default; 1.0 if a number is needed | Hard faults. A resistive open could be graded like leakage |
| `DriftFault` | Time or stress value divided by a declared reference | |
| `CompositeFault` | Declared with the composite | |

Severity as defined here describes what was injected. It is distinct from **functional severity**, which is whether and by how much the circuit violates its specifications, and is a result of the experiment, not a property of the fault. The framework must keep both and never derive one from the other.

## 6. Mapping of the ECG fault catalogue

Phase 1 must reproduce the ECG results unchanged, so every `ecgfd` fault kind needs an exact equivalent.

| `ecgfd` kind | `spicefault` type | Primitives | Remark |
|---|---|---|---|
| `open` | `OpenCircuit` | `insert_series` at the second terminal, 1 GΩ | |
| `short` | `ShortCircuit` | `insert_parallel`, 1 Ω | |
| `parametric` | `ParametricFault` | `relative(δ)`, δ ∈ {±0.05, ±0.1, ±0.2, ±0.5} | Compounds with the draw |
| `cap_degradation` | `CompositeFault` | `scale(1 − loss)` and `insert_series(ESR)` | Two fixed pairs (loss, ESR) |
| `opamp_vos`, `ina_vos` | `ParametricFault` | `absolute(v)` | Overwrites the drawn offset; only positive values are injected |
| `opamp_aol` | `ParametricFault` | `scale(k)` | Compounds with the draw |
| `ina_cmrr` | `ParametricFault` | `absolute(ratio)` | The netlist parameter is a signed ratio, sign × 10^(dB/20), and the sign is part of the Monte Carlo draw. The value to inject therefore depends on the drawn circuit, which a fault record written in advance cannot express. The adapter builds this fault per sample. A clean solution is to give the subcircuit separate magnitude and sign parameters, which changes the netlist text but not the results |
| `ina_gain` | `ParametricFault` | `absolute(error)` | |
| `electrode_off` | `ParametricFault` labelled as an open | `absolute(1 GΩ)` on the electrode series resistance | Replaces the value instead of inserting a resistance. Using `insert_series` would give a slightly different resistance (by less than 1 part in 10⁶) and break exact equivalence |
| `electrode_high_z` | `CompositeFault` | `scale(k)` on Rd and `divide(k)` on Cd, on one or two electrodes | The condition `la+ra` affects two components with one cause |

This mapping is implemented in `tests/regression/ecg_adapter.py` with the fault types of §3, and checked on every fault condition of both circuits: the netlists are identical, character by character, to those of `ecgfd`. The same file states the catalogue as applicability rules; the universe they generate (§8) is the catalogue of `ecgfd`, 293 and 307 conditions.

Three requirements follow:

- a fault must be able to carry a reported type different from its primitive (the `electrode_off` case);
- a composite must be able to span several components;
- faults need free-form tags for application labels. The ECG `origin` label (circuit or electrode) is such a tag, not a framework concept.

## 7. Identity, metadata and serialisation

Every fault condition has a stable identifier, built from its content and independent of its position in a list. Proposed form: `<location>:<type>[:<magnitude>]`, for example `R17:open`, `R11:parametric:+0.2`, as in the ECG code.

`fault.metadata()` returns a JSON-serialisable record. `class` is the type of §3 that built the fault and `fault_type` the type it is reported as; they differ when an application uses its own names. Severity is recorded with its scale and the parameters of the scale.

```json
{
  "schema_version": 1,
  "fault_id": "R11:parametric:+0.2",
  "class": "parametric",
  "fault_type": "parametric",
  "components": [
    "R11"
  ],
  "primitives": [
    {
      "op": "set_parameter",
      "component": "R11",
      "parameter": "value",
      "rule": "relative",
      "value": 0.2
    }
  ],
  "magnitude": {
    "value": 0.2,
    "unit": "relative deviation"
  },
  "severity": {
    "value": 0.2,
    "scale": "abs_relative_deviation",
    "parameters": {}
  },
  "model_parameters": {},
  "nominal_state": "within tolerance",
  "fault_state": "+0.2 relative to realised",
  "tags": {}
}
```

Requirements: a fault reconstructed from its record is equal to the original; the record is written to the campaign manifest once per condition and referenced by identifier in each sample; changing `schema_version` is the only way to change the meaning of a field.

## 8. Fault universe and coverage

For a circuit, the **fault universe** is generated from applicability rules: for each component kind, which fault types apply, at which terminals, and at which magnitudes. A campaign then selects a subset. Every excluded pair carries a reason, for example "not applicable", "no model", "excluded by user", "equivalent to another condition".

The coverage matrix (components × fault types) and the structural coverage figure are derived from this record; see RELIABILITY_METRICS.md, M10. This is `FaultUniverse`: rules generate the faults, `exclude` leaves some out with a reason, and `coverage_matrix`, `exclusions` and `metadata` give the record. Components that no rule applies to are listed, so that what lies outside the fault model is explicit. Integrated circuits modelled behaviourally are covered through their model parameters, and the matrix must say so, since a behavioural model exposes only the failure modes its parameters can express.

## 9. Assumptions and limitations

- Single fault per sample.
- Faults are static during a simulation.
- Open and short are resistive approximations with recorded values.
- Parametric faults on behavioural models are only as meaningful as the model parameter.
- The taxonomy is component-centred; faults of tracks, connectors and supplies must be represented through a component in the netlist.
- No occurrence probabilities are attached to faults. Weighting by failure-mode distributions is possible in the metrics if the user supplies weights from a cited source.

## 10. Open decisions

1. **DriftFault in the first release or deferred** until a drift law with a source is available. Recommended: deferred.
2. **Node-to-node bridging** (fourth primitive) in the first release or deferred. Recommended: deferred.
3. **Default `r_open`.** 1 GΩ as in the ECG study, with a per-circuit check against the largest impedance in the circuit.
4. **Negative offsets.** The ECG catalogue injects only positive offset faults; listed as pending there. It affects the fault universe, not the model.
