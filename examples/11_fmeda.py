# %% [markdown]
# # The FMEDA of a self-test plan
#
# The plan that example 10 chose for the Sallen-Key band-pass, handed to an assessor as
# an FMEDA: one row per failure mode of each component, with its rate, its class and the
# diagnostic coverage of the plan computed by simulation, with a bound that holds for
# every row at once, and totals that are the claim. The workbook has the totals as
# formulas, so a rate, a class or a DC can be changed and the totals follow. Also
# runnable as `python examples/11_fmeda.py`; the workbook needs
# `pip install "spicefault[fmeda]"`.
# %%

import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

examples_dir = next(
    (root / "examples" for root in (Path.cwd(), *Path.cwd().parents)
     if (root / "examples" / "rc_study.py").is_file()),
    None,
)
if examples_dir is not None:
    sys.path.insert(0, str(examples_dir))

# %%
from rc_study import OUTPUT, WORKERS  # noqa: E402

from spicefault import Circuit, Experiment  # noqa: E402
from spicefault.faults import FaultUniverse, open_rule, parametric_rule, short_rule  # noqa: E402
from spicefault.testdesign import GaussianLimits, candidates, costmodel, faultprior, fmeda  # noqa: E402
from spicefault.testdesign import metrics as mt  # noqa: E402
from spicefault.testdesign.metrics import adaptive, formulation  # noqa: E402
from spicefault.variation import tolerances  # noqa: E402

METRIC, TARGET = "spfm", 0.97  # ASIL C asks for SPFM > 97 %
ALPHA = 0.01  # false alarm budget of the plan
CONFIDENCE = 0.95
PLAN = ("ac|out|gain_db@60000", "ac|out|phase@60000")  # the optimal plan of example 10

NETLIST = """sallen_key_bandpass
.subckt opamp inp inn out
Rin inp inn 2meg
G1 0 p inp inn 1
Rp p 0 2e5
Cp p 0 31.83n
E1 o 0 p 0 1
Ro o out 75
.ends opamp
V1 in 0 dc 0 ac 1
R1 in a 5.18k
R2 a out 1k
C1 a b 5n
C2 a 0 5n
R3 b 0 2k
R4 n 0 4k
R5 out n 4k
X1 b n out opamp
.end
"""

# %% [markdown]
# ## The circuit, its faults and the plan
#
# As in example 10: the faults with their rates (MIL-HDBK-217F, ground fixed, over the
# failure modes of MIL-HDBK-338B) and severities, the board, and the two tests of the
# plan read through its ADC. The fault table of the FMEDA is the prior with the component
# and fault type of each fault.
# %%
circuit = candidates.prepare_source(Circuit(NETLIST, "sallen_key"), "V1")
tones = candidates.ac_tones("V1", [10e3, 25e3, 60e3])
tests = candidates.enumerate_tests(candidates.accessible_nodes(circuit), [tones])
plan = [t for t in tests if t.test_id in PLAN]
universe = FaultUniverse(
    circuit,
    [open_rule("RC", r_open=1e8), short_rule("RC"), parametric_rule((-0.5, 0.5), "RC")],
)
faults = list(universe.selected())
variations = tolerances(circuit, {"R": 0.05, "C": 0.10})
records = pd.DataFrame({
    "fault_id": [f.fault_id for f in faults],
    "component": [f.primitives[0].component for f in faults],
    "part": ["resistor" if f.primitives[0].component[0] == "R" else "capacitor" for f in faults],
    "fault_type": [f.fault_type for f in faults],
    "severity": [1 if f.fault_type == "parametric" else 3 for f in faults],
    "effect": ["degraded" if f.fault_type == "parametric" else "lost" for f in faults],
})  # fmt: skip
priors = faultprior.prior(records, "gf")
table = mt.fault_table(priors.index, priors.rate, priors.severity, priors.iec, priors.iso)
fault_table = priors.join(records.set_index("fault_id")[["component", "fault_type", "effect"]])
board = replace(costmodel.STM32G474, channels=4, opamps=0)
if __name__ == "__main__":
    # the amplitude of the tone, set as in example 10 from fault-free circuits
    healthy = Experiment(
        circuit, faults=[], variations=variations, conditions=candidates.conditions(tests),
        samples=1, healthy_samples=400, seed=1,
    )  # fmt: skip
    simulated = candidates.matrix(healthy.run(workers=WORKERS).to_frame(), tests)
    amplitude = costmodel.amplitudes(simulated, tests, board)
    print(f"{len(faults)} faults, plan {PLAN}")

# %% [markdown]
# ## A claim for every row
#
# The plan was chosen for DC_w; an assessor of ISO 26262 reads the single-point fault
# metric. `certify` simulates each fault until the lower bound of the SPFM reaches the
# target, with per-fault bounds that hold for all faults at every stage. The sampler
# records which tests of the plan alarm on each fault: the safety mechanism of its row.
# %%
if __name__ == "__main__":

    def sampler(seed, record=False):
        return formulation.ConditionSampler(
            circuit, faults, candidates.conditions(plan), PLAN, GaussianLimits(ALPHA),
            variations=variations, n_calibration=400, n_max=400, seed=seed, workers=WORKERS,
            transform=lambda m: costmodel.observe(m, plan, board, amplitude, seed=seed),
            record=record,
        )  # fmt: skip

    with sampler(2, record=True) as s:
        per_mode = adaptive.certify(
            s, table, TARGET, METRIC, CONFIDENCE, sizes=adaptive.schedule(25, 400)
        )
        alarming = s.alarming
    print(f"\nper mode: SPFM >= {TARGET}: {per_mode.decision} after {per_mode.simulated}"
          f" faulty circuits, lower bound {per_mode.lower:.4f}")  # fmt: skip

# %% [markdown]
# ## The FMEDA
#
# One row per component, failure mode, severity and class. Opens and shorts are
# single-point, the drifts of ±50 % safe. The DC claimed of every row is its lower bound,
# so the SPFM of the sheet is the lower bound of `certify`. A drift was simulated only
# as far as the SPFM needed, which is not far: a safe fault leaves no residual. Every
# mode of a resistor and a capacitor is modelled here, so no row is `not assessed`.
# %%
if __name__ == "__main__":
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.float_format", "{:.4g}".format)
    found = fmeda.faults(fault_table, per_mode.per_fault, CONFIDENCE, "mode", tests=alarming)
    rows = fmeda.rows(found, "mode")
    summary = fmeda.summary(rows, {METRIC: TARGET})
    print(rows[["component", "mode", "iso", "lambda", "safety_mechanism", "circuits",
                "estimate", "dc_claimed"]].head(6))  # fmt: skip
    print(summary[["claimed", "estimate", "upper", "target", "met"]])
    same = abs(summary.claimed[METRIC] - per_mode.lower) < 1e-12
    print(f"the totals of the sheet are the claim: {same}")

# %% [markdown]
# ## A claim on the metric only
#
# Weighted fault sampling decides the same target with far fewer circuits, but claims
# nothing about any row: the sheet then carries the estimates, and the Summary the
# claim of `certify_sampled`.
# %%
if __name__ == "__main__":
    with sampler(3) as s:
        per_metric = adaptive.certify_sampled(
            s, table, TARGET, METRIC, CONFIDENCE, sizes=adaptive.schedule(50, 800)
        )
    print(f"per metric: SPFM >= {TARGET}: {per_metric.decision} after"
          f" {per_metric.simulated} faulty circuits")  # fmt: skip
    ratio = per_mode.simulated / per_metric.simulated
    print(f"a claim per row costs {ratio:.0f} times more circuits")

# %% [markdown]
# ## The workbook
#
# Cover, FMEDA, Faults, Summary and Assumptions. The same content gives the same file.
# `fmeda.template` writes the same workbook empty, for an FMEDA filled by hand from the
# tables of the standards.
# %%
if __name__ == "__main__":
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cover = {
        "Case": "Sallen-Key band-pass, 25 kHz",
        "Safety goal": "Do not pass a filter that has lost its band as healthy",
        "Standard and target": f"ISO 26262, ASIL C: SPFM > {TARGET:.0%}",
        "Plan": " ".join(PLAN) + " (example 10)",
        "Claim": "Per mode: the DC claimed of every row is its lower bound, which holds for "
        "all faults at once (certify); the totals of the sheet are the claim.",
        "Confidence": CONFIDENCE,
        "Decision": f"{per_mode.decision}, lower bound {per_mode.lower:.4f}",
    }
    sampled = pd.DataFrame([{
        "metric": "SPFM", "target": TARGET, "decision": per_metric.decision,
        "lower": per_metric.lower, "circuits": per_metric.simulated,
    }])  # fmt: skip
    path = fmeda.write(OUTPUT / "sallen_key_fmeda.xlsx", rows, found, summary, cover, sampled)
    asil = mt.asil_reached(summary.claimed.spfm)
    print(f"\nwrote {path.name}, ASIL reached by the sheet: {asil}")
