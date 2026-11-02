# %% [markdown]
# # The cheapest certified self-test
#
# The Sallen-Key band-pass of example 09 again, now as the self-test of a board: which
# tests can its microcontroller run, what does its ADC read, what does each plan add to
# the board, and which is the cheapest plan whose risk-weighted coverage reaches the
# target, checked against the lower bound on circuits that did not choose it. Also
# runnable as `python examples/10_minimum_cost_plan.py`; the exact plan needs
# `pip install "spicefault[ilp]"`.
# %%

from dataclasses import replace

import pandas as pd

from spicefault import Circuit, Experiment
from spicefault.faults import FaultUniverse, open_rule, parametric_rule, short_rule
from spicefault.testdesign import GaussianLimits, bounds, candidates, costmodel, faultprior
from spicefault.testdesign import metrics as mt
from spicefault.testdesign.metrics import adaptive, formulation
from spicefault.testdesign.select import ilp, pareto
from spicefault.variation import tolerances

WORKERS = 2
TARGET = 0.97  # DC_w
MAX_TESTS = 4
ALPHA = 0.01  # false alarm budget of the plan

# Sallen-Key band-pass, 25 kHz (Aminian and Aminian, 2000), with a single-pole op-amp
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
# ## Candidate tests, faults and their priors
#
# The candidate tests of example 09. Each fault gets a rate from MIL-HDBK-217F
# (ground fixed) split over the failure modes of MIL-HDBK-338B, and a severity: here,
# opens and shorts lose the filter (class 3, dangerous) and drifts degrade it (class 1).
# %%
circuit = candidates.prepare_source(Circuit(NETLIST, "sallen_key"), "V1")
nodes = candidates.accessible_nodes(circuit)
stimuli = [
    candidates.ac_tones("V1", [10e3, 25e3, 60e3]),
    candidates.step("V1", 1.0, 200e-6, [10e-6, 20e-6, 50e-6]),
    *candidates.dc_levels("V1", [1.0]),
]
tests = candidates.enumerate_tests(nodes, stimuli)
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
if __name__ == "__main__":
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", "{:.4g}".format)
    print(f"{len(tests)} candidate tests, {len(faults)} faults")
    print("rates in FIT of the faults of R1 and C1:")
    print(priors.loc[[i for i in priors.index if i[:2] in ("R1", "C1")], ["rate", "iec"]])

# %% [markdown]
# ## The board: what it can run and what its ADC reads
#
# An STM32G474 with four ADC channels and no op-amp left free by the application. The
# simulated tests are read through its ADC model; tests it cannot run, or that no fault
# changes as read, leave the pool. A node whose source impedance is too high for the
# ADC needs a buffer.
# %%
board = replace(costmodel.STM32G474, channels=4, opamps=0)
# assumed prices, in euros; a study reads them from its own table with costmodel.prices
price = {"external_opamp_channel": 0.12, "mux_8": 0.25, "injection_switch": 0.40}
if __name__ == "__main__":
    experiment = Experiment(
        circuit,
        faults=faults,
        variations=variations,
        conditions=candidates.conditions(tests),
        samples=40,
        healthy_samples=400,
        seed=1,
    )
    simulated = candidates.matrix(experiment.run(workers=WORKERS).to_frame(), tests)
    amplitude = costmodel.amplitudes(simulated, tests, board)
    read = costmodel.observe(simulated, tests, board, amplitude, seed=1)
    runnable = [t for t in tests if not costmodel.feasible(t, board)]
    constant = set(candidates.constant_tests(read, runnable, atol=1e-6))
    pool = [t for t in runnable if t.test_id not in constant]
    access = costmodel.access(circuit, nodes, board, frequencies=[1.0, 10e3, 25e3, 60e3])
    print(f"\n{len(pool)} tests in the pool ({len(tests) - len(runnable)} not runnable)")
    print("nodes that need a buffer:", [n for n, a in access.items() if a.buffer])

# %% [markdown]
# ## The cheapest plan, exactly and by the greedy
#
# The plan is chosen on the first half of the faulty replicas. Each test gets limits of
# its own at a false alarm rate of ALPHA / MAX_TESTS, and the false alarm rate of the
# whole plan on held-out fault-free circuits is a constraint as well. The integer
# programme gives the plan of least euros, then least test time, then fewest tests; the
# greedy adds the test with most coverage per euro, on exactly the same problem.
# %%
if __name__ == "__main__":

    def duration(t):
        return costmodel.cost([t], board, access, price).time

    def problem(which):
        return ilp.instance(
            formulation.part(read, which), pool, table, "dc_w", access, board, price,
            ALPHA / MAX_TESTS, duration,
        )  # fmt: skip

    selection = problem("selection")
    best = ilp.solve(selection, TARGET, max_tests=MAX_TESTS, max_false_alarm=ALPHA)
    print(f"\noptimal plan: {best.tests}")
    print(f"  {best.euros:.2f} EUR, {1e3 * best.time:.2f} ms, DC_w {best.coverage:.3f}")
    chosen = [t for t in pool if t.test_id in best.tests]
    print("  bill:", costmodel.cost(chosen, board, access, price).items)

    def scorer(plan):
        if selection.false_alarm_rate(plan) > ALPHA:
            return float("-inf")
        return selection.coverage(plan)

    path = pareto.greedy(
        selection.tests, scorer, lambda p: (selection.euros(p), selection.time(p)),
        target=TARGET, max_tests=MAX_TESTS,
    )  # fmt: skip
    steps = pd.DataFrame([vars(s) for s in path])
    print("\ngreedy path:")
    print(steps)

# %% [markdown]
# ## Scored and certified on other circuits
#
# The coverage of the plan on the evaluation half of the replicas, which did not choose
# it, and then the claim "DC_w ≥ TARGET" decided by weighted fault sampling on new
# circuits (another seed), read through the same ADC model.
# %%
if __name__ == "__main__":
    evaluation = problem("evaluation")
    held_out = evaluation.coverage(best.tests)
    print(f"\nDC_w of the optimal plan on the evaluation circuits: {held_out:.3f}")
    with formulation.ConditionSampler(
        circuit, faults, candidates.conditions(chosen), best.tests, GaussianLimits(ALPHA),
        variations=variations, n_calibration=400, n_max=400, seed=2, workers=WORKERS,
        transform=lambda m: costmodel.observe(m, chosen, board, amplitude, seed=2),
    ) as sampler:  # fmt: skip
        claim = adaptive.certify_sampled(
            sampler, table, target=TARGET, metric="dc_w", sizes=adaptive.schedule(50, 800)
        )
    print(f"DC_w >= {TARGET}: {claim.decision} after {claim.simulated} faulty circuits")

# %% [markdown]
# ## What the plan gives up
#
# The theoretical bound of the pool: the rank of the sensitivity matrix and the
# canonical ambiguity groups. A plan chosen to detect faults can reach its coverage
# target and still identify fewer components than the pool could.
# %%
if __name__ == "__main__":
    components = ["R1", "R2", "R3", "R4", "R5", "C1", "C2"]
    perturbed = Experiment(
        circuit, faults=bounds.perturbations(components),
        conditions=candidates.conditions(pool), samples=1, healthy_samples=1, seed=0,
    )  # fmt: skip
    ids = [t.test_id for t in pool]
    S = bounds.sensitivity(candidates.matrix(perturbed.run().to_frame(), pool), ids, components)
    noise = read[read.fault_id == mt.HEALTHY][ids].std()
    full, plan = bounds.bound(S, noise), bounds.bound(S.loc[list(best.tests)], noise)
    print(f"\npool: rank {full.rank}, groups {full.groups}, identifiable {full.identifiable}")
    print(f"plan: rank {plan.rank}, identifiable {plan.identifiable}")
    print("gap:", bounds.gap(full, plan))
