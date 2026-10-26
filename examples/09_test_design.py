# %% [markdown]
# # Test design
#
# Which nodes, stimuli and readings should the self-test of a band-pass filter use, and
# what diagnostic coverage do they certify? Candidate tests, detection with its
# intervals, the coverage metrics of IEC 61508 and ISO 26262 with bounds that hold for
# all faults at once, a published test selection method, and a stopping rule that
# simulates until the coverage claim is decided. Also runnable as
# `python examples/09_test_design.py`.
# %%

import pandas as pd

from spicefault import Circuit, Experiment
from spicefault.faults import FaultUniverse, open_rule, parametric_rule, short_rule
from spicefault.testdesign import GaussianLimits, candidates, metrics
from spicefault.testdesign.metrics import adaptive
from spicefault.testdesign.select import baselines
from spicefault.variation import tolerances

WORKERS = 2

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
# ## Candidate tests
#
# A test is a feature of the voltage at an accessible node under a stimulus: here the
# gain and phase of three tones, a 1 V step read at three instants, and a DC level.
# The source gets a flat pulse so that the step can set it.
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
if __name__ == "__main__":
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", "{:.4g}".format)
    print("accessible nodes:", nodes)
    print(f"{len(tests)} candidate tests over {len(stimuli)} stimuli, {len(faults)} faults")
    print(candidates.listing(tests).head(8))

# %% [markdown]
# ## Simulate every drawn circuit under every stimulus
#
# One experiment, one operating condition per stimulus: a drawn circuit is seen under
# all of them, so its tests can be combined. The matrix has one row per circuit.
# %%
if __name__ == "__main__":
    experiment = Experiment(
        circuit,
        faults=faults,
        variations=variations,
        conditions=candidates.conditions(tests),
        samples=40,
        healthy_samples=200,
        seed=1,
    )
    matrix = candidates.matrix(experiment.run(workers=WORKERS).to_frame(), tests)
    constant = candidates.constant_tests(matrix, tests, atol=1e-6)
    ids = [t.test_id for t in tests if t.test_id not in constant]
    print(f"\n{len(matrix)} circuits; {len(constant)} tests no fault changes:", constant)

# %% [markdown]
# ## Detection, false alarms and coverage of all the tests
#
# The limits are set on half of the fault-free circuits and the false alarm rate is
# measured on the other half. The bounds of each metric hold for all faults at once.
# %%
if __name__ == "__main__":
    detector = GaussianLimits(0.01)
    every = metrics.detect(matrix, ids, detector)
    print("\nfalse alarm rate on held-out circuits:", every.false_alarm)
    table = metrics.fault_table(every.per_fault.index)
    print("\ncoverage of all the tests:")
    print(metrics.coverage(every, table))

# %% [markdown]
# ## A published test selection method
#
# The greedy selection of Zilch et al. (2022), by the hardest fault, on the binary
# dictionary of which test detects which fault.
# %%
if __name__ == "__main__":
    dictionary = baselines.detection_table(matrix, ids, lambda: GaussianLimits(0.01))
    chosen = baselines.zilch_greedy(dictionary)
    selected = metrics.detect(matrix, chosen, GaussianLimits(0.01))
    bounds = metrics.coverage(selected, table)
    print(f"\nselected tests ({len(chosen)} of {len(ids)}):", chosen)
    print(bounds)
    print("SIL allowed by the lower bound of SFF:", metrics.sil_reached(bounds.loc["sff", "lower"]))
    sets = metrics.ambiguity_sets(matrix, selected)
    print("faults the tests mostly miss:", sets[0])
    print("largest set of faults that look alike:", max(sets[1:], key=len))
    print("uncertainty about the fault, in bits:", metrics.localisation(matrix, selected))

# %% [markdown]
# ## Simulate until the claim is decided
#
# Is DC_w at least 0.9 with the selected tests? The fault of each circuit is drawn with
# the weights of the metric, so each simulated circuit is one trial of the metric and
# one interval on the total is exact. Replicas are simulated only when the rule reads
# them, and each is the circuit that the plan gives it. The 40 circuits per fault above
# bound DC_w only from about 0.79; a claim decided this way is wrong at most 5 % of the
# time.
# %%
if __name__ == "__main__":
    chosen_tests = [t for t in tests if t.test_id in chosen]
    with adaptive.SpiceSampler(
        circuit,
        faults,
        chosen_tests,
        GaussianLimits(0.01),
        variations=variations,
        n_calibration=200,
        n_max=400,
        seed=1,
        workers=WORKERS,
    ) as sampler:
        claim = adaptive.certify_sampled(
            sampler, table, target=0.9, metric="dc_w", sizes=adaptive.schedule(50, 800)
        )
    print(f"\nDC_w >= 0.9: {claim.decision} after {claim.simulated} faulty circuits")
    print(claim.history)
