# %% [markdown]
# # Operating conditions
#
# Observe the same drawn circuits under different supplies and temperatures.
# A fault that is plain at room temperature can hide when the circuit is hot.
# Also runnable as `python examples/06_operating_conditions.py`.
# %%

import sys
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
from spicefault import (  # noqa: E402
    Circuit,
    FaultCampaign,
    Measurement,
    OperatingCondition,
    SimulationConfig,
)
from spicefault.faults import ParametricFault  # noqa: E402
from spicefault.reliability import ReliabilityAnalysis, detectability_across  # noqa: E402
from spicefault.variation import tolerances  # noqa: E402

# R1 grows 0.2 % per kelvin; R2 does not
circuit = Circuit(
    "divider\nV1 in 0 dc 5\nR1 in out 10k tc1=2e-3\nR2 out 0 10k\n.end\n", name="divider"
)
conditions = [
    OperatingCondition("nominal"),
    OperatingCondition("low supply", settings={("V1", "dc"): 4.5}),
    OperatingCondition("hot", temperature=85.0),
]

if __name__ == "__main__":
    from rc_study import OUTPUT, WORKERS

    campaign = FaultCampaign(
        circuit,
        [ParametricFault("R1", deviation=d) for d in (0.02, 0.05, 0.2)],
        out_dir=OUTPUT / "conditions",
        samples_per_fault=40,
        healthy_samples=400,
        variations=tolerances(circuit, {"R": 0.01}),
        conditions=conditions,
        config=SimulationConfig(("op",), outputs=("v(out)",)),
        measurements=[Measurement.value("v(out)", name="vout")],
        seed=7,
    )
    campaign.run(workers=WORKERS, progress=False)
    samples = campaign.dataset().samples

    # one drawn circuit is simulated under every condition
    first = samples[(samples["fault_id"] == "healthy") & (samples["replica"] == 0)]
    print("the first healthy circuit, under each condition:")
    print(first[["condition", "p_R1_value", "p_R2_value", "vout"]].round(4).to_string(index=False))

    # each condition is judged against the healthy circuits under that same condition
    parts = ReliabilityAnalysis.from_dataset(campaign.out_dir).by("condition")
    pd.set_option("display.width", 160)
    print("\ndetection probability of each fault under each condition:")
    print(detectability_across(parts, alpha=0.02).round(3))
