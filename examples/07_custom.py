# %% [markdown]
# # Custom variations and measurements
#
# Define a joint variation and a measurement function. Functions used by workers must
# live at module level. Also runnable as `python examples/07_custom.py`.
# Jupyter uses one worker because functions defined in cells cannot be imported by spawned workers.
# %%

import sys
from pathlib import Path

import numpy as np

examples_dir = next(
    (root / "examples" for root in (Path.cwd(), *Path.cwd().parents)
     if (root / "examples" / "rc_study.py").is_file()),
    None,
)
if examples_dir is not None:
    sys.path.insert(0, str(examples_dir))

# %%
from rc_study import circuit, config  # noqa: E402

from spicefault import Experiment, Measurement  # noqa: E402
from spicefault.faults import ParametricFault  # noqa: E402
from spicefault.variation import Draw, JointVariation, ToleranceVariation  # noqa: E402


def matched_resistors(rng, netlist):
    """R1 and R2 from one reel: they share most of their deviation and differ a little.

    The type of the reel is drawn first and recorded as a label of the sample.
    """
    grade = str(rng.choice(["standard", "precision"]))
    spread = {"standard": 0.05, "precision": 0.005}[grade]
    common = 1.0 + spread * rng.uniform(-1, 1)
    values = {
        (name, "value"): netlist.value(name) * common * (1.0 + 0.001 * rng.uniform(-1, 1))
        for name in ("R1", "R2")
    }
    return Draw(values, {"resistor_grade": grade})


# %% [markdown]
# ## A custom frequency-response measurement
# %%
def corner_frequency(plot):
    """Frequency at which the response has fallen 3 dB from its low-frequency value [Hz]."""
    frequency = plot["frequency"].real
    gain = np.abs(plot["v(out)"])
    below = np.flatnonzero(gain < gain[0] / np.sqrt(2.0))
    return float(frequency[below[0]]) if len(below) else float(frequency[-1])


# %%
if __name__ == "__main__":
    worker_count = 1 if "ipykernel" in sys.modules else 2
    experiment = Experiment(
        circuit,
        config=config,
        faults=[ParametricFault("C1", deviation=0.5)],
        variations=[
            JointVariation("resistors", (("R1", "value"), ("R2", "value")), matched_resistors),
            ToleranceVariation("C1", 0.05),
        ],
        measurements=[
            Measurement.value("v(out)", name="dc"),
            Measurement.custom("corner", corner_frequency, analysis="ac"),
        ],
        samples=6,
        seed=3,
    )
    table = experiment.run(workers=worker_count).to_frame()
    columns = ["fault_id", "resistor_grade", "p_R1_value", "p_R2_value", "dc", "corner"]
    print(table[columns].round(3).to_string(index=False))
    # matched resistors keep the divider ratio, whatever their grade
    print("\nspread of the DC gain:", f"{table['dc'].std():.5f}")
    print("recorded in the definition of the experiment:")
    print(" ", experiment.metadata()["variations"][0])
