# %% [markdown]
# # Inspect a campaign dataset
#
# Check integrity, provenance, reproduction and model-ready arrays.
# Also runnable as `python examples/04_dataset.py`.
# %%

import sys
from pathlib import Path

examples_dir = next(
    (root / "examples" for root in (Path.cwd(), *Path.cwd().parents)
     if (root / "examples" / "rc_study.py").is_file()),
    None,
)
if examples_dir is not None:
    sys.path.insert(0, str(examples_dir))

# %%
from rc_study import WORKERS, campaign  # noqa: E402

from spicefault import Dataset  # noqa: E402

# %% [markdown]
# ## Ensure the campaign dataset exists
#
# This notebook can run independently; it creates the dataset if campaign 03 has not run.
# %%
if __name__ == "__main__":
    campaign().run(workers=WORKERS, chunk=200, progress=False)

    # everything below needs only the folder
    dataset = Dataset(campaign().out_dir)
    print(dataset)
    print("files:", sorted(p.name for p in dataset.path.iterdir()))
    print("problems found by verify():", dataset.verify())

# %% [markdown]
# ## Trace a sample back to its simulation
# %%
if __name__ == "__main__":
    sample_id = 350
    print(f"\nprovenance of sample {sample_id}:")
    for key, value in dataset.provenance(sample_id).to_dict().items():
        print(f"  {key}: {value}")

    print(f"\nthe netlist that was simulated for sample {sample_id}:")
    print(dataset.netlist(sample_id))

# %% [markdown]
# ## Reproduce measurements and prepare arrays
# %%
if __name__ == "__main__":
    print("simulating 20 samples again and comparing with what is stored:")
    report = dataset.reproduce(n=20)
    print(report.head().to_string(index=False))
    print("largest difference in the measurements:", report["max_abs_diff"].max())
    print("largest difference in the waveforms:   ", report["waveform_abs_diff"].max())

    x, y = dataset.to_ml(target="fault_location")
    print(f"\nfor a classifier: X {x.shape}, labels {sorted(set(y))}")
    waves, kind = dataset.to_ml(target="fault_type", waveforms=True)
    print(f"or the waveforms: X {waves.shape}, labels {sorted(set(kind))}")
