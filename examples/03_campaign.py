# %% [markdown]
# # A resumable fault campaign
#
# Runs 660 RC-filter simulations and resumes completed chunks on a second run.
# Also runnable as `python examples/03_campaign.py`.
# %%

import shutil
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

if __name__ == "__main__":  # workers are processes: needed on macOS and Windows
    run = campaign()

    print("checking the definitions against the circuit:")
    print(" ", run.validate())

    print("\nbefore:", run.status())
    run.run(workers=WORKERS, chunk=200, progress=False)
    print("after: ", run.status())

    print("\nper fault:")
    print(run.summary())

    samples, waveforms, manifest = run.load()
    print(f"\n{len(samples)} samples, waveforms {waveforms.shape}")
    print("columns:", list(samples.columns))
    print("written by", manifest["spicefault_version"], "with", manifest["simulator"],
          "on", manifest["workers"], "workers")  # fmt: skip

    # --- an interrupted run resumes -----------------------------------------------------
    interrupted = campaign("rc_interrupted")
    shutil.rmtree(interrupted.out_dir, ignore_errors=True)
    from spicefault.experiments import run_chunks, simulate_sample

    experiment = interrupted.experiment
    first = experiment.plan()[:400]  # two chunks of 200 are completed, then it stops
    run_chunks(first, simulate_sample, experiment, interrupted.out_dir / "parts",
               n_points=experiment.waveform.n_points, workers=WORKERS, chunk=200,
               progress=False)  # fmt: skip
    print("\ninterrupted:", interrupted.status())
    interrupted.run(workers=WORKERS, chunk=200, progress=False)
    resumed = interrupted.dataset()
    print("resumed:    ", interrupted.status())
    print("the manifest says it was resumed:", resumed.manifest.resumed)
    same = resumed.samples.drop(columns="elapsed_s").equals(
        run.dataset().samples.drop(columns="elapsed_s")
    )
    print("the resumed dataset is identical to the uninterrupted one:", same)
