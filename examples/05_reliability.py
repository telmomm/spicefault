# %% [markdown]
# # Reliability analysis
#
# Compare detection, minimum detectable deviation, ambiguity and tolerance robustness.
# Also runnable as `python examples/05_reliability.py`.
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
from rc_study import WORKERS, campaign  # noqa: E402

from spicefault.reliability import ReliabilityAnalysis, robustness  # noqa: E402

ALPHA = 0.02  # false-alarm rate the detector is set for

# %% [markdown]
# ## Build reliability analyses at two tolerance scales
#
# Compare the declared tolerance population with one three times wider.
# %%
if __name__ == "__main__":
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", "{:.3g}".format)
    scales = (1.0, 3.0)
    analyses = {}
    for scale in scales:
        run = campaign(f"rc_tolerance_x{scale:g}", tolerance_scale=scale)
        run.run(workers=WORKERS, chunk=200, progress=False)
        analyses[scale] = ReliabilityAnalysis.from_dataset(run.out_dir)
    analysis = analyses[1.0]

# %% [markdown]
# ## Measure false alarms and detection
# %%
if __name__ == "__main__":
    detection = analysis.detectability(alpha=ALPHA)
    low, high = detection.false_alarm_interval
    print(f"limit test set for a false-alarm rate of {ALPHA}")
    print(f"measured on {detection.n_evaluation} healthy samples that did not set the limits: "
          f"{detection.false_alarm:.3f} [{low:.3f}, {high:.3f}]")  # fmt: skip

    table = detection.table[["n_ok", "p_detect", "ci_low", "ci_high"]].join(
        [analysis.standardised_shift(), analysis.auc()["auc"]]
    )
    print("\ndetection probability, with its interval; shift of the best feature; AUC:")
    print(table)

# %% [markdown]
# ## Find limits and ambiguous faults
# %%
if __name__ == "__main__":
    print("\nsmallest deviation detected at least 90 % of the time:")
    print(analysis.minimum_detectable(alpha=ALPHA).drop(columns="grid").to_string(index=False))

    ambiguity = analysis.ambiguity(threshold=3.0)
    print("\nfaults not separated from the healthy circuits:", ambiguity.undetectable)
    print("components that can be mistaken for each other:", ambiguity.confusable)

# %% [markdown]
# ## Compare robustness against tolerance
# %%
if __name__ == "__main__":
    print("\ndetection against the tolerance scale (1 = the declared tolerances):")
    print(robustness(analyses, alpha=ALPHA))
