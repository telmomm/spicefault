"""Reliability analysis: what the fault responses say about the circuit.

    python examples/05_reliability.py

Detection of each fault, the smallest deviation that is visible, which faults look
alike, and how tolerance erodes detection (the same campaign at two tolerance scales).
"""

import pandas as pd
from rc_study import WORKERS, campaign

from spicefault.reliability import ReliabilityAnalysis, robustness

ALPHA = 0.02  # false-alarm rate the detector is set for

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

    print("\nsmallest deviation detected at least 90 % of the time:")
    print(analysis.minimum_detectable(alpha=ALPHA).drop(columns="grid").to_string(index=False))

    ambiguity = analysis.ambiguity(threshold=3.0)
    print("\nfaults not separated from the healthy circuits:", ambiguity.undetectable)
    print("components that can be mistaken for each other:", ambiguity.confusable)

    print("\ndetection against the tolerance scale (1 = the declared tolerances):")
    print(robustness(analyses, alpha=ALPHA))
