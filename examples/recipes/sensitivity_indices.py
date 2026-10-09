"""Recipe: variance-based sensitivity indices with SALib.

SALib chooses the points; the library simulates each one (`Experiment.evaluate`); SALib
computes the indices. The first-order index of a parameter is the share of the variance
of the measurement that it explains alone, and the total index includes its
interactions. Both are estimates with a confidence half-width, which SALib reports and
which shrinks slowly: n (2 d + 2) simulations for d parameters.

    pip install "spicefault[recipes]"
    python examples/recipes/sensitivity_indices.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rc_study import circuit, config, measurements  # noqa: E402
from SALib.analyze import sobol as analyze  # noqa: E402
from SALib.sample import sobol as sample  # noqa: E402

from spicefault import Experiment  # noqa: E402

TOLERANCE = {("R1", "value"): 0.01, ("C1", "value"): 0.05, ("R2", "value"): 0.01}
experiment = Experiment(circuit, config=config, measurements=measurements)
nominal = circuit.parameters()
targets = list(TOLERANCE)
problem = {
    "num_vars": len(targets),
    "names": [component for component, _ in targets],
    "bounds": [
        [nominal[t] * (1 - TOLERANCE[t]), nominal[t] * (1 + TOLERANCE[t])] for t in targets
    ],
}

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        points = sample.sample(problem, 64, calc_second_order=False, seed=0)
    gains = np.array([
        experiment.evaluate(dict(zip(targets, point, strict=True)))["nominal"]
        .measurements["gain_1k"]
        for point in points
    ])  # fmt: skip
    indices = analyze.analyze(problem, gains, calc_second_order=False, seed=0)
    print(f"{len(points)} simulations; indices of the gain at 1 kHz:")
    for name, first, total, half in zip(problem["names"], indices["S1"], indices["ST"],
                                        indices["ST_conf"], strict=True):  # fmt: skip
        print(f"  {name}: first order {first:5.2f}, total {total:5.2f} (+-{half:.2f})")
    print("the capacitor explains most of the variance:",
          bool(indices["ST"][1] > max(indices["ST"][0], indices["ST"][2])))  # fmt: skip
