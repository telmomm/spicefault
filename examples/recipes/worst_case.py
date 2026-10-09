"""Recipe: search for the worst case of a measurement with SciPy.

The library simulates a given parameter vector (`Experiment.evaluate`); the search is
SciPy's. What it returns is the worst case *found*: a bounded quasi-Newton search stops
at a local extreme, and nothing here proves that there is no worse point. For a
response that is monotonic in each parameter over the tolerance band, as this one, the
worst case is a corner of the band, and the search is checked against it.

    pip install "spicefault[recipes]"
    python examples/recipes/worst_case.py
"""

import sys
from itertools import product
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rc_study import circuit, config, measurements  # noqa: E402

from spicefault import Experiment  # noqa: E402

TOLERANCE = {("R1", "value"): 0.01, ("C1", "value"): 0.05, ("R2", "value"): 0.01}
experiment = Experiment(circuit, config=config, measurements=measurements)
nominal = circuit.parameters()
targets = list(TOLERANCE)


def gain(deviation) -> float:
    """The gain at 1 kHz with each parameter at nominal (1 + tolerance * deviation)."""
    values = {
        t: nominal[t] * (1 + TOLERANCE[t] * d) for t, d in zip(targets, deviation, strict=True)
    }
    return experiment.evaluate(values)["nominal"].measurements["gain_1k"]


if __name__ == "__main__":
    count = 0

    def objective(deviation):
        global count
        count += 1
        return gain(deviation)

    found = minimize(objective, x0=np.zeros(len(targets)), bounds=[(-1, 1)] * len(targets),
                     method="L-BFGS-B", options={"eps": 1e-2})  # fmt: skip
    corners = {corner: gain(corner) for corner in product((-1.0, 1.0), repeat=len(targets))}
    worst_corner = min(corners, key=corners.get)
    print(f"nominal gain at 1 kHz: {gain(np.zeros(len(targets))):.5f}")
    print(f"lowest gain found by the search: {found.fun:.5f} after {count} simulations")
    print("at deviations (in tolerances):", np.round(found.x, 3).tolist())
    print(f"lowest gain over the 8 corners: {corners[worst_corner]:.5f} at {list(worst_corner)}")
    reached = abs(found.fun - corners[worst_corner]) < 1e-6
    print("the search reached the worst corner:", bool(reached))
