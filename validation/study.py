"""A validation study: a circuit and the four things that are specific to it.

Its netlist, its fault rules, its measurements and its specification limits. The
rest (drawing circuits, injecting faults, simulating, storing, analysing) is the
framework, the same for every study.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from spicefault import (
    Circuit,
    Experiment,
    FaultCampaign,
    Measurement,
    OperatingCondition,
    SimulationConfig,
    VariationSet,
    Waveform,
)
from spicefault.faults import FaultRule, FaultUniverse

NETLISTS = Path(__file__).parent / "netlists"


def load_circuit(name: str) -> Circuit:
    """The netlist of a study, with the libraries it includes written in place.

    A simulation runs in a scratch folder, where an `.include` would not be found.
    """
    lines = []
    for line in (NETLISTS / f"{name}.cir").read_text().splitlines():
        if line.lower().startswith(".include"):
            lines.append((NETLISTS / line.split()[1]).read_text().rstrip("\n"))
        else:
            lines.append(line)
    return Circuit("\n".join(lines) + "\n", name)


@dataclass
class Study:
    name: str
    description: str
    circuit: Circuit
    rules: list[FaultRule]
    variations: VariationSet
    config: SimulationConfig
    measurements: list[Measurement]
    waveform: Waveform | None
    # measurement -> (lowest, highest) value of a circuit that meets its specification
    specifications: dict[str, tuple[float, float]]
    # measurement -> standard deviation of the noise of the instrument that reads it
    noise: dict[str, float]
    # named sets of operating conditions; "nominal" is the one of the declared design
    conditions: dict[str, list[OperatingCondition]] = field(
        default_factory=lambda: {"nominal": [OperatingCondition()]}
    )

    @property
    def features(self) -> list[str]:
        return [m.name for m in self.measurements]

    def universe(self) -> FaultUniverse:
        return FaultUniverse(self.circuit, self.rules)

    def experiment(
        self,
        samples_per_fault: int = 200,
        healthy_samples: int = 5000,
        tolerance_scale: float = 1.0,
        conditions: str = "nominal",
        seed: int = 42,
    ) -> Experiment:
        """The experiment of this study. Seeding by fault identifier, so that a fault has
        the same samples whatever the other faults, and tolerance levels share their draws.
        """
        return Experiment(
            self.circuit,
            faults=self.universe().selected(),
            samples=samples_per_fault,
            healthy_samples=healthy_samples,
            variations=self.variations.scaled(tolerance_scale),
            conditions=self.conditions[conditions],
            config=self.config,
            measurements=self.measurements,
            waveform=self.waveform,
            seed=seed,
            seeding="content",
        )

    def campaign(self, out_dir: str | Path, **kwargs) -> FaultCampaign:
        return FaultCampaign.from_experiment(self.experiment(**kwargs), out_dir)

    def compliant(self, samples: pd.DataFrame) -> pd.Series:
        """True for the samples whose measurements are all within the specification."""
        ok = pd.Series(True, index=samples.index)
        for name, (low, high) in self.specifications.items():
            ok &= samples[name].between(low, high)
        return ok

    def observed(self, samples: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
        """The samples as an instrument would read them: Gaussian noise on each measurement.

        The simulated values are noise-free. This is the measurement model, applied
        after the simulation; row order does not matter, the noise of a row depends on
        its `sample_id`.
        """
        out = samples.copy()
        names = [m for m in self.features if self.noise.get(m)]
        ids = out["sample_id"].astype(int)
        draws = np.array([np.random.default_rng([seed, i]).normal(size=len(names)) for i in ids])
        for k, name in enumerate(names):
            out[name] = out[name] + self.noise[name] * draws[:, k]
        return out
