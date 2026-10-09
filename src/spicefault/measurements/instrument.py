"""The instrument: measurements as they are observed, applied after the simulation.

The simulator output is stored noise-free. Noise, resolution and range belong to the
instrument that reads a measurement, and are applied to the stored values, so they are
parameters of a study that need no new simulation (docs/SCIENTIFIC_SCOPE.md, section 2).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Reading:
    """How one measurement is read: in this order, Gaussian noise of standard deviation
    `noise` is added, the result is rounded to a multiple of `resolution`, and it is
    clipped to `limits`. Each of the three is optional.
    """

    noise: float = 0.0
    resolution: float | None = None
    limits: tuple[float, float] | None = None

    def __post_init__(self):
        if self.noise < 0 or (self.resolution is not None and self.resolution <= 0):
            raise ValueError("noise must not be negative and resolution must be positive")
        if self.limits is not None:
            low, high = (float(limit) for limit in self.limits)
            if not low < high:
                raise ValueError(f"limits are (low, high) with low < high, not {self.limits}")
            object.__setattr__(self, "limits", (low, high))

    @property
    def uncertainty(self) -> float:
        """Standard deviation that the reading adds to a value inside its range: the
        noise and the rounding error, which is uniform within one step.
        """
        return math.hypot(self.noise, (self.resolution or 0.0) / math.sqrt(12.0))

    def metadata(self) -> dict:
        return {
            "noise": self.noise,
            "resolution": self.resolution,
            "limits": None if self.limits is None else list(self.limits),
        }


@dataclass(frozen=True)
class Instrument:
    """The readings of the measurements that an instrument observes, by name.

    `observe` returns the samples as the instrument reads them. The noise of a value
    depends only on the seed, on the measurement and on the `sample_id` of its row, so
    the result is the same whatever the order of the rows or the subset asked, and a
    measurement is read the same whatever other measurements the instrument has.
    """

    readings: Mapping[str, Reading]

    def __post_init__(self):
        readings = {
            str(name): reading if isinstance(reading, Reading) else Reading(**reading)
            for name, reading in self.readings.items()
        }
        if not readings:
            raise ValueError("an instrument reads at least one measurement")
        object.__setattr__(self, "readings", readings)

    def observe(self, samples: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
        """A copy of `samples` with the measurements of the instrument as it reads them.

        A value that is missing stays missing: a failed simulation is not read.
        """
        from ..experiments.seeding import sample_stream, text_key

        missing = [name for name in self.readings if name not in samples]
        if missing:
            raise KeyError(f"the samples have no measurement {missing}")
        observed = samples.copy()
        ids = samples["sample_id"].to_numpy(dtype=np.int64)
        for name, reading in self.readings.items():
            values = samples[name].to_numpy(dtype=float).copy()
            if reading.noise and len(ids):
                # one stream per measurement, read at the position of each sample
                stream = sample_stream(seed, text_key(name))
                values = values + reading.noise * stream.standard_normal(int(ids.max()) + 1)[ids]
            if reading.resolution is not None:
                values = np.round(values / reading.resolution) * reading.resolution
            if reading.limits is not None:
                values = np.clip(values, *reading.limits)
            observed[name] = values
        return observed

    def noise_floor(self) -> dict[str, float]:
        """The uncertainty that each reading adds, by measurement: a lower bound for the
        spread of what is observed.
        """
        return {name: reading.uncertainty for name, reading in self.readings.items()}

    def metadata(self) -> dict:
        return {name: reading.metadata() for name, reading in self.readings.items()}

    @classmethod
    def from_metadata(cls, record: dict) -> Instrument:
        return cls({
            name: Reading(r["noise"], r["resolution"],
                          None if r["limits"] is None else tuple(r["limits"]))
            for name, r in record.items()
        })  # fmt: skip
