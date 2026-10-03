"""Measurements: named functionals of a simulation result, independent of the simulator.

Time-domain statistics are weighted by time. A SPICE transient has a variable time
step, with many points where the signal changes fast, so the plain average of the
samples is not the average of the signal.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ..simulation import Plot, SimulationResult
from .response import interp_response

# short name of an analysis -> start of the name of its plot
_ANALYSES = {
    "op": "Operating Point",
    "ac": "AC Analysis",
    "tran": "Transient Analysis",
    "dc": "DC transfer characteristic",
}


def select_plot(result: SimulationResult, analysis: str | int) -> Plot:
    """Plot by position among the analyses, or the first of a kind (`op`, `ac`, `tran`, `dc`)."""
    if isinstance(analysis, int):
        return result.plot(analysis)
    return result.plot(_ANALYSES.get(analysis.lower(), analysis))


def _signal(plot: Plot, vector: str, window) -> tuple[np.ndarray, np.ndarray]:
    """(time, values) of a transient vector, cut to `window` with interpolated ends."""
    t, x = plot["time"].real, plot[vector.lower()].real
    if window is None:
        return t, x
    t0, t1 = max(window[0], t[0]), min(window[1], t[-1])
    if not t0 < t1:
        raise ValueError(f"window {window} is outside the simulated time {t[0]}..{t[-1]}")
    inside = (t > t0) & (t < t1)
    tw = np.concatenate([[t0], t[inside], [t1]])
    return tw, np.interp(tw, t, x)


_integrate = getattr(np, "trapezoid", None) or np.trapz  # numpy 2 renamed trapz


def _time_average(t: np.ndarray, y: np.ndarray) -> float:
    return float(_integrate(y, t) / (t[-1] - t[0]))


def _mean(plot, vector, window=None):
    return _time_average(*_signal(plot, vector, window))


def _rms(plot, vector, window=None):
    t, x = _signal(plot, vector, window)
    return float(np.sqrt(_time_average(t, x * x)))


def _variance(plot, vector, window=None):
    t, x = _signal(plot, vector, window)
    return _time_average(t, (x - _time_average(t, x)) ** 2)


def _peak(plot, vector, window=None):
    return float(_signal(plot, vector, window)[1].max())


def _minimum(plot, vector, window=None):
    return float(_signal(plot, vector, window)[1].min())


def _peak_to_peak(plot, vector, window=None):
    x = _signal(plot, vector, window)[1]
    return float(x.max() - x.min())


def _value(plot, vector):
    return float(plot[vector.lower()][0].real)


def _final(plot, vector):
    return float(plot[vector.lower()][-1].real)


def _at_time(plot, vector, time):
    t = plot["time"].real
    if not t[0] <= time <= t[-1]:
        raise ValueError(f"time {time} is outside the simulated time {t[0]}..{t[-1]}")
    return float(np.interp(time, t, plot[vector.lower()].real))


def _response(plot, vector, frequency) -> complex:
    f = plot["frequency"].real
    if not f[0] <= frequency <= f[-1]:
        raise ValueError(f"frequency {frequency} is outside the sweep {f[0]}..{f[-1]}")
    return interp_response(f, plot[vector.lower()], frequency)


def _magnitude(plot, vector, frequency, db=False):
    magnitude = abs(_response(plot, vector, frequency))
    return float(20.0 * np.log10(max(magnitude, 1e-300)) if db else magnitude)


def _phase(plot, vector, frequency):
    return float(np.degrees(np.angle(_response(plot, vector, frequency))))


_KINDS: dict[str, Callable[..., float]] = {
    "value": _value,
    "final": _final,
    "at_time": _at_time,
    "mean": _mean,
    "rms": _rms,
    "variance": _variance,
    "peak": _peak,
    "minimum": _minimum,
    "peak_to_peak": _peak_to_peak,
    "magnitude": _magnitude,
    "phase": _phase,
}


@dataclass(frozen=True)
class Measurement:
    """One number read from a simulation result. Build it with the class methods.

    `analysis` says which plot to read: `op`, `ac`, `tran` or `dc` for the first of
    that kind, or the position of the analysis in the simulation configuration.
    """

    name: str
    kind: str
    vector: str
    analysis: str | int
    parameters: dict = field(default_factory=dict)
    function: Callable[[Plot], float] | None = field(default=None, compare=False)

    def __call__(self, result: SimulationResult) -> float:
        plot = select_plot(result, self.analysis)
        if self.function is not None:
            return float(self.function(plot))
        return _KINDS[self.kind](plot, self.vector, **self.parameters)

    def metadata(self) -> dict:
        record = {
            "name": self.name,
            "kind": self.kind,
            "vector": self.vector,
            "analysis": self.analysis,
            "parameters": {k: list(v) if isinstance(v, tuple) else v
                           for k, v in self.parameters.items()},  # fmt: skip
        }
        if self.function is not None:
            record["function"] = getattr(
                self.function, "__qualname__", type(self.function).__qualname__
            )
        return record

    @classmethod
    def from_metadata(cls, record: dict) -> Measurement:
        """Rebuild a measurement from its record; a custom one holds a function and cannot be."""
        if record["kind"] not in _KINDS:
            raise NotImplementedError(
                f"the {record['kind']} measurement {record['name']!r} holds a function and "
                "cannot be rebuilt from its record; pass the measurements explicitly"
            )
        parameters = {
            k: tuple(v) if isinstance(v, list) else v for k, v in record["parameters"].items()
        }
        return cls(record["name"], record["kind"], record["vector"], record["analysis"], parameters)

    @classmethod
    def _build(cls, kind, vector, analysis, name, **parameters) -> Measurement:
        parameters = {k: v for k, v in parameters.items() if v is not None and v is not False}
        return cls(name or f"{kind}_{vector.lower()}", kind, vector, analysis, parameters)

    # --- operating point and single values --------------------------------------------

    @classmethod
    def value(cls, vector: str, analysis: str | int = "op", name: str = "") -> Measurement:
        """The first point of a vector: its DC value in an operating point."""
        return cls._build("value", vector, analysis, name)

    @classmethod
    def final(cls, vector: str, analysis: str | int = "tran", name: str = "") -> Measurement:
        """The last point of a vector."""
        return cls._build("final", vector, analysis, name)

    @classmethod
    def at_time(cls, vector: str, time: float, analysis: str | int = "tran", name: str = ""):
        """x(time), interpolated linearly between the simulated points."""
        return cls._build("at_time", vector, analysis, name, time=time)

    # --- transient statistics, over the whole simulation or a window (t0, t1) ---------

    @classmethod
    def mean(cls, vector: str, window=None, analysis: str | int = "tran", name: str = ""):
        """(1/T) * integral of x dt."""
        return cls._build("mean", vector, analysis, name, window=window)

    @classmethod
    def rms(cls, vector: str, window=None, analysis: str | int = "tran", name: str = ""):
        """sqrt((1/T) * integral of x^2 dt). It includes the DC component."""
        return cls._build("rms", vector, analysis, name, window=window)

    @classmethod
    def variance(cls, vector: str, window=None, analysis: str | int = "tran", name: str = ""):
        """(1/T) * integral of (x - mean)^2 dt."""
        return cls._build("variance", vector, analysis, name, window=window)

    @classmethod
    def peak(cls, vector: str, window=None, analysis: str | int = "tran", name: str = ""):
        """Largest simulated value (not the largest magnitude)."""
        return cls._build("peak", vector, analysis, name, window=window)

    @classmethod
    def minimum(cls, vector: str, window=None, analysis: str | int = "tran", name: str = ""):
        """Smallest simulated value."""
        return cls._build("minimum", vector, analysis, name, window=window)

    @classmethod
    def peak_to_peak(cls, vector: str, window=None, analysis: str | int = "tran", name: str = ""):
        """Largest minus smallest simulated value."""
        return cls._build("peak_to_peak", vector, analysis, name, window=window)

    # --- frequency response -----------------------------------------------------------

    @classmethod
    def magnitude(
        cls,
        vector: str,
        frequency: float,
        analysis: str | int = "ac",
        name: str = "",
        db: bool = False,
    ) -> Measurement:
        """|H(f)|, or 20 log10 |H(f)| with `db`. With a unit AC source this is the gain.

        Interpolated in log-magnitude and phase over log-frequency.
        """
        return cls._build("magnitude", vector, analysis, name, frequency=frequency, db=db)

    @classmethod
    def phase(cls, vector: str, frequency: float, analysis: str | int = "ac", name: str = ""):
        """Phase of H(f) in degrees, in (-180, 180]."""
        return cls._build("phase", vector, analysis, name, frequency=frequency)

    # --- anything else ----------------------------------------------------------------

    @classmethod
    def custom(cls, name: str, function: Callable[[Plot], float], analysis: str | int = 0):
        """`function(plot)` on one plot. Only its name is kept in the provenance."""
        return cls(name, "custom", "", analysis, {}, function)


@dataclass(frozen=True)
class Waveform:
    """A transient vector resampled on a uniform grid: `fs` samples per second from t = 0
    for `duration` seconds. Stored as float32, one row per sample of the experiment.
    """

    vector: str
    fs: float
    duration: float
    analysis: str | int = "tran"

    @property
    def n_points(self) -> int:
        return round(self.duration * self.fs)

    def times(self) -> np.ndarray:
        return np.arange(self.n_points) / self.fs

    def __call__(self, result: SimulationResult) -> np.ndarray:
        plot = select_plot(result, self.analysis)
        samples = np.interp(self.times(), plot["time"].real, plot[self.vector.lower()].real)
        return samples.astype(np.float32)

    def metadata(self) -> dict:
        return {
            "vector": self.vector,
            "fs": self.fs,
            "duration": self.duration,
            "analysis": self.analysis,
            "n_points": self.n_points,
        }

    @classmethod
    def from_metadata(cls, record: dict) -> Waveform:
        return cls(record["vector"], record["fs"], record["duration"], record["analysis"])
