"""Values read from simulated frequency responses and transients."""

from __future__ import annotations

import numpy as np


def interp_response(freq: np.ndarray, h: np.ndarray, f: float) -> complex:
    """Complex response at `f`, interpolating log-magnitude and phase over log-frequency."""
    logf = np.log10(freq)
    mag = np.interp(np.log10(f), logf, np.log(np.maximum(np.abs(h), 1e-300)))
    phase = np.interp(np.log10(f), logf, np.unwrap(np.angle(h)))
    return complex(np.exp(mag) * np.exp(1j * phase))


def resample(t: np.ndarray, time: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Transient `x(time)`, on the simulator's variable time step, read at instants `t`."""
    return np.interp(t, time, x)
