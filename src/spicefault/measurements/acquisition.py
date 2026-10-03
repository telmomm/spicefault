"""Model of the acquisition chain, applied to noise-free simulation results."""

from __future__ import annotations

import numpy as np


def quantise(v: np.ndarray, vmin: float, vmax: float, bits: int) -> np.ndarray:
    """Clip to the ADC range and round to the nearest code, returning volts."""
    vmin, vmax = float(vmin), float(vmax)
    lsb = (vmax - vmin) / (2 ** int(bits) - 1)
    return vmin + np.round((np.clip(v, vmin, vmax) - vmin) / lsb) * lsb
