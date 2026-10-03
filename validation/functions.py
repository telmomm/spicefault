"""Measurements on a frequency response that the built-in ones do not cover.

They are functions of one AC plot, defined at module level so that worker processes
can receive them. Frequencies are located by interpolation over log-frequency: on the
grid of the sweep they would move in steps, which would look like measurement noise.
"""

from __future__ import annotations

import numpy as np

OUT = "v(out)"


def _response(plot) -> tuple[np.ndarray, np.ndarray]:
    return np.log(plot["frequency"].real), np.log(np.maximum(np.abs(plot[OUT]), 1e-300))


def _peak(plot) -> tuple[float, float]:
    """(log frequency, log magnitude) of the largest response, by a parabola through
    the highest point and its neighbours.
    """
    x, y = _response(plot)
    k = int(np.argmax(y))
    if k in (0, len(y) - 1):
        return float(x[k]), float(y[k])
    a, b, c = np.polyfit(x[k - 1 : k + 2] - x[k], y[k - 1 : k + 2], 2)
    if a >= 0:
        return float(x[k]), float(y[k])
    dx = -b / (2 * a)
    return float(x[k] + dx), float(c - b * b / (4 * a))


def _crossings(x: np.ndarray, y: np.ndarray, level: float) -> np.ndarray:
    """Log frequencies where the response crosses `level`, by linear interpolation."""
    below = y < level
    k = np.flatnonzero(below[:-1] != below[1:])
    return x[k] + (level - y[k]) * (x[k + 1] - x[k]) / (y[k + 1] - y[k])


def centre_frequency(plot) -> float:
    """Frequency of the largest response [Hz]."""
    return float(np.exp(_peak(plot)[0]))


def peak_gain(plot) -> float:
    """Largest response."""
    return float(np.exp(_peak(plot)[1]))


def bandwidth(plot) -> float:
    """Width between the -3 dB points on each side of the peak [Hz]."""
    x, y = _response(plot)
    x0, y0 = _peak(plot)
    crossings = np.exp(_crossings(x, y, y0 - 0.5 * np.log(2.0)))
    lower, upper = crossings[crossings < np.exp(x0)], crossings[crossings > np.exp(x0)]
    if not len(lower) or not len(upper):
        raise ValueError("the response does not fall 3 dB on both sides of its peak")
    return float(upper[0] - lower[-1])


def passband_gain(plot) -> float:
    """Response at the top of the sweep, taken as the pass band of a high-pass filter."""
    return float(np.abs(plot[OUT][-1]))


def corner_frequency(plot) -> float:
    """Lowest frequency at which a high-pass response reaches its pass-band gain - 3 dB [Hz]."""
    x, y = _response(plot)
    crossings = _crossings(x, y, y[-1] - 0.5 * np.log(2.0))
    if not len(crossings):
        raise ValueError("the response never crosses 3 dB below its pass band")
    return float(np.exp(crossings[0]))
