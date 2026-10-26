"""Test design: choose which nodes, stimuli and readings a self-test uses.

- `candidates`: candidate tests (node × stimulus × feature) and the experiment that
  simulates every drawn circuit under every stimulus;
- `metrics`: detection, false alarms and the coverage metrics of IEC 61508 and
  ISO 26262 with certified bounds, and localisation; `metrics.adaptive`, the stopping
  rules that simulate until a coverage claim is decided;
- `select`: test selection, with published methods in `select.baselines`.
"""

from . import candidates, metrics, select
from .candidates import Feature, Stimulus, Test
from .metrics import Detection, GaussianLimits

__all__ = [
    "Detection",
    "Feature",
    "GaussianLimits",
    "Stimulus",
    "Test",
    "candidates",
    "metrics",
    "select",
]
