"""Test design: choose which nodes, stimuli and readings a self-test uses.

- `candidates`: candidate tests (node × stimulus × feature) and the experiment that
  simulates every drawn circuit under every stimulus;
- `metrics`: detection, false alarms and the coverage metrics of IEC 61508 and
  ISO 26262 with certified bounds, and localisation; `metrics.adaptive`, the stopping
  rules that simulate until a coverage claim is decided; `metrics.formulation`, a plan
  chosen on some circuits and scored and certified on others;
- `costmodel`: what a microcontroller needs to run a plan, what its ADC reads, and what
  the plan adds to the board in euros and test time;
- `faultprior`: the failure rate, severity weight and classes of each fault;
- `select`: test selection, with published methods in `select.baselines`, a greedy
  under a hardware cost and the Pareto front in `select.pareto`, and the exact
  minimum-cost plan in `select.ilp` (with the extra `spicefault[ilp]`);
- `bounds`: the sensitivity matrix, testability and ambiguity groups: what any plan
  can identify at best.
"""

from . import bounds, candidates, costmodel, faultprior, metrics, select
from .candidates import Feature, Stimulus, Test
from .metrics import Detection, GaussianLimits

__all__ = [
    "Detection",
    "Feature",
    "GaussianLimits",
    "Stimulus",
    "Test",
    "bounds",
    "candidates",
    "costmodel",
    "faultprior",
    "metrics",
    "select",
]
