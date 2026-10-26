"""Coverage metrics of a set of tests, with certified intervals.

It reads the test matrix of `testdesign.candidates.matrix` (one row per drawn circuit,
one column per test) and, for a set of tests:

- detection: per fault, P(alarm | fault) with its Clopper–Pearson interval, and the
  false alarm rate on fault-free circuits not used for the limits (`detect`);
- coverage: DC_w, DC, SFF, SPFM and LFM with bounds that hold for all faults at once
  (`coverage`), and the SIL and ASIL they reach;
- localisation: the residual uncertainty H(F | Y) and the ambiguity sets.

The stopping rules that simulate until a claim is decided are in `metrics.adaptive`.
"""

from .coverage import (
    ASIL_TARGETS,
    IEC_CLASSES,
    ISO_CLASSES,
    METRICS,
    SIL_ROUTE_1H,
    asil_reached,
    coverage,
    fault_table,
    metric_values,
    metrics_array,
    sil_reached,
)
from .detection import (
    HEALTHY,
    Detection,
    GaussianLimits,
    angles_of,
    circular_mean,
    detect,
    is_circular,
    split_healthy,
    values,
)
from .localisation import ambiguity_sets, localisation, signatures

__all__ = [
    "ASIL_TARGETS",
    "Detection",
    "GaussianLimits",
    "HEALTHY",
    "IEC_CLASSES",
    "ISO_CLASSES",
    "METRICS",
    "SIL_ROUTE_1H",
    "ambiguity_sets",
    "angles_of",
    "asil_reached",
    "circular_mean",
    "coverage",
    "detect",
    "fault_table",
    "is_circular",
    "localisation",
    "metric_values",
    "metrics_array",
    "signatures",
    "sil_reached",
    "split_healthy",
    "values",
]
