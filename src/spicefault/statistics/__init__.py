"""Statistics of a population of simulated circuits: distribution, quantiles and yield.

They are computed from the samples of a dataset, with nothing simulated again, and do
not need faults. Definitions and estimators are in docs/STATISTICS.md.
"""

from .describe import describe, ecdf, quantile_interval
from .yields import samples_for_half_width, yield_report, zero_failure_bound

__all__ = [
    "describe",
    "ecdf",
    "quantile_interval",
    "samples_for_half_width",
    "yield_report",
    "zero_failure_bound",
]
