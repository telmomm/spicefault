"""Test selection: which candidate tests a self-test uses.

- `baselines` holds published selection methods, reimplemented as reference
  implementations, so that comparisons between methods can be reproduced.
- `pareto`: the greedy that adds the test with the most coverage per euro added, and
  the cost–coverage Pareto front.
- `ilp`: the exact minimum-cost plan for a coverage target, by integer linear
  programming. It is not imported here, since its solver is an optional dependency:
  ``from spicefault.testdesign.select import ilp``.
"""

from . import baselines, pareto

__all__ = ["baselines", "pareto"]
