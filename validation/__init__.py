"""The validation circuits of the framework (docs/SCIENTIFIC_SCOPE.md, section 8).

Each module defines one study: a netlist, fault rules, measurements and specification
limits. `get(name)` builds it.
"""

from . import biquad, regulator, sallen_key
from .study import Study

STUDIES = {"sallen_key": sallen_key.study, "biquad": biquad.study, "regulator": regulator.study}


def get(name: str) -> Study:
    if name not in STUDIES:
        raise ValueError(f"unknown study {name!r}; available: {sorted(STUDIES)}")
    return STUDIES[name]()


__all__ = ["STUDIES", "Study", "get"]
