"""An ordered set of fault conditions: the fault list of an experiment."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import pandas as pd

from ..circuit import Circuit
from ..variation import ToleranceVariation, VariationSet
from .base import Fault, SetParameter


class FaultSet(Sequence):
    """Faults in a fixed order, each with a unique identifier.

    The order matters: with positional seeding, the random stream of a fault depends
    on its place in the list.
    """

    def __init__(self, faults: Iterable[Fault] = ()):
        self._faults = tuple(faults)
        self._by_id = {f.fault_id: f for f in self._faults}
        if len(self._by_id) != len(self._faults):
            seen: set[str] = set()
            repeated = sorted({f.fault_id for f in self._faults if f.fault_id in seen
                               or seen.add(f.fault_id)})  # fmt: skip
            raise ValueError(f"repeated fault identifiers: {repeated}")

    def __len__(self) -> int:
        return len(self._faults)

    def __getitem__(self, key: int | slice | str):
        if isinstance(key, str):
            return self._by_id[key]
        if isinstance(key, slice):
            return FaultSet(self._faults[key])
        return self._faults[key]

    def __contains__(self, item: object) -> bool:
        if isinstance(item, str):
            return item in self._by_id
        return item in self._faults

    def __add__(self, other: Iterable[Fault]) -> FaultSet:
        return FaultSet([*self._faults, *other])

    def __eq__(self, other: object) -> bool:
        return isinstance(other, FaultSet) and self._faults == other._faults

    def __repr__(self) -> str:
        return f"FaultSet({len(self)} faults)"

    def ids(self) -> list[str]:
        """The identifiers of the faults, in order."""
        return list(self._by_id)

    def select(self, ids: Iterable[str]) -> FaultSet:
        """The faults with these identifiers, in the order given."""
        return FaultSet(self._by_id[i] for i in ids)

    def filter(
        self,
        fault_type: str | None = None,
        component: str | None = None,
        tags: dict[str, str] | None = None,
        where: Callable[[Fault], bool] | None = None,
    ) -> FaultSet:
        """Faults that meet every condition given, in their original order."""

        def keep(f: Fault) -> bool:
            return (
                (fault_type is None or f.fault_type == fault_type)
                and (component is None or component.lower() in (c.lower() for c in f.components))
                and all(f.tags.get(k) == v for k, v in (tags or {}).items())
                and (where is None or where(f))
            )

        return FaultSet(f for f in self._faults if keep(f))

    # --- serialisation ----------------------------------------------------------------

    def metadata(self) -> list[dict]:
        return [f.metadata() for f in self._faults]

    @classmethod
    def from_metadata(cls, records: Iterable[dict]) -> FaultSet:
        return cls(Fault.from_metadata(record) for record in records)

    def to_json(self, path: str | Path) -> None:
        """Write the records of the faults to a file; `from_json` reads them back."""
        Path(path).write_text(json.dumps(self.metadata(), indent=2))

    @classmethod
    def from_json(cls, path: str | Path) -> FaultSet:
        return cls.from_metadata(json.loads(Path(path).read_text()))

    # --- validation -------------------------------------------------------------------

    def tolerance_overlap(
        self, variations: VariationSet, circuit: Circuit | None = None
    ) -> pd.DataFrame:
        """How much of each parametric fault population lies inside the healthy band.

        A fault on a parameter with tolerance t moves the realised value, so the total
        deviation from nominal covers an interval; where it overlaps +-t, faulty
        samples are healthy by definition and nothing can detect them. One row per
        fault made of a single parameter change on a toleranced parameter:

        - `deviation_min`, `deviation_max`: total relative deviation from nominal;
        - `inside_fraction`: share of the fault population within +-t, for the
          distribution of the tolerance.

        Faults that set an absolute value need `circuit`, for the nominal value. This
        is a report: an overlapping fault is legitimate, but it must be visible.
        """
        tolerances = {
            (v.component.lower(), v.parameter.lower()): v
            for v in variations.variations
            if isinstance(v, ToleranceVariation) and v.relative
        }
        rows = []
        for fault in self._faults:
            if len(fault.primitives) != 1 or not isinstance(fault.primitives[0], SetParameter):
                continue
            change = fault.primitives[0]
            variation = tolerances.get((change.component.lower(), change.parameter.lower()))
            if variation is None:
                continue
            t = variation.tolerance
            if change.rule == "absolute":
                if circuit is None:
                    continue
                nominal = circuit.netlist().value(change.component, change.parameter)
                if nominal == 0:
                    continue
                low = high = change.value / nominal - 1.0
                inside = float(abs(low) <= t)
            else:
                k = {"relative": 1.0 + change.value, "scale": change.value}.get(
                    change.rule, 1.0 / change.value
                )
                low, high = sorted(((1.0 - t) * k - 1.0, (1.0 + t) * k - 1.0))
                inside = _inside_fraction(k, t, variation.distribution)
            rows.append(
                {
                    "fault_id": fault.fault_id,
                    "component": change.component,
                    "parameter": change.parameter,
                    "tolerance": t,
                    "deviation_min": low,
                    "deviation_max": high,
                    "inside_fraction": inside,
                }
            )
        columns = ["fault_id", "component", "parameter", "tolerance", "deviation_min",
                   "deviation_max", "inside_fraction"]  # fmt: skip
        return pd.DataFrame(rows, columns=columns)


def _inside_fraction(k: float, t: float, distribution: str) -> float:
    """P(|(1 + t u) k - 1| <= t) for the unit deviation u of a tolerance draw."""
    if t == 0 or k == 0:
        return float(k == 1.0)
    # the band |x / x0 - 1| <= t, written as an interval of u
    a, b = sorted((((1.0 - t) / k - 1.0) / t, ((1.0 + t) / k - 1.0) / t))
    a, b = max(a, -1.0), min(b, 1.0)
    if a >= b:
        return 0.0
    if distribution == "uniform":
        return (b - a) / 2.0

    def cdf(u: float) -> float:  # normal with sigma = 1/3
        return 0.5 * (1.0 + math.erf(3.0 * u / math.sqrt(2.0)))

    return (cdf(b) - cdf(a)) / (cdf(1.0) - cdf(-1.0))
