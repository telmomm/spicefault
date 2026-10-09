"""Designed samples: circuits whose parameter values are chosen, not drawn.

A design is a table of parameter values, one row per circuit. An experiment that is
given one simulates those circuits with the same engine as drawn ones, so they are
resumable, keep their failed simulations and end in a dataset. What they are not is a
random sample: a proportion over chosen circuits is not a probability, and the dataset
says that its samples were designed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product

from ..circuit import Circuit
from ..variation import Variation, VariationSet

Target = tuple[str, str]


@dataclass(frozen=True)
class Design:
    """Parameter values of the circuits to simulate.

    `targets` are the (component, parameter) pairs and `values` one row per circuit, in
    the order of `targets`. `names` optionally labels each row; the label is stored in
    the column `design_point`. `kind` and `description` say how the design was made
    and are recorded with the dataset.
    """

    targets: Sequence[Target]
    values: Sequence[Sequence[float]]
    names: Sequence[str] | None = None
    kind: str = "table"
    description: str = ""

    def __post_init__(self):
        targets = tuple((str(c), str(p)) for c, p in self.targets)
        values = tuple(tuple(float(v) for v in row) for row in self.values)
        if not targets or len({(c.lower(), p.lower()) for c, p in targets}) != len(targets):
            raise ValueError("a design needs parameters, each named once")
        if not values or any(len(row) != len(targets) for row in values):
            raise ValueError("a design needs rows, each with one value per parameter")
        names = None if self.names is None else tuple(str(name) for name in self.names)
        if names is not None and len(names) != len(values):
            raise ValueError("a design has one name per row")
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "names", names)

    def __len__(self) -> int:
        return len(self.values)

    def row(self, index: int) -> tuple[dict[Target, float], dict[str, object]]:
        """(values, labels) of one circuit of the design."""
        values = dict(zip(self.targets, self.values[index], strict=True))
        return values, ({} if self.names is None else {"design_point": self.names[index]})

    @classmethod
    def from_table(cls, table, kind: str = "table", description: str = "") -> Design:
        """A design from a pandas table whose columns are (component, parameter) pairs,
        or `component` alone for a value, or `component.parameter`.
        """
        targets = []
        for column in table.columns:
            if isinstance(column, tuple):
                targets.append(tuple(column))
            else:
                component, _, parameter = str(column).partition(".")
                targets.append((component, parameter or "value"))
        return cls(targets, table.to_numpy(dtype=float).tolist(), kind=kind,
                   description=description)  # fmt: skip

    def metadata(self) -> dict:
        record = {
            "kind": self.kind,
            "description": self.description,
            "targets": [{"component": c, "parameter": p} for c, p in self.targets],
            "values": [list(row) for row in self.values],
        }
        if self.names is not None:
            record["names"] = list(self.names)
        return record

    @classmethod
    def from_metadata(cls, record: dict) -> Design:
        return cls(
            [(t["component"], t["parameter"]) for t in record["targets"]],
            record["values"],
            record.get("names"),
            record["kind"],
            record.get("description", ""),
        )


def corners(
    variations: VariationSet | Sequence[Variation],
    circuit: Circuit,
    limit: int = 1024,
    nominal: bool = True,
) -> Design:
    """The corners of the band of a set of variations: every combination of the lowest
    and the highest value of each parameter, and the nominal circuit first.

    Each variation must be bounded (a tolerance, a uniform or log-uniform range, a
    truncated normal). n parameters give 2^n corners; more than `limit` is refused,
    since the count doubles with every parameter. Rows are named by the side of each
    parameter, such as `-+-`, and `nominal`.

    The extremes of a measurement over the corners are its extremes over the band only
    if it is monotonic in each parameter there. A response with a maximum or a minimum
    inside the band (a resonance, a notch, a compensated error) has its worst case
    where no corner is, and corners will not show it.
    """
    netlist = circuit.netlist()
    targets, bands, centre = [], [], []
    for variation in variations:
        value = variation.nominal(netlist)
        low, high = variation.bounds(value)
        if low == high:
            continue  # a fixed value has no corner
        targets.append(variation.targets()[0])
        bands.append((low, high))
        centre.append(value)
    if not targets:
        raise ValueError("no variation with a band: there are no corners")
    if 2 ** len(targets) > limit:
        raise ValueError(
            f"{len(targets)} parameters have {2 ** len(targets)} corners, more than the limit "
            f"of {limit}; choose the parameters that matter, or raise the limit"
        )
    values, names = ([centre], ["nominal"]) if nominal else ([], [])
    for sides in product((0, 1), repeat=len(targets)):
        values.append([band[side] for band, side in zip(bands, sides, strict=True)])
        names.append("".join("-+"[side] for side in sides))
    return Design(targets, values, names, kind="corners",
                  description=f"the {2 ** len(targets)} corners of the band of each parameter"
                              + (" and the nominal circuit" if nominal else ""))  # fmt: skip
