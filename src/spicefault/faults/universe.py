"""The fault universe of a circuit and the coverage of a campaign over it.

The universe is generated from applicability rules: which fault types apply to which
components, and with which magnitudes. A campaign simulates a selection of it, and
every fault left out carries a reason. That record is what makes the coverage of a
campaign auditable (docs/FAULT_MODEL.md, section 8).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import pandas as pd

from ..circuit import Circuit, Component
from .base import Fault
from .faultset import FaultSet
from .types import LeakageFault, OpenCircuit, ParametricFault, SeriesResistanceFault, ShortCircuit


@dataclass(frozen=True)
class FaultRule:
    """Faults of one type (`name`, a column of the coverage matrix) on the components
    it applies to: those whose element letter is in `kinds` (any, if empty), whose
    name is listed if `components` is given, and whose subcircuit or device model
    (`Component.model`) is listed if `models` is given.

    `tags` are added to every fault of the rule: a dictionary, or a function of the
    component that returns one, for labels that depend on where the fault is (the
    part to locate, the origin of the defect).
    """

    name: str
    make: Callable[[Component], Sequence[Fault]]
    kinds: str = ""
    components: tuple[str, ...] | None = None
    models: tuple[str, ...] | None = None
    tags: dict[str, str] | Callable[[Component], dict[str, str]] | None = None

    def applies(self, component: Component) -> bool:
        if self.kinds and component.kind not in self.kinds.upper():
            return False
        if self.models is not None and component.model.lower() not in (
            model.lower() for model in self.models
        ):
            return False
        if self.components is None:
            return True
        return component.name.lower() in (name.lower() for name in self.components)

    def faults(self, component: Component) -> list[Fault]:
        """The faults of the rule on a component, with the tags of the rule."""
        made = list(self.make(component))
        if self.tags is None:
            return made
        tags = self.tags(component) if callable(self.tags) else self.tags
        return [fault.with_tags(tags) for fault in made]


def open_rule(kinds: str = "RCL", r_open: float = 1e9, **kwargs) -> FaultRule:
    """An open at the second terminal of every component of these kinds."""
    return FaultRule("open", lambda c: [OpenCircuit(c.name, r_open=r_open)], kinds, **kwargs)


def short_rule(kinds: str = "RCL", r_short: float = 1.0, **kwargs) -> FaultRule:
    return FaultRule("short", lambda c: [ShortCircuit(c.name, r_short=r_short)], kinds, **kwargs)


def parametric_rule(
    deviations: Sequence[float] = (),
    kinds: str = "RCL",
    parameter: str = "value",
    *,
    factors: Sequence[float] = (),
    fault_values: Sequence[float] = (),
    fault_type: str = "parametric",
    **kwargs,
) -> FaultRule:
    """One fault of `parameter` per relative deviation, per factor and per absolute value.

    `deviations` and `factors` act on the value the component has after its normal
    variation; `fault_values` replace it (see `ParametricFault`), with a severity
    relative to the nominal value when that is not zero. `fault_type` is the type the
    faults are reported under and the name of the rule, so that several parametric
    rules (an offset, a gain) can share a universe.
    """
    if not (len(deviations) or len(factors) or len(fault_values)):
        raise ValueError("a parametric rule needs deviations, factors or fault_values")

    def make(component: Component) -> list[Fault]:
        nominal = {name.lower(): v for name, v in component.parameters.items()}.get(
            parameter.lower()
        )
        common = {"fault_type": fault_type}
        return [
            *(ParametricFault(component.name, parameter, deviation=d, **common)
              for d in deviations),
            *(ParametricFault(component.name, parameter, factor=f, **common) for f in factors),
            *(
                ParametricFault(
                    component.name, parameter, fault_value=v, nominal_value=nominal, **common
                )
                for v in fault_values
            ),
        ]  # fmt: skip

    return FaultRule(fault_type, make, kinds, **kwargs)


def leakage_rule(resistances: Sequence[float], kinds: str = "C", **kwargs) -> FaultRule:
    """One fault per leakage resistance; severity on the log scale they span."""
    r_min, r_max = min(resistances), max(resistances)
    span = {} if r_min == r_max else {"r_min": r_min, "r_max": r_max}
    return FaultRule(
        "leakage", lambda c: [LeakageFault(c.name, r, **span) for r in resistances], kinds, **kwargs
    )


def series_rule(resistances: Sequence[float], kinds: str = "RCL", **kwargs) -> FaultRule:
    """One graded series-resistance fault per magnitude; severity grows with resistance."""
    r_min, r_max = min(resistances), max(resistances)
    span = {} if r_min == r_max else {"r_min": r_min, "r_max": r_max}
    return FaultRule(
        "series_resistance",
        lambda c: [SeriesResistanceFault(c.name, r, **span) for r in resistances],
        kinds,
        **kwargs,
    )


class FaultUniverse:
    def __init__(self, circuit: Circuit, rules: Sequence[FaultRule]):
        names = [rule.name for rule in rules]
        if len(names) != len(set(names)):
            raise ValueError(f"repeated rule names: {names}")
        self.circuit, self.rules = circuit, tuple(rules)
        self._cells: dict[tuple[str, str], list[str]] = {}  # (component, rule) -> fault ids
        self._excluded: dict[str, str] = {}  # fault id -> reason
        faults = []
        for component in circuit.components():
            for rule in self.rules:
                if rule.applies(component):
                    made = rule.faults(component)
                    self._cells[component.name, rule.name] = [f.fault_id for f in made]
                    faults += made
        self.faults = FaultSet(faults)

    def __len__(self) -> int:
        return len(self.faults)

    def exclude(self, which: str | Iterable[str] | Callable[[Fault], bool], reason: str) -> int:
        """Leave faults out of the campaign, by identifier or by predicate, with a reason.

        Returns how many were excluded by this call.
        """
        if not reason:
            raise ValueError("an exclusion needs a reason")
        if callable(which):
            ids = [f.fault_id for f in self.faults if which(f)]
        else:
            ids = [which] if isinstance(which, str) else list(which)
            unknown = [i for i in ids if i not in self.faults]
            if unknown:
                raise KeyError(f"not in the universe: {unknown}")
        new = [i for i in ids if i not in self._excluded]
        self._excluded.update({i: reason for i in new})
        return len(new)

    def selected(self) -> FaultSet:
        """The faults to simulate, in universe order."""
        return FaultSet(f for f in self.faults if f.fault_id not in self._excluded)

    def exclusions(self) -> pd.DataFrame:
        return pd.DataFrame(
            [{"fault_id": i, "reason": reason} for i, reason in self._excluded.items()],
            columns=["fault_id", "reason"],
        )

    def coverage(self) -> float:
        """Structural coverage: simulated conditions over conditions in the universe."""
        return 1.0 - len(self._excluded) / len(self.faults) if len(self.faults) else float("nan")

    def coverage_matrix(self, counts: str = "selected") -> pd.DataFrame:
        """Components x fault types. NaN where the fault type does not apply.

        `counts`: `selected` (conditions to simulate), `universe` (conditions that
        exist) or `fraction` (their ratio).
        """
        if counts not in ("selected", "universe", "fraction"):
            raise ValueError(f"unknown counts {counts!r}")
        components = [c.name for c in self.circuit.components()]
        table = pd.DataFrame(
            float("nan"), index=components, columns=[r.name for r in self.rules]
        ).rename_axis(index="component")
        for (component, rule), ids in self._cells.items():
            kept = sum(i not in self._excluded for i in ids)
            table.loc[component, rule] = {
                "selected": kept,
                "universe": len(ids),
                "fraction": kept / len(ids) if ids else float("nan"),
            }[counts]
        return table.loc[table.notna().any(axis=1)]

    def metadata(self) -> dict:
        """The record that makes the coverage auditable."""
        return {
            "rules": [rule.name for rule in self.rules],
            "n_universe": len(self.faults),
            "n_selected": len(self.faults) - len(self._excluded),
            "coverage": self.coverage(),
            "cells": [
                {"component": c, "fault_type": r, "fault_ids": ids}
                for (c, r), ids in self._cells.items()
            ],
            "exclusions": [{"fault_id": i, "reason": r} for i, r in self._excluded.items()],
            "components_without_faults": [
                c.name
                for c in self.circuit.components()
                if not any((c.name, r.name) in self._cells for r in self.rules)
            ],
        }
