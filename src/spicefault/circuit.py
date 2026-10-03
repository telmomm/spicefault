"""The circuit under investigation: a SPICE netlist seen as components, nodes and parameters."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from .netlist import Netlist


@dataclass(frozen=True)
class Component:
    name: str
    kind: str  # SPICE element letter: R, C, L, V, X, ...
    nodes: tuple[str, ...]  # empty if the terminals of this element type are not known
    parameters: dict[str, float]  # numeric ones only: `value`, `dc`, instance parameters


class Circuit:
    """A nominal circuit. It is never modified: every simulation works on a copy.

    Only the top level of the netlist is exposed; what is inside a subcircuit is
    reached through the parameters of its instances.
    """

    def __init__(self, netlist: str, name: str = "circuit"):
        self.name = name
        self._text = netlist

    @classmethod
    def from_netlist(cls, path: str | Path, name: str | None = None) -> Circuit:
        path = Path(path)
        return cls(path.read_text(), name or path.stem)

    def to_netlist(self) -> str:
        return self._text

    def netlist(self) -> Netlist:
        """A fresh, editable copy of the netlist."""
        return Netlist(self._text)

    def components(self) -> list[Component]:
        net = self.netlist()
        found = []
        for name in net.components():
            try:
                nodes = tuple(net.nodes(name))
            except NotImplementedError:
                nodes = ()
            found.append(Component(name, name[0].upper(), nodes, net.parameters(name)))
        return found

    def component(self, name: str) -> Component:
        for component in self.components():
            if component.name.lower() == name.lower():
                return component
        raise KeyError(f"no component named {name!r}")

    def nodes(self) -> list[str]:
        return sorted({node for c in self.components() for node in c.nodes})

    def parameters(self) -> dict[tuple[str, str], float]:
        """(component, parameter) -> nominal value, for every numeric parameter."""
        return {
            (c.name, parameter): value
            for c in self.components()
            for parameter, value in c.parameters.items()
        }

    def metadata(self) -> dict:
        digest = hashlib.sha256(self._text.encode()).hexdigest()
        return {"circuit": self.name, "netlist_sha256": digest}

    def __repr__(self) -> str:
        return f"Circuit({self.name!r}, {len(self.components())} components)"
