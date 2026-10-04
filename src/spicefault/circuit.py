"""The circuit under investigation: a SPICE netlist seen as components, nodes and parameters."""

from __future__ import annotations

import hashlib
import re
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

    _ANALYSIS = re.compile(r"^\s*\.(op|ac|tran|dc|noise|tf|sens|pz|disto)\b(.*)$", re.I)
    _CONTROL, _ENDC, _END = ".control", ".endc", ".end"
    _SUPPORTED_DIRECTIVES = {
        ".backanno", ".control", ".end", ".endc", ".global", ".include", ".inc", ".lib",
        ".model", ".options", ".param", ".save", ".subckt", ".ends", ".endl", ".temp",
        ".title",
    }

    def __init__(
        self,
        netlist: str,
        name: str = "circuit",
        source_dir: str | Path | None = None,
        imported_analyses=(),
    ):
        self.name = name
        self._analysis_commands: tuple[str, ...] = tuple(imported_analyses)
        self._include_paths: tuple[Path, ...] = ()
        self._unresolved_includes: tuple[str, ...] = ()
        self._include_records: tuple[dict, ...] = ()
        text = netlist.replace("µ", "u").replace("μ", "u")
        text, include_paths, unresolved = self._resolve_includes(text, source_dir)
        text = self._convert_dot_analyses(text)
        self._include_paths = tuple(include_paths)
        self._unresolved_includes = tuple(unresolved)
        self._include_records = tuple(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in self._include_paths
            if path.is_file()
        )
        self._text = text

    @classmethod
    def from_netlist(cls, path: str | Path, name: str | None = None) -> Circuit:
        path = Path(path)
        data = path.read_bytes()
        if data.startswith((b"\xff\xfe", b"\xfe\xff")):
            text = data.decode("utf-16")
        else:
            try:
                text = data.decode("utf-8-sig")
            except UnicodeDecodeError:
                text = data.decode("latin-1")
        return cls(text, name or path.stem, source_dir=path.parent)

    @classmethod
    def _resolve_includes(cls, text: str, source_dir: str | Path | None):
        base = None if source_dir is None else Path(source_dir).resolve()
        paths = []
        unresolved = []
        ending = "\n" if text.endswith("\n") else ""
        lines = [
            cls._resolve_include_line(line, base, paths, unresolved) for line in text.splitlines()
        ]
        return "\n".join(line for line in lines if line is not None) + ending, paths, unresolved

    @classmethod
    def _resolve_include_line(cls, line, base, paths, unresolved):
        stripped = line.strip().lower()
        if stripped == ".backanno":
            return None
        parts = line.lstrip().split(maxsplit=1)
        if len(parts) < 2 or parts[0].lower() not in (".include", ".inc", ".lib"):
            return line
        directive, rest = parts
        quote = rest[0] if rest.startswith(('"', "'")) else ""
        if quote:
            close = rest.find(quote, 1)
            original, suffix = rest[1:close], rest[close + 1 :]
        else:
            original, separator, suffix = rest.partition(" ")
            if directive.lower() == ".lib" and not separator:
                return line
        include = Path(original)
        if include.is_absolute() or base is None:
            candidate = include
        else:
            candidate = base / include
        if not candidate.exists():
            unresolved.append(original)
            return line
        resolved = candidate.resolve()
        paths.append(resolved)
        quote = '"' if any(char.isspace() for char in str(resolved)) else ""
        return f"{directive.lower()} {quote}{resolved}{quote}{suffix}"

    def _convert_dot_analyses(self, text: str) -> str:
        ending = "\n" if text.endswith("\n") else ""
        lines, commands, has_control = self._extract_analyses(text.splitlines())
        if has_control and commands:
            return text
        self._analysis_commands = tuple(commands)
        return "\n".join(lines) + ending

    def _extract_analyses(self, source_lines):
        lines, commands = [], []
        in_control = has_control = False
        for line in source_lines:
            head = line.strip().lower()
            if head.startswith(self._CONTROL):
                in_control = has_control = True
            elif head.startswith(self._ENDC):
                in_control = False
            command = None if in_control else self._dot_analysis_command(line)
            if command is None:
                lines.append(line)
            else:
                commands.append(command)
        return lines, commands, has_control

    @classmethod
    def _dot_analysis_command(cls, line):
        match = cls._ANALYSIS.match(line)
        return None if match is None else f"{match.group(1).lower()}{match.group(2)}".rstrip()

    def to_netlist(self) -> str:
        return self._text

    @property
    def imported_analyses(self) -> tuple[str, ...]:
        """Dot analyses imported from a schematic export, for Experiment to run."""
        return self._analysis_commands

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
        record = {"circuit": self.name, "netlist_sha256": digest}
        if self._include_records:
            record["includes"] = list(self._include_records)
        if self._analysis_commands:
            record["imported_analyses"] = list(self._analysis_commands)
        return record

    def check(self) -> list[str]:
        """Report elements and directives the library cannot inspect or mutate safely."""
        problems = self._component_problems()
        problems.extend(self._directive_problems())
        problems.extend(
            f"included file could not be resolved: {path}" for path in self._unresolved_includes
        )
        problems.extend(
            f"included file is missing: {path}"
            for path in self._include_paths
            if not path.is_file()
        )
        return problems

    def _component_problems(self):
        problems = []
        netlist = self.netlist()
        for component in self.components():
            try:
                netlist.nodes(component.name)
            except NotImplementedError:
                problems.append(f"{component.name}: terminal layout is unsupported")
            if component.kind.lower() in "rcl" and "value" not in component.parameters:
                problems.append(f"{component.name}: no readable numeric value")
            if component.kind.lower() == "c" and any(
                parameter.lower() == "rser" for parameter in component.parameters
            ):
                problems.append(
                    f"{component.name}: Rser is a vendor-specific capacitor parameter; "
                    "ngspice may reject it"
                )
        return problems

    def _directive_problems(self):
        problems = []
        depth = 0
        for line_number, line in enumerate(self._text.splitlines(), start=1):
            head = line.strip().lower()
            if head.startswith(".subckt"):
                depth += 1
            elif head.startswith(".ends"):
                depth = max(0, depth - 1)
            elif head.startswith(".") and not head.startswith(".+"):
                directive = head.split()[0]
                if directive not in self._SUPPORTED_DIRECTIVES and not depth:
                    problems.append(f"line {line_number}: unsupported directive {directive}")
            elif depth and head and not head.startswith(("*", "+")):
                problems.append(f"line {line_number}: subcircuit element is not addressable: "
                                f"{head.split()[0]}")  # fmt: skip
        return problems

    def __repr__(self) -> str:
        return f"Circuit({self.name!r}, {len(self.components())} components)"
