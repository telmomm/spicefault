"""SPICE netlist as text, with the three fault-injection primitives.

A fault is a list of these primitives and nothing else changes the netlist (see
docs/FAULT_MODEL.md):

- `set_parameter`    change a component value or an instance parameter;
- `insert_series`    put a resistance between one terminal and its node;
- `insert_parallel`  put a resistance between the nodes of two terminals.

The text is edited in place, token by token, so everything the primitives do not
touch stays byte for byte as it was written. Only top-level elements can be
addressed: lines inside `.subckt` definitions and `.control` blocks are skipped.

Limitations: instance parameters must be written `name=value` without spaces, and
the terminals of an element are known only for the types in `_TERMINALS` and for
subcircuit instances (X).
"""

from __future__ import annotations

import re

RULES = ("absolute", "relative", "scale", "divide")

# number of terminals a fault can act on; for controlled sources, the output pair
_TERMINALS = {
    "r": 2, "c": 2, "l": 2, "v": 2, "i": 2, "d": 2, "b": 2,
    "e": 2, "g": 2, "f": 2, "h": 2, "q": 3, "j": 3, "m": 4,
}  # fmt: skip
_SUFFIX = {
    "t": 1e12,
    "g": 1e9,
    "meg": 1e6,
    "k": 1e3,
    "mil": 25.4e-6,
    "m": 1e-3,
    "u": 1e-6,
    "n": 1e-9,
    "p": 1e-12,
    "f": 1e-15,
}
_NUMBER = re.compile(r"([+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?)(meg|mil|[tgkmunpf])?[a-z]*", re.I)


def parse_value(token: str) -> float:
    """Number in SPICE notation: `10k`, `1meg`, `4.7uF`, `1e-9`."""
    match = _NUMBER.fullmatch(token)
    if match is None:
        raise ValueError(f"not a number: {token!r}")
    return float(match[1]) * _SUFFIX.get((match[2] or "").lower(), 1.0)


def format_value(value: float) -> str:
    """Shortest text that reads back as the same float."""
    return repr(float(value))


def apply_rule(old: float, rule: str, value: float) -> float:
    """`absolute`: value; `relative`: old * (1 + value); `scale`: old * value; `divide`."""
    if rule == "absolute":
        return float(value)
    if rule == "relative":
        return old * (1.0 + value)
    if rule == "scale":
        return old * value
    if rule == "divide":
        return old / value
    raise ValueError(f"unknown rule {rule!r}; expected one of {RULES}")


class Netlist:
    def __init__(self, text: str):
        self.lines = text.split("\n")

    def __str__(self) -> str:
        return "\n".join(self.lines)

    # --- reading --------------------------------------------------------------------

    def _elements(self) -> dict[str, tuple[int, int]]:
        """Top-level element name (lower case) -> its lines [start, stop)."""
        found: dict[str, tuple[int, int]] = {}
        depth, in_control, current = 0, False, None
        for i, line in enumerate(self.lines[1:], start=1):  # line 0 is the title
            head = line.strip().lower()
            if head.startswith(".control"):
                in_control = True
            elif head.startswith(".endc"):
                in_control = False
            elif head.startswith(".subckt"):
                depth += 1
            elif head.startswith(".ends"):
                depth -= 1
            elif head.startswith("+") and current is not None:
                found[current] = (found[current][0], i + 1)
                continue
            elif head and head[0] not in "*.+" and depth == 0 and not in_control:
                current = head.split()[0]
                found[current] = (i, i + 1)
                continue
            if head and not head.startswith("*"):
                current = None
        return found

    def components(self) -> list[str]:
        """Names of the top-level elements, in order and as written."""
        return [self.lines[start].split()[0] for start, _ in self._elements().values()]

    def _tokens(self, component: str) -> tuple[int, list[tuple[int, int, int]]]:
        """(first line of the element, [(line, start, end)] of each of its tokens)."""
        try:
            start, stop = self._elements()[component.lower()]
        except KeyError:
            raise KeyError(f"no top-level element named {component!r}") from None
        spans = []
        for i in range(start, stop):
            line = self.lines[i]
            skip = line.index("+") + 1 if i > start else 0
            spans += [
                (i, m.start() + skip, m.end() + skip) for m in re.finditer(r"\S+", line[skip:])
            ]
        return start, spans

    def _text(self, span: tuple[int, int, int]) -> str:
        line, a, b = span
        return self.lines[line][a:b]

    def _replace(self, span: tuple[int, int, int], new: str) -> None:
        line, a, b = span
        self.lines[line] = self.lines[line][:a] + new + self.lines[line][b:]

    def _terminals(self, component: str) -> list[tuple[int, int, int]]:
        _, spans = self._tokens(component)
        kind = component[0].lower()
        if kind in _TERMINALS:
            return spans[1 : 1 + _TERMINALS[kind]]
        if kind == "x":
            first_param = next(
                (k for k, s in enumerate(spans) if "=" in self._text(s)), len(spans)
            )
            return spans[1 : first_param - 1]  # the last one is the subcircuit name
        raise NotImplementedError(f"terminals of {component!r}: element type not supported")

    def nodes(self, component: str) -> list[str]:
        """Nodes of the terminals of a component, in netlist order."""
        return [self._text(s) for s in self._terminals(component)]

    def _parameter(self, component: str, parameter: str) -> tuple[tuple[int, int, int], str]:
        """(span of the value text, value text) of a parameter of a component."""
        _, spans = self._tokens(component)
        kind = component[0].lower()
        if parameter == "value":
            if kind not in "rcl":
                raise NotImplementedError(f"{component!r} has no positional value")
            return spans[3], self._text(spans[3])
        if parameter == "dc" and kind in "vi":
            # `V1 a b dc 5 ...` or the bare form `V1 a b 5`
            texts = [self._text(s).lower() for s in spans]
            k = texts.index("dc") + 1 if "dc" in texts else 3
            if k >= len(spans):
                raise KeyError(f"{component!r} has no DC value")
            return spans[k], self._text(spans[k])
        prefix = parameter.lower() + "="
        for line, a, b in spans:
            token = self.lines[line][a:b]
            if token.lower().startswith(prefix):
                return (line, a + len(prefix), b), token[len(prefix) :]
        raise KeyError(f"{component!r} has no parameter {parameter!r}")

    def value(self, component: str, parameter: str = "value") -> float:
        return parse_value(self._parameter(component, parameter)[1])

    def parameters(self, component: str) -> dict[str, float]:
        """Numeric parameters of a component: `value`, `dc` and its `name=value` pairs."""
        _, spans = self._tokens(component)
        names = ["value", "dc"] + [
            self._text(s).partition("=")[0] for s in spans if "=" in self._text(s)
        ]
        found = {}
        for name in names:
            try:
                found[name] = self.value(component, name)
            except (KeyError, ValueError, NotImplementedError, IndexError):
                pass  # absent, or an expression instead of a number
        return found

    def add_directive(self, line: str) -> None:
        """Add a line such as `.options temp=85` before the control block or `.end`."""
        heads = [text.strip().lower() for text in self.lines]
        at = next(
            (i for i, h in enumerate(heads) if i and (h.startswith(".control") or h == ".end")),
            len(self.lines),
        )
        self.lines.insert(at, line)

    # --- primitives -----------------------------------------------------------------

    def set_parameter(self, component: str, parameter: str, rule: str, value: float) -> float:
        """Change a parameter (`"value"` for R, C and L) and return its new value."""
        span, text = self._parameter(component, parameter)
        new = apply_rule(0.0 if rule == "absolute" else parse_value(text), rule, value)
        self._replace(span, format_value(new))
        return new

    def insert_series(
        self,
        component: str,
        terminal: int,
        resistance: float,
        name: str | None = None,
        node: str | None = None,
    ) -> str:
        """Reconnect a terminal (1-based) to its node through a resistance.

        The resistor (`Rser_<component>` by default) is written before the component
        and the new internal node is `<component>_x`. Returns the resistor name.
        """
        name = name or f"Rser_{component}"
        node = node or f"{component}_x"
        if name.lower() in self._elements():
            raise ValueError(f"{name!r} already exists")
        start, _ = self._tokens(component)
        span = self._terminals(component)[terminal - 1]
        original = self._text(span)
        self._replace(span, node)
        self.lines.insert(start, f"{name} {node} {original} {format_value(resistance)}")
        return name

    def insert_parallel(
        self,
        component: str,
        terminal_a: int,
        terminal_b: int,
        resistance: float,
        name: str | None = None,
    ) -> str:
        """Add a resistance between the nodes of two terminals (1-based).

        The resistor (`Rpar_<component>` by default) is written after the component.
        Returns the resistor name.
        """
        name = name or f"Rpar_{component}"
        if name.lower() in self._elements():
            raise ValueError(f"{name!r} already exists")
        nodes = self.nodes(component)
        _, stop = self._elements()[component.lower()]
        self.lines.insert(
            stop,
            f"{name} {nodes[terminal_a - 1]} {nodes[terminal_b - 1]} {format_value(resistance)}",
        )
        return name
