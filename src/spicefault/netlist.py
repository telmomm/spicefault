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
_PLAIN_NUMBER = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?", re.I)
_PARAM_ASSIGNMENT = re.compile(r"([A-Za-z_]\w*)\s*=\s*([^\s,]+)")
_SOURCE_FUNCTIONS = {
    "pulse": ("v1", "v2", "td", "tr", "tf", "pw", "per"),
    "sin": ("vo", "va", "freq", "td", "theta", "phase"),
}


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
        self._index: dict[str, tuple[int, int]] | None = None

    def __str__(self) -> str:
        return "\n".join(self.lines)

    # --- reading --------------------------------------------------------------------

    def _elements(self) -> dict[str, tuple[int, int]]:
        """Top-level element name (lower case) -> its lines [start, stop).

        Kept until a line is inserted: changing a value does not move any element.
        """
        if self._index is None:
            self._index = self._scan()
        return self._index

    def _scan(self) -> dict[str, tuple[int, int]]:
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
            return spans[1 : self._first_parameter(spans) - 1]  # the last is the subcircuit
        raise NotImplementedError(f"terminals of {component!r}: element type not supported")

    def _first_parameter(self, spans: list[tuple[int, int, int]]) -> int:
        """Position of the first `name=value` token of an element; their number if none."""
        return next(
            (
                k
                for k, span in enumerate(spans)
                if "=" in self._text(span)
                or (k + 1 < len(spans) and self._text(spans[k + 1]) == "=")
            ),
            len(spans),
        )

    def model(self, component: str) -> str:
        """The subcircuit of an X instance or the model of a device (D, Q, J, M), as written.

        Empty for the other elements. A bipolar transistor may have a substrate node
        before its model and an area after it; a plain number after the model is
        taken as the area.
        """
        _, spans = self._tokens(component)
        kind = component[0].lower()
        words = [self._text(span) for span in spans[1 : self._first_parameter(spans)]]
        if kind == "x":
            return words[-1] if words else ""
        if kind not in "dqjm":
            return ""
        rest = words[_TERMINALS[kind] :]
        if kind == "q" and len(rest) > 1 and not _PLAIN_NUMBER.fullmatch(rest[1]):
            return rest[1]  # the first one is the substrate node
        return rest[0] if rest else ""

    def nodes(self, component: str) -> list[str]:
        """Nodes of the terminals of a component, in netlist order."""
        return [self._text(s) for s in self._terminals(component)]

    def _parameter(self, component: str, parameter: str) -> tuple[tuple[int, int, int], str]:
        """(span of the value text, value text) of a parameter of a component."""
        _, spans = self._tokens(component)
        kind = component[0].lower()
        parameter = parameter.lower()
        if parameter == "value":
            if kind not in "rcl":
                raise NotImplementedError(f"{component!r} has no positional value")
            return spans[3], self._text(spans[3])
        if parameter == "dc" and kind in "vi":
            return self._dc_parameter(component, spans)
        if kind in "vi":
            if parameter in ("ac", "acmag", "acphase"):
                return self._source_ac_parameter(component, spans, parameter)
            if "." in parameter:
                return self._source_function_parameter(component, spans, parameter)
        return self._named_parameter(component, spans, parameter)

    def _dc_parameter(self, component, spans):
        texts = [self._text(span).lower() for span in spans[3:]]
        index = texts.index("dc") + 4 if "dc" in texts else 3
        if index >= len(spans):
            raise KeyError(f"{component!r} has no DC value")
        return spans[index], self._text(spans[index])

    def _source_ac_parameter(self, component, spans, parameter):
        wanted = int(parameter == "acphase")
        for index, span in enumerate(spans[3:], start=3):
            token = self._text(span)
            if token.lower() == "ac" and index + 1 + wanted < len(spans):
                value_span = spans[index + 1 + wanted]
                return value_span, self._text(value_span).rstrip(")")
            if token.lower().startswith("ac=") and wanted == 0:
                return (span[0], span[1] + 3, span[2]), token[3:]
        raise KeyError(f"{component!r} has no parameter {parameter!r}")

    def _source_function_parameter(self, component, spans, parameter):
        function, argument = parameter.split(".", 1)
        names = _SOURCE_FUNCTIONS.get(function)
        if names is None or argument not in names:
            raise KeyError(f"unsupported source parameter {parameter!r}")
        for index, span in enumerate(spans):
            if self._text(span).lower().startswith(f"{function}("):
                values = self._function_arguments(spans, index, function)
                position = names.index(argument)
                if position < len(values):
                    return values[position]
                break
        raise KeyError(f"{component!r} has no {parameter!r} parameter")

    def _function_arguments(self, spans, index, function):
        values = []
        for arg_index in range(index, len(spans)):
            span = spans[arg_index]
            text = self._text(span)
            start = span[1] + (len(function) + 1 if arg_index == index else 0)
            stop = span[2] - int(text.endswith(")"))
            if stop > start:
                values.append(((span[0], start, stop), self.lines[span[0]][start:stop]))
            if text.endswith(")"):
                break
        return values

    def _named_parameter(self, component, spans, parameter):
        for index, span in enumerate(spans):
            token = self._text(span)
            lowered = token.lower()
            prefix = parameter + "="
            if lowered.startswith(prefix):
                return (span[0], span[1] + len(prefix), span[2]), token[len(prefix) :]
            if (
                lowered == parameter
                and index + 2 < len(spans)
                and self._text(spans[index + 1]) == "="
            ):
                return spans[index + 2], self._text(spans[index + 2])
        raise KeyError(f"{component!r} has no parameter {parameter!r}")

    def _numeric_parameter(self, text: str) -> float:
        try:
            return parse_value(text)
        except ValueError:
            expression = text.strip().strip("{}").strip()
            try:
                return parse_value(expression)
            except ValueError:
                pass
            definitions = {}
            active = False
            for line in self.lines:
                head = line.strip()
                if head.startswith("+") and active:
                    head = head[1:].strip()
                else:
                    active = head.lower().startswith(".param")
                if active:
                    definitions.update(
                        {name.lower(): value for name, value in _PARAM_ASSIGNMENT.findall(head)}
                    )
            value = definitions.get(expression.lower())
            if value is not None:
                return parse_value(value)
            raise

    def value(self, component: str, parameter: str = "value") -> float:
        return self._numeric_parameter(self._parameter(component, parameter)[1])

    def parameters(self, component: str) -> dict[str, float]:
        """Numeric parameters of a component: `value`, `dc` and its `name=value` pairs."""
        _, spans = self._tokens(component)
        kind = component[0].lower()
        names = ["value", "dc", "ac", "acphase"]
        for index, span in enumerate(spans):
            token = self._text(span)
            if "=" in token:
                names.append(token.partition("=")[0])
            elif index + 1 < len(spans) and self._text(spans[index + 1]) == "=":
                names.append(token)
        if kind in "vi":
            for span in spans:
                token = self._text(span).lower()
                function = next(
                    (name for name in _SOURCE_FUNCTIONS if token.startswith(f"{name}(")), None
                )
                if function is not None:
                    names.extend(f"{function}.{name}" for name in _SOURCE_FUNCTIONS[function])
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
        self._index = None
        self.lines.insert(at, line)

    # --- primitives -----------------------------------------------------------------

    def set_parameter(self, component: str, parameter: str, rule: str, value: float) -> float:
        """Change a parameter (`"value"` for R, C and L) and return its new value."""
        span, text = self._parameter(component, parameter)
        new = apply_rule(0.0 if rule == "absolute" else self._numeric_parameter(text), rule, value)
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
        self._index = None
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
        self._index = None
        self.lines.insert(
            stop,
            f"{name} {nodes[terminal_a - 1]} {nodes[terminal_b - 1]} {format_value(resistance)}",
        )
        return name
