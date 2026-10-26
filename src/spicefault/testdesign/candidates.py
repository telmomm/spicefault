"""Candidate tests: node × stimulus × feature, enumerated from a netlist.

A test is one number a self-test can read: a feature (a `Measurement`) of the voltage
at an accessible node while the circuit is driven by a stimulus. Each stimulus becomes
an `OperatingCondition` with its own analysis, so one `Experiment` simulates every
drawn circuit under every stimulus and the tests of different stimuli can be combined
on the same circuit. Nothing here changes the experiment engine.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

import pandas as pd

from ..circuit import Circuit
from ..conditions import OperatingCondition
from ..measurements import Measurement
from ..simulation import SimulationConfig

GROUND = "0"


@dataclass(frozen=True)
class Feature:
    """A named feature of a node voltage: `build(vector, name)` returns its Measurement."""

    name: str
    build: Callable[[str, str], Measurement] = field(compare=False)


@dataclass(frozen=True)
class Stimulus:
    """How the circuit is driven: one analysis and the source settings it needs.

    `settings` maps (source, parameter) to a value, as in an `OperatingCondition`;
    `features` are the features read at every node under this stimulus; `kind` is the
    type of analysis (ac, tran or op).
    """

    name: str
    analysis: str
    settings: dict[tuple[str, str], float] = field(default_factory=dict)
    features: tuple[Feature, ...] = ()
    kind: str = ""


@dataclass(frozen=True)
class Test:
    """One candidate test. Its `test_id` is also the name of its measurement column."""

    stimulus: Stimulus
    node: str
    feature: Feature

    @property
    def test_id(self) -> str:
        return f"{self.stimulus.name}|{self.node}|{self.feature.name}"

    @property
    def vector(self) -> str:
        return f"v({self.node})"

    def measurement(self) -> Measurement:
        return self.feature.build(self.vector, self.test_id)


# --- stimuli -------------------------------------------------------------------------


def _label(value: float) -> str:
    return f"{value:g}"


def _gain(frequency: float) -> Feature:
    return Feature(
        f"gain_db@{_label(frequency)}",
        lambda v, n: Measurement.magnitude(v, frequency, name=n, db=True),
    )


def _phase(frequency: float) -> Feature:
    return Feature(
        f"phase@{_label(frequency)}", lambda v, n: Measurement.phase(v, frequency, name=n)
    )


def _at_time(time: float) -> Feature:
    return Feature(f"v@{_label(time)}", lambda v, n: Measurement.at_time(v, time, name=n))


def ac_tones(
    source: str,
    frequencies: Sequence[float],
    points_per_decade: int = 50,
    gain: bool = True,
    phase: bool = True,
    name: str = "ac",
) -> Stimulus:
    """Sine tones of unit amplitude at `frequencies`, read as gain (dB) and phase (degrees).

    One AC sweep covers all the tones; each tone is read by interpolation.
    """
    frequencies = sorted(float(f) for f in frequencies)
    features = []
    for f in frequencies:
        if gain:
            features.append(_gain(f))
        if phase:
            features.append(_phase(f))
    # half a decade beyond the outer tones, so that even a single tone has points around it
    sweep = (
        f"ac dec {points_per_decade} {_label(frequencies[0] / 10**0.5)} "
        f"{_label(frequencies[-1] * 10**0.5)}"
    )
    return Stimulus(name, sweep, {(source, "ac"): 1.0}, tuple(features), "ac")


def step(
    source: str,
    amplitude: float,
    duration: float,
    times: Sequence[float],
    rise: float = 1e-7,
    points: int = 2000,
    name: str = "step",
) -> Stimulus:
    """A step of `amplitude` volts at t = 0, read at `times`, plus its final value, peak
    and minimum.

    The source needs a `pulse(...)` function: see `prepare_source`.
    """
    settings = {
        (source, "pulse.v1"): 0.0,
        (source, "pulse.v2"): float(amplitude),
        (source, "pulse.td"): 0.0,
        (source, "pulse.tr"): rise,
        (source, "pulse.tf"): rise,
        (source, "pulse.pw"): 10 * duration,
        (source, "pulse.per"): 20 * duration,
    }
    features = [_at_time(t) for t in sorted(times)]
    features += [
        Feature("final", lambda v, n: Measurement.final(v, name=n)),
        Feature("peak", lambda v, n: Measurement.peak(v, name=n)),
        Feature("minimum", lambda v, n: Measurement.minimum(v, name=n)),
    ]
    tran = f"tran {_label(duration / points)} {_label(duration)}"
    return Stimulus(name, tran, settings, tuple(features), "tran")


def dc_levels(source: str, levels: Sequence[float], name: str = "dc") -> list[Stimulus]:
    """One operating-point stimulus per DC level of the source, read as the node voltage."""
    feature = Feature("value", lambda v, n: Measurement.value(v, name=n))
    return [
        Stimulus(f"{name}{_label(level)}", "op", {(source, "dc"): float(level)}, (feature,), "op")
        for level in levels
    ]


def prepare_source(circuit: Circuit, source: str) -> Circuit:
    """The circuit with `pulse(...)` added to `source`, so that step stimuli can set it.

    The pulse is flat at 0 V, so it changes nothing in AC and DC analyses. A source that
    already has a pulse is left as it is.
    """
    lines = circuit.to_netlist().splitlines()
    for i, line in enumerate(lines):
        tokens = line.split()
        if tokens and tokens[0].lower() == source.lower():
            if "pulse(" not in line.lower():
                lines[i] = f"{line} pulse(0 0 0 1n 1n 1 2)"
            break
    else:
        raise KeyError(f"no source {source!r} in the netlist")
    return Circuit("\n".join(lines) + "\n", name=circuit.name)


# --- nodes and tests -----------------------------------------------------------------


def _natural(node: str) -> list:
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", node)]


def accessible_nodes(circuit: Circuit, exclude: Iterable[str] = ()) -> list[str]:
    """Top-level nodes that a test could read: all but ground, the nodes that a voltage
    source fixes (the stimulus input and the supply rails) and `exclude`, in natural order.
    """
    fixed = {node for c in circuit.components() if c.kind == "V" for node in c.nodes}
    skip = fixed | {GROUND} | {str(n) for n in exclude}
    return sorted((node for node in circuit.nodes() if node not in skip), key=_natural)


def enumerate_tests(nodes: Sequence[str], stimuli: Sequence[Stimulus]) -> list[Test]:
    """Every node × stimulus × feature, in that order of stimulus, node and feature."""
    names = [s.name for s in stimuli]
    if len(set(names)) != len(names):
        raise ValueError(f"repeated stimulus names: {names}")
    return [Test(s, node, f) for s in stimuli for node in nodes for f in s.features]


def conditions(tests: Sequence[Test]) -> tuple[OperatingCondition, ...]:
    """One operating condition per stimulus, with its analysis and the measurements of
    its tests, in the order in which the stimuli first appear.
    """
    by_stimulus: dict[str, list[Test]] = {}
    for test in tests:
        by_stimulus.setdefault(test.stimulus.name, []).append(test)
    found = []
    for name, group in by_stimulus.items():
        stimulus = group[0].stimulus
        outputs = tuple(dict.fromkeys(t.vector for t in group))
        if stimulus.kind == "ac":
            outputs = ("frequency", *outputs)
        elif stimulus.kind == "tran":
            outputs = ("time", *outputs)
        found.append(
            OperatingCondition(
                name=name,
                settings=dict(stimulus.settings),
                config=SimulationConfig(analyses=(stimulus.analysis,), outputs=outputs),
                measurements=tuple(t.measurement() for t in group),
            )
        )
    return tuple(found)


def listing(tests: Sequence[Test]) -> pd.DataFrame:
    """The tests as a table: id, stimulus, kind, node and feature."""
    return pd.DataFrame(
        [
            {
                "test_id": t.test_id,
                "stimulus": t.stimulus.name,
                "kind": t.stimulus.kind,
                "node": t.node,
                "feature": t.feature.name,
            }
            for t in tests
        ]
    )


def _ids(tests: Sequence[Test | str]) -> list[str]:
    return [t if isinstance(t, str) else t.test_id for t in tests]


def matrix(frame: pd.DataFrame, tests: Sequence[Test | str]) -> pd.DataFrame:
    """One row per drawn circuit (fault and replica) and one column per test.

    `frame` is `ExperimentResult.to_frame()`: it has one row per circuit and stimulus.
    A circuit whose simulation failed under any stimulus is dropped, since its tests
    cannot be combined.
    """
    ids = _ids(tests)
    keys = ["fault_id", "replica"]
    ok = frame.groupby(keys)["status"].transform(lambda s: (s == "SUCCESS").all())
    merged = frame[ok].groupby(keys, sort=False)[ids].first()
    return merged.reset_index()


def constant_tests(
    matrix: pd.DataFrame, tests: Sequence[Test | str], atol: float = 1e-9
) -> list[str]:
    """Tests that read the same value, within `atol`, in every circuit of the matrix: no
    fault changes them, so they carry no information (for example a DC level behind a
    series capacitor).
    """
    ids = _ids(tests)
    spread = matrix[ids].max() - matrix[ids].min()
    return list(spread[spread.abs() <= atol].index)
