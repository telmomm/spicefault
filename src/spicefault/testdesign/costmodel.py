"""Cost and instrument model of an embedded self-test.

A self-test plan is a set of candidate tests (`candidates.Test`); this module says what
it takes for a microcontroller to run them and what its ADC actually reads:

- the microcontroller: ADC (range, noise from SINAD, static errors, source impedance it
  tolerates at each sampling time) and DAC (output range, rate, settling), with the
  resources the application leaves free for the self-test (`Microcontroller`), and how
  the firmware acquires each kind of feature (`Acquisition`);
- feasibility: whether the DAC can apply a stimulus and the ADC can acquire a feature
  (`feasible`);
- access: whether a node is read directly or needs a buffer, from its source impedance
  (`node_impedance`, `access`);
- observation: the test matrix of `candidates.matrix` as the ADC reads it, so that tests
  whose signal is below the ADC noise (virtual-ground nodes) lose their information
  before any selection (`amplitudes`, `observe`);
- cost: hardware added to the board in euros, and test time, which is kept apart as a
  limit and not added to the cost (`cost`). The prices are inputs (`prices`), not
  package data.

The circuits are assumed biased at mid-range of the ADC, as a single-supply board would
be: a simulated value v at unit stimulus is read as mid + amplitude · v.

`observe` is the instrument model of an embedded ADC as the cost model needs it, per
kind of test; `measurements.Instrument` is the general one, applied to any reading.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..circuit import Circuit
from ..experiments.seeding import sample_stream, text_key
from ..simulation import SimulationConfig, Simulator
from .candidates import Test
from .metrics.detection import HEALTHY

__all__ = [
    "ADC",
    "DAC",
    "STM32G474",
    "Access",
    "Acquisition",
    "DEFAULT_ACQUISITION",
    "Cost",
    "Microcontroller",
    "access",
    "amplitudes",
    "cost",
    "draws",
    "feasible",
    "node_impedance",
    "observe",
    "prices",
]


# --- hardware -------------------------------------------------------------------------


@dataclass(frozen=True)
class ADC:
    """A successive-approximation ADC, single-ended, from 0 to `vref`.

    `sinad_db` is the signal to noise and distortion ratio at full scale, which includes
    the quantisation; `offset_lsb` and `gain_lsb` are the maximum static errors after
    calibration. `max_source` maps a sampling time in clock cycles to the largest source
    impedance (ohm) that settles within it, for fast and for slow channels.
    """

    bits: int
    vref: float
    clock: float
    sinad_db: float
    offset_lsb: float
    gain_lsb: float
    max_source: Mapping[float, tuple[float, float]]

    @property
    def lsb(self) -> float:
        return self.vref / 2**self.bits

    @property
    def noise(self) -> float:
        """Noise per sample, in volts rms, without the quantisation: the SINAD of a
        full-scale sine minus the quantisation that rounding adds afterwards."""
        total = self.vref / (2 * math.sqrt(2)) * 10 ** (-self.sinad_db / 20)
        return math.sqrt(max(total**2 - self.lsb**2 / 12, 0.0))

    def rate(self, cycles: float) -> float:
        """Samples per second with a sampling time of `cycles`."""
        return self.clock / (cycles + self.bits + 0.5)

    def source_limit(self, cycles: float, fast: bool = False) -> float:
        """Largest source impedance that settles in `cycles`, on a fast or slow channel."""
        return self.max_source[cycles][0 if fast else 1]


@dataclass(frozen=True)
class DAC:
    """A buffered DAC output: from `low` to `high` volts, `rate` updates per second and
    `settling` seconds for a full-scale step."""

    bits: int
    low: float
    high: float
    rate: float
    settling: float

    @property
    def amplitude(self) -> float:
        """Largest amplitude around mid-range."""
        return (self.high - self.low) / 2


@dataclass(frozen=True)
class Microcontroller:
    """The ADC and DAC of a microcontroller and the resources free for the self-test:
    ADC channels, internal op-amps that can buffer a node, and DAC outputs."""

    name: str
    adc: ADC
    dac: DAC
    channels: int
    opamps: int
    dac_outputs: int
    source: str = ""


# An example instance. STM32G474, DS12288 Rev 2: Table 69 (range, clock), Table 70
# (maximum RAIN at 12 bits, fast and slow channels), Table 71 (single-ended, single ADC,
# VDDA = VREF+ = 3 V, 25 °C: SINAD typ 65.6 dB, offset max 4 LSB, gain max 6.6 LSB),
# Table 77 (buffered DAC output 0.2 V to VREF+ - 0.2 V, settling max 2.9 µs to 1 LSB) and
# the summary (1 MSPS buffered DAC channels, 6 op-amps in PGA mode). Free channels and
# op-amps are those of a board whose application uses part of the package; they are
# parameters of the case, to be changed with `dataclasses.replace`.
STM32G474 = Microcontroller(
    name="STM32G474",
    adc=ADC(
        bits=12, vref=3.3, clock=60e6, sinad_db=65.6, offset_lsb=4.0, gain_lsb=6.6,
        max_source={2.5: (100, 0), 6.5: (330, 100), 12.5: (680, 470), 24.5: (1500, 1200),
                    47.5: (2200, 1800), 92.5: (4700, 3900), 247.5: (12e3, 10e3),
                    640.5: (39e3, 33e3)},
    ),
    dac=DAC(bits=12, low=0.2, high=3.1, rate=1e6, settling=2.9e-6),
    channels=16,
    opamps=6,
    dac_outputs=3,
    source="DS12288 Rev 2, Tables 69, 70, 71 and 77",
)  # fmt: skip


@dataclass(frozen=True)
class Acquisition:
    """How the firmware acquires each kind of feature.

    Tones are applied with `tone_points` DAC updates per period and read over
    `tone_samples` ADC samples at `fast_cycles`; a second channel reads the stimulus,
    so gain and phase are ratios and the DAC errors cancel. DC levels and step instants
    are averaged over `averages` readings at `slow_cycles`; a step is repeated once per
    average. `dc_settle` is the wait before a DC reading.
    """

    tone_samples: int = 1024
    tone_points: int = 10
    tone_settle_periods: int = 5
    fast_cycles: float = 47.5
    slow_cycles: float = 640.5
    averages: int = 16
    dc_settle: float = 1e-3


# the acquisition used when none is given
DEFAULT_ACQUISITION = Acquisition()


# --- features and feasibility ---------------------------------------------------------


def _feature(test: Test) -> tuple[str, float | None]:
    """The kind of a feature (gain, phase, instant, final, peak, minimum, value) and its
    frequency or instant."""
    name = test.feature.name
    match = re.fullmatch(r"(gain_db|phase|v)@(.+)", name)
    if match:
        kind = {"gain_db": "gain", "phase": "phase", "v": "instant"}[match.group(1)]
        return kind, float(match.group(2))
    return name, None


def feasible(test: Test, mcu: Microcontroller, acq: Acquisition = DEFAULT_ACQUISITION) -> str:
    """Why the microcontroller cannot run `test`, or "" if it can.

    A tone needs `tone_points` DAC updates per period and four ADC samples per period;
    a step instant must come after twice the DAC settling time, since the simulated step
    is ideal.
    """
    kind, at = _feature(test)
    if kind in ("gain", "phase"):
        limit = min(mcu.dac.rate / acq.tone_points, mcu.adc.rate(acq.fast_cycles) / 4)
        if at > limit:
            return f"tone above {limit:g} Hz"
    if kind == "instant" and at < 2 * mcu.dac.settling:
        return f"instant before {2 * mcu.dac.settling:g} s"
    return ""


# --- access ---------------------------------------------------------------------------


def node_impedance(circuit: Circuit, node: str, frequencies: Sequence[float]) -> np.ndarray:
    """|Z| seen at `node`, in ohm, at each frequency: a 1 A AC current is injected with
    every voltage source set to zero AC."""
    # the first line of the netlist is its title, as for any simulation, and is kept
    title, *lines = circuit.to_netlist().splitlines()
    for i, line in enumerate(lines):
        if line.split() and line.split()[0][0].lower() == "v":
            lines[i] = re.sub(r"\bac\s+\S+", "ac 0", line, flags=re.IGNORECASE)
    deck = "\n".join([title, *lines]).replace("\n.end", f"\nIzprobe 0 {node} dc 0 ac 1\n.end")
    frequencies = sorted(float(f) for f in frequencies)
    # a sweep across the frequencies asked, read at each one by interpolation
    sweep = f"ac dec 50 {frequencies[0]:g} {max(frequencies[-1], 1.01 * frequencies[0]):g}"
    config = SimulationConfig(analyses=(sweep,), outputs=("frequency", f"v({node})"))
    result = Simulator().run(deck, config)
    if not result.ok:
        raise RuntimeError(f"impedance of {node}: {result.status}")
    plot = result.plot()
    f, z = np.real(plot["frequency"]), np.abs(plot[f"v({node})"])
    return np.interp(np.log10(frequencies), np.log10(f), z)


@dataclass(frozen=True)
class Access:
    """How a node is read: its worst source impedance and whether it needs a buffer."""

    node: str
    impedance: float
    buffer: bool


def access(
    circuit: Circuit,
    nodes: Sequence[str],
    mcu: Microcontroller,
    acq: Acquisition = DEFAULT_ACQUISITION,
    frequencies: Sequence[float] = (1.0, 1e5),
) -> dict[str, Access]:
    """The access of each node: a buffer when its impedance, at any of `frequencies`,
    exceeds what a slow ADC channel settles at the fast sampling time."""
    limit = mcu.adc.source_limit(acq.fast_cycles)
    found = {}
    for node in nodes:
        z = float(node_impedance(circuit, node, frequencies).max())
        found[node] = Access(node, z, z > limit)
    return found


# --- observation ----------------------------------------------------------------------


def amplitudes(
    matrix: pd.DataFrame, tests: Sequence[Test], mcu: Microcontroller, headroom: float = 0.9
) -> dict[str, float]:
    """Per stimulus, the largest amplitude that keeps every fault-free reading within
    `headroom` of the ADC half-range, and never more than the DAC gives: what a board
    designer would set. The simulated values are at unit stimulus."""
    healthy, found = matrix[matrix.fault_id == HEALTHY], {}
    for name in dict.fromkeys(t.stimulus.name for t in tests):
        group = [t for t in tests if t.stimulus.name == name and _feature(t)[0] != "phase"]
        ids = [t.test_id for t in group]
        if group[0].stimulus.kind == "ac":
            largest = float((10 ** (healthy[ids].max() / 20)).max())
        else:
            largest = float(healthy[ids].abs().max().max())
        found[name] = float(min(mcu.dac.amplitude, headroom * mcu.adc.vref / 2 / largest))
    return found


def draws(seed: int, key: str, faults: np.ndarray, replicas: np.ndarray) -> np.ndarray:
    """One standard normal per row, which depends only on the seed, `key`, the fault and
    the replica of the row: the same row gets the same draw in any order and in any
    matrix that contains it."""
    out = np.empty(len(faults))
    for fault in np.unique(faults):
        rows = faults == fault
        stream = sample_stream(seed, text_key(key), text_key(str(fault)))
        out[rows] = stream.standard_normal(int(replicas[rows].max()) + 1)[replicas[rows]]
    return out


def _cdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1 + np.vectorize(math.erf)(z / math.sqrt(2)))


def observe(
    matrix: pd.DataFrame,
    tests: Sequence[Test],
    mcu: Microcontroller,
    amplitude: float | Mapping[str, float] = 1.0,
    acq: Acquisition = DEFAULT_ACQUISITION,
    seed: int = 0,
) -> pd.DataFrame:
    """A copy of the test matrix with each test as the ADC reads it, in the units of the
    simulation (unit stimulus).

    `amplitude` is the stimulus amplitude in volts, one for all or one per stimulus (see
    `amplitudes`); the simulated values scale with it. Each row is one board: its channel
    at each node has a static offset and gain error, drawn uniformly within the maximum
    of the ADC, and each reading adds the noise of the ADC, averaged over the samples it
    takes. Voltages are rounded to one LSB and clipped to the ADC range; a tone that
    would leave the range is clipped at its edge. Every draw comes from `draws`, so the
    result does not depend on the order of the rows.
    """
    adc, mid = mcu.adc, mcu.adc.vref / 2
    out = matrix.copy()
    faults, replicas = matrix.fault_id.to_numpy(), matrix.replica.to_numpy(dtype=np.int64)

    def uniform(key: str) -> np.ndarray:
        return 2 * _cdf(draws(seed, key, faults, replicas)) - 1

    for test in tests:
        kind, _ = _feature(test)
        a = amplitude if isinstance(amplitude, float | int) else amplitude[test.stimulus.name]
        x = matrix[test.test_id].to_numpy(dtype=float)
        z = draws(seed, test.test_id, faults, replicas)
        gain = 1 + uniform(f"gain|{test.node}") * adc.gain_lsb / 2**adc.bits
        if kind in ("gain", "phase"):
            sigma = adc.noise * math.sqrt(2 / acq.tone_samples)
            if kind == "gain":
                level = a * 10 ** (x / 20)
                read = np.minimum(gain * level, mid) + sigma * z
                out[test.test_id] = 20 * np.log10(np.maximum(np.abs(read), adc.lsb / 2**8) / a)
            else:
                # the phase spread depends on the amplitude of the tone at the node
                twin = test.test_id.replace("|phase@", "|gain_db@")
                level = a * 10 ** (matrix[twin].to_numpy(float) / 20) if twin in matrix else a
                spread = np.degrees(sigma / np.maximum(level, sigma))
                out[test.test_id] = (x + spread * z + 180) % 360 - 180
        else:
            offset = uniform(f"offset|{test.node}") * adc.offset_lsb * adc.lsb
            volts = gain * (mid + a * x) + offset + adc.noise / math.sqrt(acq.averages) * z
            volts = np.clip(np.round(volts / adc.lsb) * adc.lsb, 0, adc.vref - adc.lsb)
            out[test.test_id] = (volts - mid) / a
    return out


# --- cost -----------------------------------------------------------------------------


def prices(path: str | Path) -> dict[str, float]:
    """Price in euros of each item of a price table: a CSV file with the columns `item`
    and `price_eur`, one row for each of external_opamp_channel, mux_8 and
    injection_switch."""
    table = pd.read_csv(path)
    return dict(zip(table["item"], table["price_eur"], strict=True))


@dataclass(frozen=True)
class Cost:
    """Hardware added to the board for a plan, in euros, its bill of items, the test
    time in seconds and the nodes read."""

    euros: float
    items: dict[str, int] = field(default_factory=dict)
    time: float = 0.0
    nodes: tuple[str, ...] = ()


def cost(
    tests: Sequence[Test],
    mcu: Microcontroller,
    accesses: Mapping[str, Access],
    price: Mapping[str, float],
    acq: Acquisition = DEFAULT_ACQUISITION,
) -> Cost:
    """What a plan adds to the board.

    Each node read takes an ADC channel and, if its access needs it, a buffer: an internal
    op-amp while there are free ones, then an external op-amp channel. AC tests take one
    more channel to read the stimulus. Channels beyond the free ones go through 8:1
    multiplexers, each of which takes one channel. Each stimulus source needs a switch
    to inject the DAC output. Time is the sum of the acquisitions, one node at a time.
    """
    nodes = tuple(dict.fromkeys(t.node for t in tests))
    stimuli = {t.stimulus.name: t.stimulus for t in tests}
    items: dict[str, int] = {}
    buffers = sum(accesses[n].buffer for n in nodes)
    items["external_opamp_channel"] = max(buffers - mcu.opamps, 0)
    channels = len(nodes) + any(s.kind == "ac" for s in stimuli.values())
    excess = channels - mcu.channels
    items["mux_8"] = math.ceil(excess / 7) if excess > 0 else 0
    sources = {source for s in stimuli.values() for source, _ in s.settings}
    items["injection_switch"] = len(sources)
    euros = sum(price[item] * n for item, n in items.items())
    return Cost(round(euros, 4), items, _time(tests, mcu, acq), nodes)


def _time(tests: Sequence[Test], mcu: Microcontroller, acq: Acquisition) -> float:
    acquisitions = set()
    for t in tests:
        kind, at = _feature(t)
        if kind in ("gain", "phase"):
            acquisitions.add(("tone", t.node, at))
        elif t.stimulus.kind == "tran":
            acquisitions.add(("step", t.node, t.stimulus.name))
        else:
            acquisitions.add(("dc", t.node, t.stimulus.name))
    fast, slow = mcu.adc.rate(acq.fast_cycles), mcu.adc.rate(acq.slow_cycles)
    durations = {t.stimulus.name: _step_duration(t.stimulus.analysis) for t in tests}
    total = 0.0
    for what, _, at in acquisitions:
        if what == "tone":
            total += acq.tone_settle_periods / at + acq.tone_samples / fast
        elif what == "step":
            total += acq.averages * durations[at]
        else:
            total += acq.dc_settle + acq.averages / slow
    return total


def _step_duration(analysis: str) -> float:
    words = analysis.split()
    return float(words[2]) if words[0] == "tran" else 0.0
