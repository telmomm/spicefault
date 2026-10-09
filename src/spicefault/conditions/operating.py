"""Conditions under which a circuit operates or is tested."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..measurements import Measurement, Waveform
from ..netlist import Netlist
from ..simulation import SimulationConfig


@dataclass(frozen=True)
class OperatingCondition:
    """A named set of deterministic settings, applied after the fault.

    `settings` maps (component, parameter) to an absolute value, for example
    `{("Vcc", "dc"): 3.0}` for a supply or `{("Rload", "value"): 1e3}` for a load.
    `temperature` is the simulation temperature in degrees Celsius; it only has an
    effect on devices whose model depends on temperature.

    `config` and `measurements` replace those of the experiment for this condition.
    `waveform` does the same for the stored waveform: None keeps the waveform of the
    experiment, a `Waveform` stores that one, and False stores none, so the rows of
    the condition hold NaN. Every waveform of an experiment has the same number of
    points, since they share one array.
    """

    name: str = "nominal"
    temperature: float | None = None
    settings: dict[tuple[str, str], float] = field(default_factory=dict)
    config: SimulationConfig | None = None
    measurements: tuple[Measurement, ...] | None = None
    waveform: Waveform | bool | None = None

    def __post_init__(self):
        if self.waveform is True:
            raise ValueError("waveform is a Waveform, False for none, or None for the default")
        if self.measurements is not None:
            object.__setattr__(self, "measurements", tuple(self.measurements))

    def apply(self, netlist: Netlist) -> None:
        for (component, parameter), value in self.settings.items():
            netlist.set_parameter(component, parameter, "absolute", value)
        if self.temperature is not None:
            netlist.add_directive(f".options temp={float(self.temperature)}")

    def metadata(self) -> dict:
        record = {
            "name": self.name,
            "temperature": self.temperature,
            "settings": [
                {"component": c, "parameter": p, "value": v} for (c, p), v in self.settings.items()
            ],
        }
        if self.config is not None:
            record["config"] = self.config.metadata()
        if self.measurements is not None:
            record["measurements"] = [measurement.metadata() for measurement in self.measurements]
        if self.waveform is not None:
            record["waveform"] = self.waveform.metadata() if self.waveform else False
        return record

    @classmethod
    def from_metadata(cls, record: dict) -> OperatingCondition:
        settings = {(s["component"], s["parameter"]): s["value"] for s in record["settings"]}
        config = record.get("config")
        measurements = record.get("measurements")
        waveform = record.get("waveform")
        return cls(
            record["name"],
            record["temperature"],
            settings,
            None if config is None else SimulationConfig.from_metadata(config),
            None
            if measurements is None
            else tuple(Measurement.from_metadata(m) for m in measurements),
            Waveform.from_metadata(waveform) if waveform else waveform,
        )
