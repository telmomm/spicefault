"""Conditions under which a circuit operates or is tested."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..netlist import Netlist


@dataclass(frozen=True)
class OperatingCondition:
    """A named set of deterministic settings, applied after the fault.

    `settings` maps (component, parameter) to an absolute value, for example
    `{("Vcc", "dc"): 3.0}` for a supply or `{("Rload", "value"): 1e3}` for a load.
    `temperature` is the simulation temperature in degrees Celsius; it only has an
    effect on devices whose model depends on temperature.
    """

    name: str = "nominal"
    temperature: float | None = None
    settings: dict[tuple[str, str], float] = field(default_factory=dict)

    def apply(self, netlist: Netlist) -> None:
        for (component, parameter), value in self.settings.items():
            netlist.set_parameter(component, parameter, "absolute", value)
        if self.temperature is not None:
            netlist.add_directive(f".options temp={float(self.temperature)}")

    def metadata(self) -> dict:
        return {
            "name": self.name,
            "temperature": self.temperature,
            "settings": [
                {"component": c, "parameter": p, "value": v} for (c, p), v in self.settings.items()
            ],
        }

    @classmethod
    def from_metadata(cls, record: dict) -> OperatingCondition:
        settings = {(s["component"], s["parameter"]): s["value"] for s in record["settings"]}
        return cls(record["name"], record["temperature"], settings)
