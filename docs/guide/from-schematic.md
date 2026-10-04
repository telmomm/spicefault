# From a schematic

Export a SPICE netlist from the schematic tool, then load it directly:

```python
from spicefault import Circuit, FaultCampaign, Measurement, SimulationConfig
from spicefault.faults import LeakageFault

circuit = Circuit.from_netlist("examples/from_schematic/ltspice_rc.cir")
print(circuit.check())

campaign = FaultCampaign(
    circuit,
    [LeakageFault("C1", 1e6)],
    out_dir="data/ltspice_rc",
    config=SimulationConfig(),  # imported .ac/.tran commands are retained
    measurements=[Measurement.magnitude("v(out)", 1e3, name="gain_1k")],
)
campaign.run()
```

## KiCad

In KiCad, export the schematic as a SPICE netlist (`.cir`/`.net`) and pass that file to
`Circuit.from_netlist`. The sample [KiCad export](../../examples/from_schematic/kicad_rc.cir)
shows a capacitor `Rser` parameter and a transient analysis directive.

## LTspice

In LTspice, use **File > Export netlist** and pass the resulting `.net` file. Relative
`.include` and file-based `.lib` paths are resolved against the exported file's folder;
their paths and SHA-256 fingerprints are recorded in experiment metadata. Keep those
model files available at the recorded paths when reproducing a dataset on another machine.
UTF-8, UTF-16 and Latin-1 exports are read, and the micro sign is normalized to `u`.

The [LTspice sample](../../examples/from_schematic/ltspice_rc.cir) covers a constant
`.param`, a braced component value, a source `PULSE`, a relative model include and an `.ac`
directive. Numeric `.param` constants are available to
`Circuit.parameters()` and can be replaced by variations or faults. Arbitrary expressions
remain untouched and are reported by `Circuit.check()` when their value cannot be read.
Vendor-specific capacitor `Rser` parameters are also reported by `Circuit.check()` because
ngspice may reject them unless a compatible dialect mode is selected.

## Check the import

`Circuit.check()` reports elements whose terminals or numeric values cannot be addressed,
components inside subcircuits (only top-level instances can be faulted), unresolved includes
and directives the library does not recognize. Dot analyses such as `.ac`, `.tran` and `.op`
are translated into the default `Experiment` configuration and validated by plot name and
order. `.backanno` is discarded. The tool does not translate vendor-specific dialects;
ngspice compatibility modes remain the simulator's responsibility.