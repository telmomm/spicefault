# %% [markdown]
# # Quickstart
#
# A circuit, one fault, one simulation; then a small experiment.
# Run the same source with `python examples/01_quickstart.py` or open its Binder notebook.
# %%

from spicefault import Circuit, Experiment, Measurement, SimulationConfig, Simulator
from spicefault.faults import OpenCircuit, ParametricFault
from spicefault.variation import ToleranceVariation

# %% [markdown]
# ## A circuit is a SPICE netlist
# %%
circuit = Circuit(
    """resistive divider
V1 in 0 dc 1
R1 in out 10k
R2 out 0 10k
.end
""",
    name="divider",
)
print("components:", [c.name for c in circuit.components()])
print("parameters:", circuit.parameters())

# %% [markdown]
# ## A fault changes a copy of the netlist
# %%
fault = OpenCircuit("R2")
netlist = circuit.netlist()
fault.apply(netlist)
print("\nnetlist with", fault.fault_id, "injected:")
print(netlist)

# %% [markdown]
# ## One simulation, with an explicit status
# %%
config = SimulationConfig(analyses=("op",), outputs=("v(out)",))
output = Measurement.value("v(out)", name="vout")
simulator = Simulator("ngspice")
for label, text in (("healthy", circuit.to_netlist()), (fault.fault_id, str(netlist))):
    result = simulator.run(text, config)
    print(f"{label}: {result.status.value}, vout = {output(result):.4f} V")

# %% [markdown]
# ## An experiment: tolerances, faults, a seed
# %%
experiment = Experiment(
    circuit,
    config=config,
    faults=[OpenCircuit("R2"), ParametricFault("R1", deviation=0.2)],
    variations=[ToleranceVariation("R1", 0.01), ToleranceVariation("R2", 0.01)],
    measurements=[output],
    samples=5,
    seed=42,
)
table = experiment.run().to_frame()
print("\n15 simulations: 5 healthy circuits and 5 of each fault")
print(table[["fault_id", "replica", "status", "p_R1_value", "p_R2_value", "vout"]].round(4))

# the same seed gives the same circuits, whatever the number of workers
again = experiment.run().to_frame()
print("\nsame results when run again:", table["vout"].equals(again["vout"]))
