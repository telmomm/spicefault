"""A backend without a simulator, for the campaign tests: a module so that workers import it."""

import numpy as np

from spicefault import SimulationResult, SimulationStatus
from spicefault.netlist import Netlist
from spicefault.simulation import Plot


class Divider:
    """Solves `V1 in 0`, `R1 in out`, `R2 out 0` by the divider formula.

    An open R2 does not converge, and with R2 above 10.04 kOhm the output is missing.
    """

    name = "divider"

    def version(self):
        return "1"

    def run(self, netlist, config):
        if "Rser_R2" in netlist:
            return SimulationResult(SimulationStatus.CONVERGENCE_ERROR, message="no convergence")
        net = Netlist(netlist)
        r1, r2, v = net.value("R1"), net.value("R2"), net.value("V1", "dc")
        missing = r2 > 10040
        if "Rpar_R2" in netlist:
            r2 = 1 / (1 / r2 + 1 / net.value("Rpar_R2"))
        out = v * r2 / (r1 + r2)
        time = np.linspace(0.0, 1.0, 11)
        vectors = {"v(in)": np.array([v])} if missing else {"v(out)": np.array([out])}
        plots = (
            Plot("Operating Point", vectors),
            Plot("Transient Analysis", {"time": time, "v(out)": out * time}),
        )
        return SimulationResult(SimulationStatus.SUCCESS, plots, elapsed=0.001)
