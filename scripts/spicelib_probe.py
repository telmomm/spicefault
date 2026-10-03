"""What spicelib does on a fault-injection task, checked by running it.

    pip install spicelib          # in an environment of its own: it is GPL-3.0
    python scripts/spicelib_probe.py

It runs spicelib with ngspice on the Sallen-Key filter of `validation/` and prints the
facts that docs/COMPARISON.md relies on: how values are written, whether a fault can be
injected with its editor, how a failed analysis is counted, and whether its Monte Carlo
gives the same samples twice. `spicefault` does not import spicelib; this script is
the only place where it is used.
"""

from __future__ import annotations

import tempfile
from importlib import metadata
from pathlib import Path

from spicelib import SimRunner, SpiceEditor
from spicelib.sim.tookit.montecarlo import Montecarlo
from spicelib.simulators.ngspice_simulator import NGspiceSimulator

HERE = Path(__file__).resolve().parents[1]
CONTROL = ".control\nac dec 100 1k 1e6\nwrite\nquit\n.endc\n.end\n"


def netlist(folder: Path) -> Path:
    """The Sallen-Key netlist in the form spicelib accepts: a comment as first line, the
    library written in place, and a control block that writes and quits.
    """
    source = HERE / "validation" / "netlists"
    lines = (source / "sallen_key.cir").read_text().splitlines()
    lines[0] = "* " + lines[0]
    lines = [(source / "opamp.lib").read_text().rstrip() if x.startswith(".include") else x
             for x in lines if x.strip() != ".end"]  # fmt: skip
    path = folder / "sallen_key.net"
    path.write_text("\n".join(lines) + "\n" + CONTROL)
    return path


def line_of(path: Path, name: str) -> str:
    return next(line for line in path.read_text().splitlines() if line.startswith(name + " "))


def main() -> None:
    version = metadata.version("spicelib")
    print(f"spicelib {version}, ngspice at {NGspiceSimulator.spice_exe}")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        path = netlist(tmp)

        net = SpiceEditor(str(path))
        net["R1"].value = 5359.872341234567
        net.save_netlist(str(tmp / "number.net"))
        net["R1"].value = "5359.872341234567"
        net.save_netlist(str(tmp / "text.net"))
        print("value given as a number is written as:", line_of(tmp / "number.net", "R1"))
        print("value given as text is written as:    ", line_of(tmp / "text.net", "R1"))

        net.reset_netlist()
        net["R3"].ports = ["b", "R3_x"]  # an open: the terminal moved to a new node ...
        net.add_instruction("Rser_R3 R3_x 0 1e9")  # ... and a large resistance to the old one
        net.save_netlist(str(tmp / "open.net"))
        print("an open written with the editor:", line_of(tmp / "open.net", "R3"), "+ Rser_R3")

        bad = str(tmp / "bad")
        runner = SimRunner(simulator=NGspiceSimulator, output_folder=bad, verbose=False)
        net.reset_netlist()
        net.add_instruction("V2 in 0 5")  # a second ideal source across the input: no solution
        runner.run(net, run_filename="bad.net")
        runner.wait_completion()
        log = (tmp / "bad" / "bad.log").read_text()
        print(f"unsolvable circuit: counted ok {runner.okSim}, failed {runner.failSim};",
              "log says:", "singular" in log.lower() or "aborted" in log.lower())  # fmt: skip

        draws = []
        for tag in ("a", "b"):
            runner = SimRunner(simulator=NGspiceSimulator, output_folder=str(tmp / tag),
                               parallel_sims=4, verbose=False)  # fmt: skip
            analysis = Montecarlo(SpiceEditor(str(path)), runner)
            analysis.set_tolerance("R", 0.05)
            analysis.run_analysis(num_runs=6)
            files = sorted((tmp / tag).glob("*.net"))
            draws.append([line_of(f, "R1").split()[3] for f in files])
        print("Monte Carlo, R1 of six runs:", draws[0])
        print("the same again:             ", draws[1])
        print("same samples in both:", draws[0] == draws[1])

        try:
            analysis.prepare_testbench(num_runs=10)
            print("in-simulator Monte Carlo testbench: prepared")
        except NotImplementedError as error:
            print("in-simulator Monte Carlo testbench with ngspice:", error)


if __name__ == "__main__":
    main()
