import json

import numpy as np
import pytest

from spicefault import SimulationConfig, SimulationStatus, Simulator
from spicefault.simulation import NgspiceBackend, Plot, build_deck, run_deck

DIVIDER = "divider\nV1 in 0 dc 1 ac 1\nR1 in out 10k\nR2 out 0 10k\nC1 out 0 100n\n.end\n"
OP_AC = SimulationConfig(analyses=("op", "ac dec 10 1 1e4"), outputs=("v(out)",))


def test_build_deck_adds_one_write_per_analysis():
    deck = build_deck(DIVIDER, OP_AC)
    assert deck == DIVIDER.replace(
        ".end\n",
        ".control\nset noaskquit\nset appendwrite\nop\nwrite out.raw v(out)\n"
        "ac dec 10 1 1e4\nwrite out.raw v(out)\n.endc\n.end\n",
    )
    assert build_deck(DIVIDER, SimulationConfig()) == DIVIDER
    with pytest.raises(ValueError, match="already has a .control block"):
        build_deck(deck, OP_AC)


def test_analysis_specific_outputs_and_noise_plot_selection():
    config = SimulationConfig(
        analyses=(
            ("op", ("v(out)", "v(ref)")),
            ("noise v(out) V1 dec 10 1 1k", ("onoise_spectrum",), "integrated"),
        ),
        expected_plots=("Operating Point", "Integrated Noise"),
    )
    deck = build_deck(DIVIDER, config)
    assert "op\nwrite out.raw v(out) v(ref)\n" in deck
    assert "noise v(out) V1 dec 10 1 1k\nsetplot noise2\n" in deck
    assert "write out.raw onoise_spectrum\n" in deck
    record = json.loads(json.dumps(config.metadata()))
    assert SimulationConfig.from_metadata(record) == config


def test_expected_plots_check_custom_control_blocks():
    config = SimulationConfig(expected_plots=("AC Analysis",))
    assert NgspiceBackend._check([Plot("AC Analysis", {})], config) is None
    problem = NgspiceBackend._check([Plot("Operating Point", {})], config)
    assert problem == "expected plots ['AC Analysis'], got ['Operating Point']"


def test_unknown_backend():
    with pytest.raises(ValueError, match="unknown backend"):
        Simulator("xyce")


def test_custom_backend_is_used_as_given():
    class Fake:
        name = "fake"

        def version(self):
            return "1.0"

        def run(self, netlist, config):
            return (netlist, config)

    simulator = Simulator(Fake())
    assert simulator.run("deck") == ("deck", SimulationConfig())
    assert simulator.metadata() == {"simulator": "fake", "simulator_version": "1.0"}


@pytest.mark.ngspice
class TestNgspice:
    def test_success(self):
        simulator = Simulator("ngspice")
        result = simulator.run(DIVIDER, OP_AC)
        assert result.ok and result.status is SimulationStatus.SUCCESS and result.message == ""
        assert len(result.plots) == 2 and result.elapsed > 0
        assert result.plot(0)["v(out)"][0] == pytest.approx(0.5)
        assert result.plot("ac") is result.plots[1]
        assert result.plot("AC Analysis") is result.plots[1]
        with pytest.raises(KeyError):
            result.plot("tran")
        assert simulator.metadata()["simulator_version"].startswith("ngspice-")

    def test_same_output_as_the_plain_runner(self):
        ours = Simulator().run(DIVIDER, OP_AC)
        theirs = run_deck(build_deck(DIVIDER, OP_AC))
        for a, b in zip(ours.plots, theirs, strict=True):
            assert a.name == b.name
            assert all(np.array_equal(a[k], b[k]) for k in a.vectors)

    def test_deck_with_its_own_control_block(self):
        result = Simulator().run(build_deck(DIVIDER, OP_AC))
        assert result.ok and len(result.plots) == 2

    def test_timeout(self):
        config = SimulationConfig(analyses=("tran 10n 1 0 10n",), timeout=0.5)
        result = Simulator().run(DIVIDER, config)
        assert result.status is SimulationStatus.TIMEOUT and not result.ok and result.plots == ()
        assert "timed out" in result.message

    def test_convergence_error(self):
        # two ideal voltage sources in parallel: the matrix is singular
        loop = "loop\nV1 a 0 dc 1\nV2 a 0 dc 2\nR1 a 0 1k\n.end\n"
        result = Simulator().run(loop, SimulationConfig(analyses=("op",)))
        assert result.status is SimulationStatus.CONVERGENCE_ERROR
        assert "singular matrix" in result.message and result.log

    def test_failed_analysis_is_not_mistaken_for_a_result(self):
        """After a failed analysis ngspice saves the previous plot again."""
        config = SimulationConfig(analyses=("op", "ac dec 10 0 1e4"), outputs=("v(out)",))
        result = Simulator().run(DIVIDER, config)  # an AC sweep cannot start at 0 Hz
        assert result.status is SimulationStatus.INVALID_OUTPUT
        assert "expected plots" in result.message
        # the plain runner returns the repeated plot without complaint
        assert [p.name[:9] for p in run_deck(build_deck(DIVIDER, config))] == ["Operating"] * 2

    def test_no_output_is_a_failure(self):
        result = Simulator().run("bad\nR1 a 0 1k\n.control\nfoo\n.endc\n.end\n")
        assert result.status is SimulationStatus.FAILED and "no output" in result.message

    def test_missing_simulator(self, monkeypatch):
        monkeypatch.setattr("spicefault.simulation.ngspice.shutil.which", lambda name: None)
        result = NgspiceBackend().run(DIVIDER, OP_AC)
        assert result.status is SimulationStatus.FAILED and "not found" in result.message
