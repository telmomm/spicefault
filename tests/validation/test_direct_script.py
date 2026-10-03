"""The direct ngspice script of the comparison does the same work as the framework."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repository root

import validation  # noqa: E402
from validation.direct import sallen_key_direct as direct  # noqa: E402

pytestmark = pytest.mark.ngspice


def test_script_does_not_use_the_framework():
    source = Path(direct.__file__).read_text()
    assert "import spicefault" not in source and "from spicefault" not in source


def test_same_circuits_give_the_same_measurements():
    """Without tolerance both simulate the nominal circuit with each fault."""
    study = validation.get("sallen_key")
    assert study.features == direct.FEATURES
    assert len(study.universe()) == len(direct.fault_list()) == 58
    experiment = study.experiment(samples_per_fault=1, healthy_samples=1, tolerance_scale=0.0)
    ours = experiment.run(workers=1).to_frame()
    results = [direct.simulate(task) for task in direct.tasks(1, 1, 0, 0.0)]
    theirs = pd.DataFrame([row for row, _ in results])
    assert (ours["status"] == "SUCCESS").all() and theirs["ok"].all()
    # the two compute a band edge at the end of the sweep in different ways: last bits
    a, b = ours[direct.FEATURES].to_numpy(), theirs[direct.FEATURES].to_numpy()
    assert np.allclose(a, b, rtol=1e-12, atol=0.0)
    # the framework names the faults by content; the script by position
    assert list(ours["fault_id"])[:3] == ["healthy", "R1:open", "R1:short"]
    assert list(theirs["fault"])[:3] == ["healthy", "R1:open:+0", "R1:short:+0"]


def test_script_runs_a_small_campaign(tmp_path):
    seconds = direct.run(tmp_path / "direct", healthy=6, per_fault=1, workers=2)
    samples = pd.read_csv(tmp_path / "direct" / "samples.csv")
    waveforms = np.load(tmp_path / "direct" / "waveforms.npy")
    assert seconds > 0 and len(samples) == len(waveforms) == 64 and samples["ok"].all()
    assert waveforms.shape[1] == 150 and not np.isnan(waveforms).any()
    assert samples.loc[samples["fault"] == "healthy", "p_R1"].between(4921, 5439).all()
