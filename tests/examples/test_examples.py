"""The example scripts run and print what they promise. They are the ones in the documentation."""

import os
import subprocess
import sys
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
SCRIPTS = sorted(p.name for p in EXAMPLES.glob("0*.py"))

EXPECTED = {
    "01_quickstart.py": ["R2:open: SUCCESS, vout = 1.0000 V", "same results when run again: True"],
    "02_faults_and_coverage.py": ["structural coverage: 95%", "it is the same fault: True"],
    "03_campaign.py": ["'pending': 0", "identical to the uninterrupted one: True"],
    "04_dataset.py": [
        "problems found by verify(): []",
        "largest difference in the measurements: 0.0",
    ],
    "05_reliability.py": ["smallest deviation detected", "critical_tolerance"],
    "06_operating_conditions.py": ["detection probability of each fault under each condition"],
    "07_custom.py": ["resistor_grade", "spread of the DC gain"],
    "08_monte_carlo.py": [
        "problems found by verify(): []",
        "yield per specification and of all of them",
        "the failure probability is below 0.0030, not zero",
        "circuits for a yield near 0.95 within +-0.005: 7299",
    ],
}


def test_every_example_is_checked():
    assert SCRIPTS == sorted(EXPECTED)


def test_binder_notebooks_match_the_documentation_scripts():
    root = EXAMPLES.parent
    notebook_dir = root / "binder" / "notebooks"
    script_stems = {Path(name).stem for name in SCRIPTS}
    notebook_stems = {path.stem for path in notebook_dir.glob("0[1-9]_*.ipynb")}
    assert notebook_stems == script_stems
    for path in notebook_dir.glob("0[1-9]_*.ipynb"):
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        assert notebook.metadata["kernelspec"]["name"] == "python3"
    subprocess.run(
        [sys.executable, str(root / "scripts" / "sync_binder_examples.py"), "--check"],
        cwd=root,
        check=True,
    )


@pytest.mark.parametrize(
    "notebook_name",
    ["01_quickstart", "02_faults_and_coverage", "07_custom"],
)
def test_binder_example_notebook_executes(notebook_name):
    notebook_path = EXAMPLES.parent / "binder" / "notebooks" / f"{notebook_name}.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    NotebookClient(
        notebook,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(EXAMPLES.parent)}},
    ).execute()


@pytest.fixture(scope="module")
def output(tmp_path_factory):
    return tmp_path_factory.mktemp("examples")


@pytest.mark.ngspice
@pytest.mark.parametrize("script", SCRIPTS)
def test_example_runs(script, output):
    env = {**os.environ, "SPICEFAULT_EXAMPLES_OUTPUT": str(output)}
    done = subprocess.run(
        [sys.executable, str(EXAMPLES / script)], capture_output=True, text=True, env=env,
        timeout=600,
    )  # fmt: skip
    assert done.returncode == 0, done.stderr[-2000:]
    for text in EXPECTED[script]:
        assert text in done.stdout, f"{text!r} not in the output of {script}"
