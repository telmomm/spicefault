"""The example scripts run and print what they promise. They are the ones in the documentation."""

import os
import subprocess
import sys
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
SCRIPTS = sorted(p.name for p in EXAMPLES.glob("[0-9][0-9]_*.py"))

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
    "09_test_design.py": [
        "accessible nodes: ['a', 'b', 'n', 'out']",
        "selected tests (2 of 52): ['step|a|peak', 'ac|a|phase@10000']",
        "faults the tests mostly miss: ['C2:parametric:+0.5']",
        "DC_w >= 0.9: certified after 50 faulty circuits",
    ],
    "10_minimum_cost_plan.py": [
        "optimal plan: ('ac|out|gain_db@60000', 'ac|out|phase@60000')",
        "0.40 EUR, 1.11 ms, DC_w 0.989",
        "DC_w >= 0.97: certified after 400 faulty circuits",
        "pool: rank 6, groups (('R4', 'R5'),)",
    ],
    "11_fmeda.py": [
        "per mode: SPFM >= 0.97: certified after 2675 faulty circuits, lower bound 0.9739",
        "the totals of the sheet are the claim: True",
        "per metric: SPFM >= 0.97: certified after 132 faulty circuits",
        "wrote sallen_key_fmeda.xlsx, ASIL reached by the sheet: C",
    ],
}


def test_every_example_is_checked():
    assert SCRIPTS == sorted(EXPECTED)


def test_binder_notebooks_match_the_documentation_scripts():
    root = EXAMPLES.parent
    notebook_dir = root / "binder" / "notebooks"
    script_stems = {Path(name).stem for name in SCRIPTS}
    # 00 is the guide to the notebooks, not an example
    notebooks = [p for p in notebook_dir.glob("[0-9][0-9]_*.ipynb") if not p.name.startswith("00_")]
    assert {path.stem for path in notebooks} == script_stems
    for path in notebooks:
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
