"""The recipes that connect the library with SciPy and SALib run and find what they say."""

import subprocess
import sys
from pathlib import Path

import pytest

RECIPES = Path(__file__).resolve().parents[2] / "examples" / "recipes"
EXPECTED = {
    "worst_case.py": ("scipy", ["the search reached the worst corner: True"]),
    "sensitivity_indices.py": ("SALib", ["the capacitor explains most of the variance: True"]),
}


def test_every_recipe_is_checked():
    assert sorted(p.name for p in RECIPES.glob("*.py")) == sorted(EXPECTED)


@pytest.mark.ngspice
@pytest.mark.parametrize("script", sorted(EXPECTED))
def test_recipe_runs(script):
    package, texts = EXPECTED[script]
    pytest.importorskip(package)  # an optional extra: pip install "spicefault[recipes]"
    done = subprocess.run(
        [sys.executable, str(RECIPES / script)], capture_output=True, text=True, timeout=600
    )
    assert done.returncode == 0, done.stderr[-2000:]
    for text in texts:
        assert text in done.stdout, f"{text!r} not in the output of {script}"
