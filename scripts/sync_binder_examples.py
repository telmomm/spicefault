"""Synchronise documentation scripts with their Binder notebook pairs."""

from __future__ import annotations

import argparse
from pathlib import Path

import jupytext
import nbformat

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
NOTEBOOKS = ROOT / "binder" / "notebooks"
PAIR_FORMATS = "ipynb,../../examples//py:percent"
SCRIPTS = sorted(EXAMPLES.glob("0[1-7]_*.py"))


def _script_notebook(path: Path):
    notebook = jupytext.read(str(path), fmt="py:percent")
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata["language_info"] = {"name": "python"}
    notebook.metadata.setdefault("jupytext", {})["formats"] = PAIR_FORMATS
    return notebook


def _signature(notebook) -> list[tuple[str, str]]:
    return [(cell.cell_type, cell.source) for cell in notebook.cells]


def sync_from_scripts(check_only: bool = False) -> int:
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    missing = []
    stale = []
    for script in SCRIPTS:
        notebook_path = NOTEBOOKS / f"{script.stem}.ipynb"
        expected = _script_notebook(script)
        if not notebook_path.exists():
            missing.append(notebook_path.relative_to(ROOT))
            if not check_only:
                jupytext.write(expected, str(notebook_path), fmt="ipynb")
                print(f"created {notebook_path.relative_to(ROOT)}")
            continue
        current = nbformat.read(notebook_path, as_version=4)
        if _signature(expected) != _signature(current):
            stale.append(notebook_path.relative_to(ROOT))
            if not check_only:
                jupytext.write(expected, str(notebook_path), fmt="ipynb")
                print(f"updated {notebook_path.relative_to(ROOT)}")

    if check_only and (missing or stale):
        for path in missing:
            print(f"missing notebook: {path}")
        for path in stale:
            print(f"out-of-date notebook: {path}")
        return 1
    if check_only:
        print(f"{len(SCRIPTS)} Binder notebooks are synchronised")
    return 0


def sync_to_scripts() -> int:
    for script in SCRIPTS:
        notebook_path = NOTEBOOKS / f"{script.stem}.ipynb"
        if not notebook_path.exists():
            print(f"missing notebook: {notebook_path.relative_to(ROOT)}")
            return 1
        notebook = nbformat.read(notebook_path, as_version=4)
        jupytext.write(notebook, str(script), fmt="py:percent")
        print(f"updated {script.relative_to(ROOT)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check", action="store_true", help="report missing or stale notebook pairs"
    )
    mode.add_argument(
        "--to-scripts", action="store_true", help="write notebook cells back to scripts"
    )
    args = parser.parse_args()
    return sync_to_scripts() if args.to_scripts else sync_from_scripts(check_only=args.check)


if __name__ == "__main__":
    raise SystemExit(main())