PY ?= .venv/bin/python

.PHONY: setup lint test test-all examples docs docs-serve build clean

setup:
	python3 -m venv .venv
	$(PY) -m pip install -e ".[dev,docs]"

lint:
	$(PY) -m ruff check .

# the library and its examples: about two minutes
test:
	$(PY) -m pytest tests/unit tests/examples

# everything, including the validation circuits and the benchmarks at tiny size
test-all:
	$(PY) -m pytest

examples:
	for f in examples/0*.py; do echo "== $$f"; $(PY) $$f || exit 1; done

docs:
	$(PY) -m mkdocs build --strict

docs-serve:
	$(PY) -m mkdocs serve

build:
	rm -rf dist
	$(PY) -m build
	$(PY) -m twine check dist/*

clean:
	rm -rf dist build site .pytest_cache .ruff_cache examples/output
