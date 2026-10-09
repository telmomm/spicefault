# Contributing

Thank you for considering a contribution. Bug reports, questions, documentation fixes
and code are all welcome.

Before proposing a feature, see [what belongs in the library](docs/scope.md) and what is
left to other tools.

## Reporting a problem

Open an [issue](https://github.com/telmomm/spicefault/issues) with:

- what you did, as a script or netlist small enough to run;
- what you expected and what happened, with the full error message;
- the versions of `spicefault`, Python and ngspice (`ngspice --version`), and your
  operating system.

A simulation that fails is recorded with a status and a message, not raised. If the
problem is a wrong status, include the message stored with the sample.

## Setting up

You need Python 3.10 or later and [ngspice](https://ngspice.sourceforge.io/) on the
`PATH` (`brew install ngspice` or `apt install ngspice`).

```bash
git clone https://github.com/telmomm/spicefault.git
cd spicefault
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,docs]"
```

## Before you open a pull request

```bash
ruff check .
pytest tests/unit tests/examples
```

- **Tests.** New behaviour comes with a test. A test should check a result that is
  known independently (a formula, a count, a circuit with a closed-form answer), not
  only that the code runs. Tests that need ngspice carry the `ngspice` marker and are
  skipped where it is missing.
- **Reproducibility.** Anything random draws from the stream of its sample
  (`spicefault.experiments.sample_stream`) and from nothing else. Results must not
  depend on the number of workers or on the order of execution.
- **Failures are data.** A simulation or a measurement that fails becomes a row with a
  status. It never stops a campaign and is never dropped silently.
- **Faults are primitives.** A fault changes the netlist only through `SetParameter`,
  `InsertSeries` and `InsertParallel`, and says what it models in its docstring.
- **Metrics are defined.** A new metric comes with its definition, its estimator and
  a test against a population with a known answer.
- **Style.** `ruff` decides; lines up to 100 characters. Docstrings say what a
  function is for and what it assumes.
- **Documentation.** If the public interface changes, update the page under `docs/`
  and add a line to `CHANGELOG.md` under *Unreleased*.

The other test folders are slower and not needed for most changes: `tests/validation`
and `tests/benchmarks` simulate small campaigns, and `tests/regression` compares the
library with the project it was extracted from and needs that project installed.

## Pull requests

Keep a pull request to one change. Say what it does and why, and how you checked it.
The checks of GitHub Actions must pass.

## Releases

For maintainers: see [docs/releasing.md](docs/releasing.md).

## Conduct

This project follows a [code of conduct](CODE_OF_CONDUCT.md).
