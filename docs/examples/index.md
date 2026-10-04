# Examples

Runnable scripts, in the `examples/` folder of the repository. The pages below show
the same `.py` files that the tests run. Each also has a paired, executable notebook in
[`binder/notebooks/`](https://github.com/telmomm/spicefault/tree/main/binder/notebooks);
the documentation continues to show the scripts, while Binder opens the notebooks.

```bash
git clone https://github.com/telmomm/spicefault.git
cd spicefault
pip install -e .
python examples/01_quickstart.py
```

| Example | Script | Binder notebook | What it shows |
|---|---|---|---|
| [Quickstart](quickstart.md) | `01_quickstart.py` | [Open](https://mybinder.org/v2/gh/telmomm/spicefault/main?labpath=binder%2Fnotebooks%2F01_quickstart.ipynb) | A circuit from a netlist, one fault injected and simulated, then an experiment of fifteen simulations with tolerances and a seed. |
| [Faults and coverage](faults.md) | `02_faults_and_coverage.py` | [Open](https://mybinder.org/v2/gh/telmomm/spicefault/main?labpath=binder%2Fnotebooks%2F02_faults_and_coverage.ipynb) | Fault types and their records, a fault universe generated from rules, its coverage matrix, the faults left out with their reasons, and the parametric faults that lie inside the tolerance band. Nothing is simulated. |
| [A fault campaign](campaign.md) | `03_campaign.py` | [Open](https://mybinder.org/v2/gh/telmomm/spicefault/main?labpath=binder%2Fnotebooks%2F03_campaign.ipynb) | 660 simulations written to a dataset folder; then the same campaign interrupted after two chunks and resumed, giving an identical dataset. |
| [The dataset](dataset.md) | `04_dataset.py` | [Open](https://mybinder.org/v2/gh/telmomm/spicefault/main?labpath=binder%2Fnotebooks%2F04_dataset.ipynb) | Integrity check, the provenance of one sample, the netlist that was simulated for it, samples simulated again and compared, and arrays for a classifier. |
| [Reliability analysis](reliability.md) | `05_reliability.py` | [Open](https://mybinder.org/v2/gh/telmomm/spicefault/main?labpath=binder%2Fnotebooks%2F05_reliability.ipynb) | Detection probability with its interval, the smallest detectable deviation, the faults that look alike, and detection against the tolerance scale. |
| [Operating conditions](conditions.md) | `06_operating_conditions.py` | [Open](https://mybinder.org/v2/gh/telmomm/spicefault/main?labpath=binder%2Fnotebooks%2F06_operating_conditions.ipynb) | The same drawn circuits under three operating conditions, and the detection of each fault under each. |
| [Custom variation and measurement](custom.md) | `07_custom.py` | [Open](https://mybinder.org/v2/gh/telmomm/spicefault/main?labpath=binder%2Fnotebooks%2F07_custom.ipynb) | Two resistors drawn together with a recorded label, and a measurement defined by a function. |

They share one study, defined in `examples/rc_study.py`: an RC low-pass filter with a
resistive load, 18 faults, 1 % resistors and 5 % capacitors.

```python
--8<-- "examples/rc_study.py"
```

Each takes from a second to about a minute. Datasets are written under
`examples/output/`.

A larger one, `examples/filter/reliability.py`, runs every metric on the same filter
at three tolerance scales, with 15,000 simulations.
