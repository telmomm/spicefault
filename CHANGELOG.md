# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html). Until version 1.0 the
public interface may change between minor versions.

## [Unreleased]

### Added

- `Campaign`: the neutral name of a campaign, with `samples=`. Without faults it is a
  Monte Carlo study of the healthy population. `FaultCampaign` and `samples_per_fault`
  remain. Guide and example *Monte Carlo and yield*, with its Binder notebook.
- `Dataset.statistics` and `spicefault.statistics.describe`: mean, spread and quantiles
  of each measurement, with a distribution-free interval for the quantiles; a quantile
  the sample cannot support is reported as missing. `ecdf` and `quantile_interval`.
- `Dataset.yield_report` and `spicefault.statistics.yield_report`: yield per
  specification and overall with binomial intervals, bounds for the failed simulations
  and the margin to each limit. `samples_for_half_width` and `zero_failure_bound`.
- Page *Statistics of a population*, with the definitions and estimators, and page
  *Validation of the statistics*: the estimates against two circuits whose distribution
  has a closed form (`validation/statistical.py`).
- Page *Scope of the library*: the criterion for what belongs in it, and recipes for what
  is left to pandas, SciPy, SALib and scikit-learn.

### Changed

- `waveforms.npy` holds only the rows of the samples whose operating condition stores a
  waveform, and the manifest names those conditions under `waveform_conditions`.
  `Dataset.waveforms` reads as before, aligned with the samples and with `NaN` where
  nothing is stored, through `StoredWaveforms`, which also gives the array of the file
  (`stored`) and the sample of each of its rows (`rows`). Datasets in which every
  condition stores the waveform, and those written by earlier versions, are unchanged.

## [0.3.0] - 2026-10-09

### Added

- `Measurement.group`: several named values from one function, called once per
  simulation, one column per name.
- `waveform` on `OperatingCondition`: the waveform is stored only in the conditions that
  declare one, or in all but those that decline it with `waveform=False`; the rows of
  the others hold `NaN`. `Experiment.waveform_for` and `Experiment.waveform_points`.
- `Specification` and `Dataset.label`: pass/fail labels (`ok_<name>`, `compliant`,
  `violated`) from limits that are kept in the manifest and can be changed without
  simulating again. `Dataset.specifications` reads them back.
- `Dataset.cases`: one row per drawn circuit across its operating conditions, with the
  waveform of the condition chosen. `Dataset.to_ml` and `Dataset.analysis` take
  `by_case=True`.
- `Experiment.nominal`: the nominal circuit, simulated and measured under each
  operating condition, optionally with one fault.
- Fault rules: `tags=` on every rule (a dictionary or a function of the component),
  `models=` to select instances by subcircuit or device model, and `factors=`,
  `fault_values=` and `fault_type=` in `parametric_rule`. `Component.model`,
  `Netlist.model` and `Fault.with_tags`.
- `instance_tolerances`: the same tolerances, relative or absolute, for the parameters of
  every instance of a subcircuit. `CatalogueVariation`: a part whose type is drawn
  first and its parameters around the medians of that type, with the type as a label.
- The manifest records under `source` the git commit of the project that ran the
  campaign and whether its tracked files had uncommitted changes (`git_source`).

### Changed

- `Dataset.to_ml(waveforms=True)` returns only the samples of the conditions that store
  a waveform.
- `parametric_rule` names its rule after `fault_type`, and its `deviations` are optional.
- The Binder examples are split into guided notebook steps, stored with their outputs.

### Fixed

- `Dataset.verify()` no longer reports a problem when operating conditions have their
  own measurements: it expects in each row only the measurements of its condition, and
  reports a value in a column the condition does not declare.
- `Dataset.netlist()` works for a condition with custom measurements: it no longer
  rebuilds them to write the netlist.
- ngspice is found in the common installation folders when it is not on `PATH`.
- The banner of the README is shown on PyPI.

## [0.2.0] - 2026-10-09

### Added

- `Measurement.custom_result`: a measurement computed from the complete result, across
  several plots. Outputs can be selected per analysis, and
  `SimulationConfig.expected_plots` validates the plots a simulation returns and their
  order, also when the netlist owns its `.control` block. Noise analyses select the
  spectrum or the integrated-noise plot.
- An `OperatingCondition` can override the simulation `config` and the `measurements`;
  measurements not declared for a condition are stored as `NaN`.
- `FaultCampaign.from_experiment`. `tag_columns` on `FaultCampaign` turns fault tags into
  label columns, and `metadata` is recorded under `user` in the manifest.
- `SeriesResistanceFault`, a graded series resistance, with `series_rule` for the
  `FaultUniverse` and `FaultSeverity.log_series_resistance`.
- Source parameters in `Netlist.set_parameter` and in operating conditions: `ac`, and
  the arguments of source functions as `pulse.*` and `sin.*`.
- `Dataset.update_columns`: derived columns, with the sample fingerprint updated and the
  operation appended to the manifest history.
- `split_by_replica` and `split_by_magnitude`: train/test splits without leakage between
  the conditions of one replica.
- Netlists exported from KiCad and LTspice: `.param` constants, braced values, relative
  `.include` and `.lib` files with their fingerprints, UTF-8, UTF-16 and Latin-1 files,
  and dot analyses translated into the default configuration. `Circuit.check()` reports
  what cannot be addressed and `Circuit.imported_analyses()` lists the imported analyses.
- Binder notebooks for the examples, paired with the `.py` sources through Jupytext.
- Guide *From a schematic*, and the spicefault brand in the README and the documentation.

## [0.1.0] - 2026-10-04

First version.

### Added

- `Circuit`: a SPICE netlist seen as components, nodes and parameters.
- Fault types `OpenCircuit`, `ShortCircuit`, `LeakageFault`, `ParametricFault` and
  `CompositeFault`, each a list of primitive netlist changes with a record;
  `FaultSeverity`, `FaultSet`, and `FaultUniverse` with its coverage matrix.
- Variations of the healthy population (tolerance, normal, log-normal, uniform,
  log-uniform, fixed, custom, joint) and `VariationSet`.
- `OperatingCondition`, applied after the fault.
- `Simulator` with an ngspice backend, and `SimulationResult` with an explicit status.
- `Experiment` (in memory) and `FaultCampaign` (to disk, in chunks, resumable, one
  process per folder), with three seeding schemes.
- `Measurement` and `Waveform`.
- `Dataset`, `Manifest` and `Provenance`: integrity, traceability and reproduction of
  a campaign from its folder.
- `ReliabilityAnalysis`: detection probability, standardised shift, AUC, failure
  probability, diagnostic coverage, severity response, robustness against tolerance,
  dependence on operating conditions, separability and ambiguity.

[Unreleased]: https://github.com/telmomm/spicefault/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/telmomm/spicefault/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/telmomm/spicefault/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/telmomm/spicefault/releases/tag/v0.1.0
