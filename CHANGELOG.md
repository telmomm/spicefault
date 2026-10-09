# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html). Until version 1.0 the
public interface may change between minor versions.

## [Unreleased]

## [0.2.0] - 2026-10-09

### Added

- `Measurement.custom_result`: a measurement computed from the complete result, across
  several plots. Outputs can be selected per analysis, and
  `SimulationConfig.expected_plots` validates the plots a simulation returns and their
  order, also when the netlist owns its `.control` block. Noise analyses select the
  spectrum or the integrated-noise plot.
- An `OperatingCondition` can override the simulation `config` and the `measurements`;
  measurements not declared for a condition are stored as `NaN`.
- `FaultCampaign.from_experiment`, and user metadata labels recorded in the manifest.
- `SeriesResistanceFault`, a graded series resistance, with `series_rule` for the
  `FaultUniverse` and `FaultSeverity.log_series_resistance`.
- Source functions (`PULSE`, `SIN`, ...) in netlists can be addressed as parameters.
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

### Fixed

- ngspice is found in the common installation folders when it is not on `PATH`.

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

[Unreleased]: https://github.com/telmomm/spicefault/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/telmomm/spicefault/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/telmomm/spicefault/releases/tag/v0.1.0
