# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html). Until version 1.0 the
public interface may change between minor versions.

## [Unreleased]

## [0.1.0] - 2026-10-03

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

[Unreleased]: https://github.com/telmomm/spicefault/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/telmomm/spicefault/releases/tag/v0.1.0
