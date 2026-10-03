"""Experiment G: which faults, and which components, can be told apart?

    python -m validation.experiments.g_separability --circuit biquad

It reads the campaign of a study at its declared tolerances (the dataset of scale 1
of experiment E) and writes the ambiguity structure found in the measurements and,
if the study stores one, in the waveform. It then simulates the local sensitivity of
the measurements to each component and compares what that predicts (testability rank,
collinear groups) with what the campaign found.

The waveform is reduced to its first principal components, fitted on the healthy
samples; the instrument noise of one waveform sample is that of the pulse peak.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import validation
from spicefault import Dataset
from spicefault.reliability import (
    ReliabilityAnalysis,
    collinear_groups,
    local_sensitivity,
    normalised_sensitivity,
    testability_rank,
)

from .common import DATA, THRESHOLD, analysis_of, save
from .e_variability import folder

N_COMPONENTS = 10


def waveform_analysis(study, path: Path, seed: int = 0) -> ReliabilityAnalysis:
    dataset = Dataset(path)
    ok = dataset.ok
    samples = dataset.samples[ok].reset_index(drop=True)
    sigma = study.noise["pulse_peak"]
    rng = np.random.default_rng(seed)
    waves = np.asarray(dataset.waveforms[ok], dtype=float)
    waves = waves + sigma * rng.normal(size=waves.shape)
    healthy = (samples["fault_id"] == "healthy").to_numpy()
    mean = waves[healthy].mean(axis=0)
    _, _, axes = np.linalg.svd(waves[healthy] - mean, full_matrices=False)
    scores = (waves - mean) @ axes[:N_COMPONENTS].T
    names = [f"pc_{k + 1}" for k in range(scores.shape[1])]
    frame = pd.concat([samples[["fault_id"]], pd.DataFrame(scores, columns=names)], axis=1)
    return ReliabilityAnalysis(
        frame, names, ok=None, noise_floor=dict.fromkeys(names, sigma),
        faults=dataset.metadata["faults"],
    )  # fmt: skip


def structure(analysis: ReliabilityAnalysis, threshold: float) -> dict:
    found = analysis.ambiguity(threshold)
    groups = [g for g in found.groups if len(g) > 1]
    located = [c for c, others in found.confusable.items() if not others]
    partners = [len(others) for others in found.confusable.values()]
    return {
        "n_features": len(analysis.features),
        "conditions_not_separated_from_healthy": found.undetectable,
        "n_ambiguity_groups": len(groups),
        "largest_ambiguity_group": max((len(g) for g in groups), default=1),
        "ambiguity_groups": groups,
        "confusable_components": found.confusable,
        "components_located_without_ambiguity": located,
        "mean_confusable_partners": float(np.mean(partners)) if partners else 0.0,
        "component_groups": found.component_groups,
    }


def analyse(circuit, data, threshold=THRESHOLD) -> dict:
    study = validation.get(circuit)
    path = folder(data, circuit, 1.0)
    analysis = analysis_of(study, path)
    result = {
        "experiment": "G, fault separability",
        "circuit": circuit,
        "threshold": threshold,
        "from_measurements": structure(analysis, threshold),
    }
    tables = {"separation_measurements": analysis.separation()}
    if study.waveform is not None:
        waves = waveform_analysis(study, path)
        result["from_waveform"] = structure(waves, threshold)
        result["from_waveform"]["method"] = (
            f"{N_COMPONENTS} principal components fitted on the healthy samples"
        )
        tables["separation_waveform"] = waves.separation()

    # what local sensitivity predicts, at the nominal circuit and without noise
    sensitivity = local_sensitivity(study.circuit, study.measurements, study.config)
    z = normalised_sensitivity(sensitivity, analysis.response("healthy").spread())
    groups, insensitive = collinear_groups(z)
    result["predicted_by_local_sensitivity"] = {
        "n_components": z.shape[1],
        "n_measurements": z.shape[0],
        "testability_rank": testability_rank(z),
        "collinear_groups": [g for g in groups if len(g) > 1],
        "insensitive_components": insensitive,
    }
    tables["sensitivity"] = sensitivity
    path = save("g_separability", circuit, result, tables)

    measured = result["from_measurements"]
    predicted = result["predicted_by_local_sensitivity"]
    print(study.description)
    print("  not separated from healthy:", len(measured["conditions_not_separated_from_healthy"]))
    print("  located without ambiguity:", measured["components_located_without_ambiguity"])
    print("  component groups:", measured["component_groups"])
    print(
        f"  testability rank {predicted['testability_rank']} for {predicted['n_components']} "
        f"components; collinear groups {predicted['collinear_groups']}"
    )
    if "from_waveform" in result:
        print("  from the waveform, component groups:", result["from_waveform"]["component_groups"])
    print("saved to", path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--circuit", default="biquad", choices=sorted(validation.STUDIES))
    parser.add_argument("--data", type=Path, default=DATA)
    args = parser.parse_args()
    analyse(args.circuit, args.data)


if __name__ == "__main__":
    main()
