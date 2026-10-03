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
    ambiguity_groups,
    collinear_groups,
    component_groups,
    confusable_components,
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
    """The ambiguity found, over all the fault conditions and over the detectable ones.

    A fault that is not separated from healthy is close to every other small fault, so
    over all the conditions each component ends up confusable with almost every other
    one. The question of interest is the second: once a fault is visible, which
    component could it be?
    """
    found = analysis.ambiguity(threshold)
    d = analysis.separation()
    hidden = set(found.undetectable)
    component_of = {f: "+".join(r["components"]) for f, r in analysis.records.items()}
    visible = [f for f in analysis.fault_ids if f not in hidden]
    among = d.loc[visible, visible]
    owner = {f: component_of[f] for f in visible}
    partners = confusable_components(among, owner, threshold)
    groups = [g for g in ambiguity_groups(among, threshold) if len(g) > 1]
    every = sorted(set(component_of.values()))
    return {
        "n_features": len(analysis.features),
        "n_conditions": len(analysis.fault_ids),
        "conditions_not_separated_from_healthy": found.undetectable,
        "components_with_no_detectable_fault": [c for c in every if c not in partners],
        "among_detectable_conditions": {
            "n_conditions": len(visible),
            "n_ambiguity_groups": len(groups),
            "largest_ambiguity_group": max((len(g) for g in groups), default=1),
            "ambiguity_groups": groups,
            "confusable_components": partners,
            "components_located_without_ambiguity": [c for c, o in partners.items() if not o],
            "mean_confusable_partners": float(np.mean([len(o) for o in partners.values()]))
            if partners
            else 0.0,
            "component_groups": component_groups(among, owner, threshold),
        },
        "over_all_conditions": {
            "mean_confusable_partners": float(np.mean([len(o) for o in found.confusable.values()])),
            "component_groups": found.component_groups,
        },
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
    visible = measured["among_detectable_conditions"]
    predicted = result["predicted_by_local_sensitivity"]
    print(study.description)
    print(
        f"  {len(measured['conditions_not_separated_from_healthy'])} of "
        f"{measured['n_conditions']} fault conditions are not separated from healthy"
    )
    print("  no detectable fault at all:", measured["components_with_no_detectable_fault"])
    print("  among the detectable ones:")
    print("    located without ambiguity:", visible["components_located_without_ambiguity"])
    print("    component groups:", visible["component_groups"])
    print(f"    mean confusable partners: {visible['mean_confusable_partners']:.1f}")
    print(
        f"  local sensitivity: rank {predicted['testability_rank']} for "
        f"{predicted['n_components']} components; collinear {predicted['collinear_groups']}; "
        f"insensitive {predicted['insensitive_components']}"
    )
    if "from_waveform" in result:
        waves = result["from_waveform"]["among_detectable_conditions"]
        print("  from the waveform, among the detectable ones:")
        print("    located without ambiguity:", waves["components_located_without_ambiguity"])
        print("    component groups:", waves["component_groups"])
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
