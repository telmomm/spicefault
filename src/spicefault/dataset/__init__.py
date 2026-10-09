"""Dataset on disk: one of the representations of the results of a campaign."""

from .dataset import Dataset, Provenance
from .manifest import MANIFEST, Manifest, file_record, git_source
from .specification import Specification
from .splits import split_by_magnitude, split_by_replica
from .store import (
    METADATA,
    SAMPLES,
    WAVEFORMS,
    StoredWaveforms,
    assemble,
    load_dataset,
    load_metadata,
    read_waveforms,
)

__all__ = [
    "MANIFEST",
    "METADATA",
    "SAMPLES",
    "WAVEFORMS",
    "Dataset",
    "Manifest",
    "Provenance",
    "Specification",
    "StoredWaveforms",
    "assemble",
    "file_record",
    "git_source",
    "load_dataset",
    "load_metadata",
    "read_waveforms",
    "split_by_magnitude",
    "split_by_replica",
]
