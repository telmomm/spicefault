"""Dataset on disk: one of the representations of the results of a campaign."""

from .dataset import Dataset, Provenance
from .manifest import MANIFEST, Manifest, file_record, git_source
from .specification import Specification
from .splits import split_by_magnitude, split_by_replica
from .store import METADATA, SAMPLES, WAVEFORMS, assemble, load_dataset, load_metadata

__all__ = [
    "MANIFEST",
    "METADATA",
    "SAMPLES",
    "WAVEFORMS",
    "Dataset",
    "Manifest",
    "Provenance",
    "Specification",
    "assemble",
    "file_record",
    "git_source",
    "load_dataset",
    "load_metadata",
    "split_by_magnitude",
    "split_by_replica",
]
