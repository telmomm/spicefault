"""The manifest of a dataset: the record of the run that produced it."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

SCHEMA_VERSION = 1
MANIFEST = "manifest.json"


def file_record(path: Path) -> dict:
    """Size and SHA-256 of a file, read in blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def git_source(folder: str | Path | None = None) -> dict:
    """The commit of the git repository that holds `folder`, by default the working
    directory, and whether its tracked files had uncommitted changes.

    This is the version of the project that ran the campaign, not of the library.
    An empty record outside a repository, or where `git` is not available.
    """

    def git(*arguments: str) -> str:
        done = subprocess.run(
            ["git", *arguments], cwd=folder, capture_output=True, text=True, timeout=30, check=True
        )
        return done.stdout.strip()

    try:
        commit = git("rev-parse", "HEAD")
        return {"commit": commit, "dirty": bool(git("status", "--porcelain", "-uno"))}
    except (OSError, subprocess.SubprocessError):
        return {}


@dataclass
class Manifest:
    """How and with what a dataset was produced, and the fingerprint of its files.

    `summary` holds the counts the application adds (completed, failed, status
    counts); in the file they are written next to the other fields. `user` is what
    the application passed as metadata, and `source` the git commit of the project
    that ran the campaign (see `git_source`), absent if there was none. The manifest
    is written last: its presence marks a complete dataset.
    """

    created: str
    spicefault_version: str
    simulator: str
    python: str
    numpy: str
    pandas: str
    platform: str
    n_samples: int
    workers: int
    chunk: int
    resumed: bool
    elapsed_last_run_s: float
    elapsed_total_s: float
    files: dict[str, dict] = field(default_factory=dict)
    summary: dict = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION
    user: dict = field(default_factory=dict)
    source: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        record = asdict(self)
        if not record["source"]:
            del record["source"]
        return {**{k: v for k, v in record.items() if k != "summary"}, **record["summary"]}

    @classmethod
    def from_dict(cls, record: dict) -> Manifest:
        known = {f.name for f in fields(cls)} - {"summary"}
        if record.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported manifest schema version {record.get('schema_version')}")
        return cls(
            **{k: v for k, v in record.items() if k in known},
            summary={k: v for k, v in record.items() if k not in known},
        )

    def write(self, folder: str | Path) -> None:
        (Path(folder) / MANIFEST).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def read(cls, folder: str | Path) -> Manifest:
        return cls.from_dict(json.loads((Path(folder) / MANIFEST).read_text()))

    def verify(self, folder: str | Path) -> list[str]:
        """Files that are missing or no longer match their recorded fingerprint."""
        problems = []
        for name, expected in self.files.items():
            path = Path(folder) / name
            if not path.exists():
                problems.append(f"{name}: missing")
            elif file_record(path) != expected:
                problems.append(f"{name}: content differs from the manifest")
        return problems
