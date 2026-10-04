"""The equivalence tests need `ecgfd` as it was before it adopted this library.

They compare the library with the simulation code it was extracted from. That code has
since moved on (its sampling of the electrodes changed, for a new dataset), so against
the current `ecgfd` they would fail for reasons that have nothing to do with the
library. They are collected only when the files they depend on are the pinned ones.
"""

import hashlib
import importlib.util
from pathlib import Path

import pytest

PINNED_COMMIT = "6a49f58"
PINNED_FILES = [
    "src/ecgfd/circuit.py",
    "src/ecgfd/faults.py",
    "src/ecgfd/sampling.py",
    "src/ecgfd/simulate.py",
    "src/ecgfd/specs.py",
    "src/ecgfd/spice.py",
    "src/ecgfd/dataset.py",
    "src/ecgfd/config.py",
    "configs/default.yaml",
]
PINNED_FINGERPRINT = "8627454ab93edf0aa5e80c2cadca8f14a4e2a65088c907fa1da4bb94ea9862f2"


def ecgfd_state() -> str:
    """`missing`, `pinned`, or `other` if `ecgfd` is installed in another version."""
    spec = importlib.util.find_spec("ecgfd")
    if spec is None or spec.origin is None:
        return "missing"
    root = Path(spec.origin).resolve().parents[2]
    digest = hashlib.sha256()
    for name in PINNED_FILES:
        file = root / name
        if not file.exists():
            return "other"
        digest.update(name.encode())
        digest.update(file.read_bytes())
    return "pinned" if digest.hexdigest() == PINNED_FINGERPRINT else "other"


STATE = ecgfd_state()

if STATE != "pinned":
    collect_ignore_glob = ["test_ecg_*.py"]
else:
    from ecgfd.circuit import CIRCUITS
    from ecgfd.config import DEFAULT_CONFIG, load_config

    @pytest.fixture(scope="session", params=sorted(CIRCUITS))
    def cfg(request) -> dict:
        """The default configuration of the ECG study, once per circuit."""
        return load_config(DEFAULT_CONFIG, request.param)
