import importlib.util

import pytest

from spicefault.simulation import ngspice_path


def pytest_collection_modifyitems(config, items):
    missing = {}
    if ngspice_path() is None:
        missing["ngspice"] = "ngspice not installed"
    if importlib.util.find_spec("ecgfd") is None:
        missing["ecgfd"] = "ecgfd (the ECG baseline) not installed"
    for item in items:
        for marker, reason in missing.items():
            if marker in item.keywords:
                item.add_marker(pytest.mark.skip(reason=reason))
