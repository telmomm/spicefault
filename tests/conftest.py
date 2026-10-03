import pytest

from spicefault.simulation import ngspice_path


def pytest_collection_modifyitems(config, items):
    if ngspice_path() is not None:
        return
    skip = pytest.mark.skip(reason="ngspice not installed")
    for item in items:
        if "ngspice" in item.keywords:
            item.add_marker(skip)
