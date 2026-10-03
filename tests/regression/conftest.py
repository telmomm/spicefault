import importlib.util

import pytest

if importlib.util.find_spec("ecgfd") is None:
    # without the ECG baseline there is nothing to compare with
    collect_ignore_glob = ["test_*.py"]
else:
    from ecgfd.circuit import CIRCUITS
    from ecgfd.config import DEFAULT_CONFIG, load_config

    @pytest.fixture(scope="session", params=sorted(CIRCUITS))
    def cfg(request) -> dict:
        """The default configuration of the ECG study, once per circuit."""
        return load_config(DEFAULT_CONFIG, request.param)
