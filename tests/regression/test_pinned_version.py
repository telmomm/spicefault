"""Says, in the test report, why the equivalence tests ran or did not."""

import pytest

from conftest import PINNED_COMMIT, STATE


def test_equivalence_tests_have_their_baseline():
    if STATE == "missing":
        pytest.skip("ecgfd is not installed: the equivalence tests are not collected")
    if STATE == "other":
        pytest.skip(
            f"ecgfd is installed in a version other than commit {PINNED_COMMIT}, the one the "
            "library was extracted from: the equivalence tests are not collected. "
            "See tests/regression/README.md to run them against that commit."
        )
    assert STATE == "pinned"
