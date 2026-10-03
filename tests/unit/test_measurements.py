import numpy as np
import pytest

from spicefault.measurements import interp_response, quantise, resample


def test_interp_response_of_a_single_pole():
    freq = np.logspace(0, 4, 401)
    h = 1.0 / (1.0 + 1j * freq / 100.0)
    assert interp_response(freq, h, freq[57]) == pytest.approx(h[57], rel=1e-12)
    # between grid points the pole response is recovered closely
    assert interp_response(freq, h, 123.4) == pytest.approx(1 / (1 + 1.234j), rel=1e-4)


def test_quantise_clips_and_rounds():
    v = np.array([-1.0, 0.0, 1.6501, 3.3, 5.0])
    q = quantise(v, 0.0, 3.3, 12)
    lsb = 3.3 / 4095
    assert q[0] == 0.0 and q[-1] == pytest.approx(3.3)
    assert np.allclose(q / lsb, np.round(q / lsb))
    assert np.abs(q[1:4] - v[1:4]).max() <= lsb / 2


def test_resample():
    time = np.array([0.0, 1.0, 3.0])
    assert np.array_equal(resample(np.array([0.5, 2.0]), time, time * 2), [1.0, 4.0])
