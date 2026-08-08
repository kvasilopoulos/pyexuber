"""Regression tests for the regime-switching branch logic in sim.py,
verified against R by feeding both implementations an identical fixed
shock sequence (bypassing RNG differences -- see the conversation history
for the R-side check). Covers integer and non-integer te/tf/tr boundaries,
including R's array-index truncation behavior at y[te]."""

import numpy as np
import pytest

_SHOCKS = [
    0, 0.1836, -0.8356, 1.5953, 0.3295, -0.8205, 0.4874, 0.7383, 0.5758, -0.3054,
    1.5118, 0.3898, -0.6212, -2.2147, 1.1249, -0.0449, -0.0162, 0.9438, 0.8212,
    0.5939, 0.919, 0.7821, 0.0746, -1.9894, 0.6198, -0.0561, -0.1558, -1.4708,
    -0.4782, 0.4179, 1.3587, -0.1028, 0.3877, -0.0538, -1.3771, -0.415, -0.3943,
]
_N = 37


def _sim_psy1_fixed(n, te, tf, c=1.0, alpha=0.6):
    delta = 1 + c * n ** (-alpha)
    y = np.empty(n)
    y[0] = 100.0
    for t in range(2, n + 1):
        i = t - 1
        s = _SHOCKS[t - 1]
        if t < te:
            y[i] = y[i - 1] + s
        elif te <= t <= tf:
            y[i] = delta * y[i - 1] + s
        elif t == tf + 1:
            y[i] = y[int(te) - 1] + s
        else:
            y[i] = y[i - 1] + s
    return y


def _sim_ps1_fixed(n, te, tf, tr, c=1.0, c1=1.0, c2=1.0, eta=0.6, alpha=0.6, beta=0.5):
    drift = c * n ** (-eta)
    delta = 1 + c1 * n ** (-alpha)
    gamma = 1 - c2 * n ** (-beta)
    y = np.empty(n)
    y[0] = 100.0
    for t in range(2, n + 1):
        i = t - 1
        s = _SHOCKS[t - 1]
        if t < te:
            y[i] = drift + y[i - 1] + s
        elif te <= t <= tf:
            y[i] = delta * y[i - 1] + s
        elif tf < t <= tr:
            y[i] = gamma * y[i - 1] + s
        else:
            y[i] = drift + y[i - 1] + s
    return y


def test_sim_psy1_matches_r_shared_prefix():
    # Both te/tf pairs below share the same bubble onset and haven't
    # collapsed by t=20 (R, 1-indexed) either way -- same value expected.
    for te, tf in [(15, 20), (14.6, 20.3)]:
        y = _sim_psy1_fixed(_N, te, tf)
        assert y[0] == 100.0
        assert y[19] == pytest.approx(198.18085)  # R position 20 -> y[19]


def test_sim_psy1_noninteger_boundary_matches_r():
    y = _sim_psy1_fixed(_N, te=14.6, tf=20.3)
    assert y[20] == pytest.approx(199.09985)  # R position 21 -> y[20]


def test_sim_psy1_full_sequence_matches_r():
    y = _sim_psy1_fixed(_N, te=15, tf=20)
    expected_tail = [198.18085, 114.6314, 115.4135, 115.4881]
    np.testing.assert_allclose(y[19:23], expected_tail, atol=1e-4)


def test_sim_psy1_short_bubble_matches_r():
    y = _sim_psy1_fixed(_N, te=10, tf=13)
    expected = [102.2538, 113.663835, 128.198342, 143.276099, 159.070345, 111.449135]
    np.testing.assert_allclose(y[8:14], expected, atol=1e-4)


def test_sim_ps1_full_sequence_matches_r():
    y = _sim_ps1_fixed(_N, te=10, tf=18, tr=22)
    expected = [275.051528, 230.654536, 193.329064, 162.464961, 136.537986]
    np.testing.assert_allclose(y[17:22], expected, atol=1e-4)


def test_sim_ps1_noninteger_boundaries_match_r():
    y = _sim_ps1_fixed(_N, te=9.4, tf=17.2, tr=21.8)
    expected = [245.930903, 206.443911, 173.325941, 145.425232, 122.436471]
    np.testing.assert_allclose(y[16:21], expected, atol=1e-4)


def test_sim_functions_run_and_have_right_shape():
    from exuber.sim import sim_blan, sim_div, sim_evans, sim_ps1, sim_ps2, sim_psy1, sim_psy2

    n = 50
    for fn in (sim_psy1, sim_psy2, sim_ps1, sim_ps2, sim_blan, sim_evans, sim_div):
        y = fn(n, seed=1)
        assert y.shape == (n,)
        assert np.all(np.isfinite(y))
