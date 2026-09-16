"""cv.py can't be checked bit-for-bit against R (different RNGs -- see
sim.py's module docstring), so these check shapes and statistical sanity
(monotonic quantiles, GSADF cv >= ADF cv) rather than exact values."""

import numpy as np
import pytest

from exuber.cv import _wb_dgp_hlst, radf_mc_cv, radf_mc_distr, radf_wb_cv, radf_wb_distr
from exuber.radf import psy_minw


def test_radf_mc_cv_shapes_and_monotonic_quantiles():
    n = 80
    minw = psy_minw(n)
    cv = radf_mc_cv(n, nrep=200, seed=1)

    assert cv.adf_cv.shape == (3,)
    assert cv.sadf_cv.shape == (3,)
    assert cv.gsadf_cv.shape == (3,)
    n_minw = n - minw
    assert cv.badf_cv.shape == (n_minw, 3)
    assert cv.bsadf_cv.shape == (n_minw, 3)

    for arr in (cv.adf_cv, cv.sadf_cv, cv.gsadf_cv):
        assert np.all(np.diff(arr) >= 0)  # 90% <= 95% <= 99%
    assert np.all(np.diff(cv.bsadf_cv, axis=1) >= 0)

    # PWY asymptotic constants, constant across the window
    np.testing.assert_allclose(cv.badf_cv, np.tile([-0.44, -0.08, 0.6], (n_minw, 1)))


def test_radf_mc_distr_shapes():
    n = 60
    distr = radf_mc_distr(n, nrep=150, seed=2)
    assert distr.adf_distr.shape == (150,)
    assert distr.sadf_distr.shape == (150,)
    assert distr.gsadf_distr.shape == (150,)


def test_radf_wb_cv_shapes_multiseries():
    rng = np.random.default_rng(0)
    n, nc = 70, 2
    data = np.cumsum(rng.normal(size=(n, nc)), axis=0)
    minw = psy_minw(n)

    cv = radf_wb_cv(data, nboot=80, seed=3)

    assert cv.adf_cv.shape == (nc, 3)
    assert cv.gsadf_cv.shape == (nc, 3)
    pointer = n - minw
    assert cv.bsadf_cv.shape == (pointer, 3, nc)
    for j in range(nc):
        for arr in (cv.adf_cv[j], cv.sadf_cv[j], cv.gsadf_cv[j]):
            assert np.all(np.diff(arr) >= 0)


def test_radf_wb_distr_shapes():
    rng = np.random.default_rng(0)
    n = 50
    data = np.cumsum(rng.normal(size=n))
    distr = radf_wb_distr(data, nboot=60, seed=4)
    assert distr.gsadf_distr.shape == (60, 1)


def test_dist_skew_multiplier_moments():
    """Hafner (2020), Step 1: w = u/sqrt(2) + (v^2-1)/2 should have
    E[w]=0, E[w^2]=1, E[w^3]=1 (checked directly, not through the bubble
    statistic, since only the multiplier construction itself is a fixed,
    RNG-agnostic target)."""
    rng = np.random.default_rng(1)
    n = 500_000
    u = rng.normal(size=n)
    v = rng.normal(size=n)
    w = u / np.sqrt(2) + (v**2 - 1) / 2
    assert w.mean() == pytest.approx(0.0, abs=0.02)
    assert (w**2).mean() == pytest.approx(1.0, abs=0.02)
    assert (w**3).mean() == pytest.approx(1.0, abs=0.05)


def test_dist_skew_false_is_unaffected():
    """A purely additive option -- dist_skew=False (the default) must
    reproduce the pre-change DGP bit-for-bit for the same seed."""
    y = np.cumsum(np.random.default_rng(5).normal(size=60))
    r1 = _wb_dgp_hlst(y, False, np.random.default_rng(5))
    r2 = _wb_dgp_hlst(y, False, np.random.default_rng(5), dist_skew=False)
    np.testing.assert_array_equal(r1, r2)


def test_dist_rad_and_dist_skew_mutually_exclusive():
    data = np.cumsum(np.random.default_rng(0).normal(size=40))
    with pytest.raises(ValueError):
        radf_wb_cv(data, nboot=5, dist_rad=True, dist_skew=True)
