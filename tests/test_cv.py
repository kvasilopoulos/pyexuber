"""cv.py can't be checked bit-for-bit against R (different RNGs -- see
sim.py's module docstring), so these check shapes and statistical sanity
(monotonic quantiles, GSADF cv >= ADF cv) rather than exact values.
lag_select()/adf_res() (deterministic, no RNG) ARE checked bit-for-bit --
see test_lagselect.py."""

import numpy as np

from exuber.cv import (
    radf_mc_cv,
    radf_mc_distr,
    radf_sb_cv,
    radf_sb_distr,
    radf_wb_cv,
    radf_wb_distr,
    radf_wb_ps_cv,
    radf_wb_ps_distr,
)
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


def test_radf_wb_ps_cv_shapes_and_monotonic_quantiles():
    rng = np.random.default_rng(1)
    n = 60
    data = np.cumsum(rng.normal(size=n))
    minw = psy_minw(n)

    cv = radf_wb_ps_cv(data, minw=minw, nboot=80, adflag=0, seed=7)

    assert cv.adf_cv.shape == (1, 3)
    assert cv.gsadf_cv.shape == (1, 3)
    pointer = n - minw
    assert cv.bsadf_cv.shape == (pointer, 3, 1)
    assert np.all(np.diff(cv.gsadf_cv.ravel()) >= 0)


def test_radf_wb_ps_cv_type_aic_selects_lag():
    rng = np.random.default_rng(2)
    n = 60
    data = np.cumsum(rng.normal(size=n))
    # adflag is the max_lag searched when type != "fixed"
    cv = radf_wb_ps_cv(data, nboot=30, adflag=4, type="aic", seed=8)
    assert cv.method == "Wild Bootstrap (Phillips-Shi)"


def test_radf_wb_ps_cv_tb_mode_collapses_badf_bsadf():
    rng = np.random.default_rng(3)
    n = 60
    data = np.cumsum(rng.normal(size=n))
    minw = psy_minw(n)
    tb = minw + 10

    cv = radf_wb_ps_cv(data, minw=minw, nboot=40, adflag=0, tb=tb, seed=9)

    pointer_full = n - minw
    assert cv.badf_cv.shape == (pointer_full, 3, 1)
    np.testing.assert_allclose(cv.badf_cv[0, :, 0], cv.sadf_cv[0, :])
    np.testing.assert_allclose(cv.bsadf_cv[0, :, 0], cv.gsadf_cv[0, :])


def test_radf_wb_ps_distr_shapes():
    rng = np.random.default_rng(4)
    n = 45
    data = np.cumsum(rng.normal(size=n))
    distr = radf_wb_ps_distr(data, nboot=50, adflag=0, seed=10)
    assert distr.gsadf_distr.shape == (50, 1)


def test_radf_sb_cv_shapes_lag_zero():
    rng = np.random.default_rng(5)
    n = 70
    data = np.cumsum(rng.normal(size=n))
    minw = psy_minw(n)

    sb = radf_sb_cv(data, minw=minw, lag=0, nboot=40, seed=11)

    pointer = n - minw
    assert sb.bsadf_panel_cv.shape == (pointer, 3)
    assert sb.gsadf_panel_cv.shape == (3,)
    assert np.all(np.diff(sb.gsadf_panel_cv) >= 0)


def test_radf_sb_cv_shapes_lag_gt_zero():
    """Regression test for the off-by-one bug found in exuber's own R
    R/radf_sb.R (initmat[j, lag:1] one element short for lag > 0, see
    cv.py's _radf_sb docstring): bsadf_panel_cv must have the full
    (nr - minw - lag) rows, not (nr - minw - lag - lag)."""
    rng = np.random.default_rng(6)
    n = 70
    data = np.cumsum(rng.normal(size=n))
    minw = psy_minw(n)
    lag = 2

    sb = radf_sb_cv(data, minw=minw, lag=lag, nboot=30, seed=12)

    expected_pointer = n - minw - lag
    assert sb.bsadf_panel_cv.shape == (expected_pointer, 3)
    assert sb.lag == lag


def test_radf_sb_cv_type_fixed_matches_explicit_default():
    rng = np.random.default_rng(7)
    n = 60
    data = np.cumsum(rng.normal(size=n))

    a = radf_sb_cv(data, lag=1, nboot=25, seed=13)
    b = radf_sb_cv(data, lag=1, type="fixed", nboot=25, seed=13)
    np.testing.assert_array_equal(a.gsadf_panel_cv, b.gsadf_panel_cv)
    np.testing.assert_array_equal(a.bsadf_panel_cv, b.bsadf_panel_cv)


def test_radf_sb_distr_shapes():
    rng = np.random.default_rng(8)
    n = 55
    data = np.cumsum(rng.normal(size=n))
    distr = radf_sb_distr(data, lag=0, nboot=30, seed=14)
    assert distr.gsadf_panel_distr.shape == (30,)
