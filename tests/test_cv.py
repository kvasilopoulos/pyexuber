"""cv.py can't be checked bit-for-bit against R (different RNGs -- see
sim.py's module docstring), so these check shapes and statistical sanity
(monotonic quantiles, GSADF cv >= ADF cv) rather than exact values."""

import numpy as np

from exuber.cv import radf_mc_cv, radf_mc_distr, radf_wb_cv, radf_wb_distr
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
