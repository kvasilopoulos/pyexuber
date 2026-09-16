"""Tests for exuber.multivariate. Several checks mirror docs/replication/
multivariate/*.py (same checks, re-implemented here rather than imported
-- those scripts live in the umbrella repo, one level up from pyexuber's
own repo boundary, so pyexuber's CI (which only checks out
`kvasilopoulos/pyexuber`) can't reach them; see those scripts' module
docstrings for full narrative, R reference numbers, and (for
radf_common_validation.py) why that boundary exists)."""

import numpy as np
import pytest

from exuber.cv import radf_mc_cv
from exuber.multivariate import (
    RadfCommonCv,
    RadfCommonResult,
    _coexplosive_select_lag,
    _coexplosive_stat,
    _pca,
    cobubble_test,
    radf_common,
    radf_common_cv,
)
from exuber.radf import psy_minw


def test_pca_formula_exact():
    """_pca() (radf_common()'s PCA step, factored out so it's testable
    without the compiled _core extension) against an independent
    eigendecomposition of the sample covariance matrix."""
    rng = np.random.default_rng(0)
    n, nc, r = 60, 5, 2
    x = np.cumsum(rng.normal(size=(n, nc)), axis=0)

    loadings, scores, explained = _pca(x, r)

    xc = x - x.mean(axis=0)
    cov = (xc.T @ xc) / (n - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    for j in range(r):
        sign = np.sign(np.dot(loadings[:, j], eigvecs[:, j])) or 1.0
        diff = float(np.max(np.abs(loadings[:, j] - sign * eigvecs[:, j])))
        assert diff < 1e-10, f"loadings column {j}: diff={diff:.2e}"

    assert np.allclose(explained, eigvals[:r] / np.sum(eigvals), atol=1e-10)
    assert np.allclose(scores, xc @ loadings, atol=1e-10)


def test_radf_common_rejects_single_series():
    x = np.cumsum(np.random.default_rng(0).normal(size=(50, 1)), axis=0)
    with pytest.raises(ValueError):
        radf_common(x)


def test_radf_common_shapes_and_loadings():
    rng = np.random.default_rng(2)
    n, nc = 80, 4
    panel = np.cumsum(rng.normal(size=(n, nc)), axis=0)
    minw = psy_minw(n)

    res = radf_common(panel, minw=minw, r=2)

    assert isinstance(res, RadfCommonResult)
    assert res.adf.shape == (1,)
    assert res.bsadf.shape == (n - minw, 1)
    assert res.loadings.shape == (nc, 2)
    assert res.explained_variance_ratio.shape == (2,)
    # Columns of loadings are orthonormal (PCA property, independent of radf()).
    np.testing.assert_allclose(res.loadings.T @ res.loadings, np.eye(2), atol=1e-10)


def test_radf_common_cv_shape_and_n_field():
    cv = radf_common_cv(n=50, N=3, nrep=40, seed=5)

    assert isinstance(cv, RadfCommonCv)
    assert cv.N == 3
    assert cv.method == "Monte Carlo"
    assert cv.gsadf_cv.shape == (3,)
    n_minw = 50 - cv.minw
    assert cv.bsadf_cv.shape == (n_minw, 3)
    for arr in (cv.adf_cv, cv.sadf_cv, cv.gsadf_cv):
        assert np.all(np.diff(arr) >= 0)


def test_radf_common_cv_grows_with_n():
    """The independent-validation finding docs/multivariate.md documents:
    Theorem 4.3's claim that the null is asymptotically independent of
    panel width N does NOT hold at practical N -- the true null quantile
    *grows* with N. R reference numbers (n=100, nrep=300, seed=42):
    N=4 95% gsadf_cv=2.39, N=20 95% gsadf_cv=3.48 (see
    docs/replication/multivariate/radf_common_validation.py)."""
    cv_small = radf_common_cv(n=60, N=4, nrep=150, seed=1)
    cv_large = radf_common_cv(n=60, N=20, nrep=150, seed=1)
    assert cv_large.gsadf_cv[1] > cv_small.gsadf_cv[1]


def test_radf_common_cv_not_same_as_plain_mc_cv():
    """The whole point of radf_common_cv(): it must NOT collapse to
    radf_mc_cv() -- see docs/multivariate.md's independent-validation
    finding this file quotes above."""
    n, minw = 80, psy_minw(80)
    common_cv = radf_common_cv(n=n, N=15, minw=minw, nrep=80, seed=7)
    plain_cv = radf_mc_cv(n=n, minw=minw, nrep=80, seed=7)

    assert common_cv.gsadf_cv[1] > plain_cv.gsadf_cv[1]


# -- cobubble_test() ---------------------------------------------------------


def test_coexplosive_stat_formula_exact():
    """_coexplosive_stat() vs. an independent brute-force computation
    (separate lstsq call, manual loop-based cumulative sum) -- mirrors
    docs/replication/multivariate/radf_cobubble_validation.py's check 1."""
    rng = np.random.default_rng(1)
    tn = 100
    x = rng.normal(size=tn)
    y = 2 + 0.5 * x + rng.normal(size=tn)
    lag = 2

    S, _resid, _sigma2, _n = _coexplosive_stat(y, x, lag)

    lo, hi = max(lag, 0), tn + min(lag, 0)
    yy, xx = y[lo:hi], x[lo - lag : hi - lag]
    design = np.column_stack([np.ones(len(yy)), xx])
    beta, *_ = np.linalg.lstsq(design, yy, rcond=None)
    e_brute = yy - design @ beta
    n_brute = len(e_brute)
    sigma2_brute = np.sum(e_brute**2) / n_brute
    running = 0.0
    s_manual_sum = 0.0
    for e in e_brute:
        running += e
        s_manual_sum += running**2
    s_brute = s_manual_sum / (sigma2_brute * n_brute**2)

    assert abs(S - s_brute) < 1e-8


def test_cobubble_test_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        cobubble_test(np.zeros(10), np.zeros(9))


def test_cobubble_test_lag_recovery():
    """True lag baked into the DGP is recovered by lag selection --
    smaller/faster version of the validation script's check 5."""
    true_lag = 3
    for seed in range(1, 6):
        rng = np.random.default_rng(seed)
        tn, te = 150, 90
        ex = np.cumsum(rng.normal(size=te))
        expl = ex[-1] * 1.06 ** np.arange(1, tn - te + 1) + np.cumsum(
            rng.normal(scale=0.3, size=tn - te)
        )
        x = np.concatenate([ex, expl])
        y = np.full(tn, np.nan)
        for t in range(true_lag, tn):
            y[t] = 1 + 0.8 * x[t - true_lag] + rng.normal(scale=0.5)
        y[:true_lag] = x[:true_lag] + rng.normal(scale=0.5, size=true_lag)
        assert _coexplosive_select_lag(y, x, range(-6, 7)) == true_lag


def test_cobubble_test_rejects_independent_bubbles():
    """Power check: y and x have independent explosive episodes (not
    co-explosive) -- should reject co-explosivity."""
    rng = np.random.default_rng(42)
    tn, te = 150, 90
    ex = np.cumsum(rng.normal(size=te))
    expl_x = ex[-1] * 1.05 ** np.arange(1, tn - te + 1) + np.cumsum(
        rng.normal(scale=0.3, size=tn - te)
    )
    x = np.concatenate([ex, expl_x])
    ey = np.cumsum(rng.normal(size=te))
    expl_y = ey[-1] * 1.05 ** np.arange(1, tn - te + 1) + np.cumsum(
        rng.normal(scale=0.3, size=tn - te)
    )
    y = np.concatenate([ey, expl_y])

    out = cobubble_test(y, x, lag=0, nboot=199, seed=1)
    assert out.reject


def test_cobubble_test_not_reject_co_explosive_pair():
    """A genuinely co-explosive pair (y driven by x contemporaneously plus
    iid noise): co-explosivity should typically not be rejected."""
    rng = np.random.default_rng(7)
    tn, te = 150, 90
    ex = np.cumsum(rng.normal(size=te))
    expl = ex[-1] * 1.05 ** np.arange(1, tn - te + 1) + np.cumsum(
        rng.normal(scale=0.3, size=tn - te)
    )
    x = np.concatenate([ex, expl])
    y = 1 + 0.8 * x + rng.normal(size=tn)

    out = cobubble_test(y, x, lag=0, nboot=199, seed=1)
    assert not out.reject
