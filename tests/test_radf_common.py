"""Tests for exuber.radf_common. Several checks mirror docs/replication/
multivariate/radf_common_validation.py (same checks, re-implemented here
rather than imported -- that script lives in the umbrella repo, one
level up from pyexuber's own repo boundary, so pyexuber's CI (which only
checks out `kvasilopoulos/pyexuber`) can't reach it; see that script's
module docstring for full narrative, R reference numbers, and why that
boundary exists)."""

import numpy as np
import pytest

from exuber.cv import radf_mc_cv
from exuber.radf import psy_minw
from exuber.radf_common import RadfCommonCv, RadfCommonResult, _pca, radf_common, radf_common_cv


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
