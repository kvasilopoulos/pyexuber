"""Tests for exuber.radf_sign -- the sign-based sGSADF/s-bar-GSADF bubble
tests. Port of exuber's R/radf_sign.R; see docs/volatility-robustness.md
(root repo) for the source paper.

Formula-exact reference numbers below were produced by feeding the SAME
deterministic input series to both R and this port (no RNG involved at
the formula level, so results agree to numpy/R floating-point precision,
not just approximately) -- see
docs/replication/volatility-robustness/sign_based_finite_T_crosscheck.py
(root repo) for the fuller narrative version and the exact R commands
used.
"""

import numpy as np
import pytest

from exuber.radf_sign import (
    radf_sign,
    radf_sign_cv,
    radf_sign_dm,
    radf_sign_dm_cv,
    sign_demean_transform,
    sign_transform,
)

# set.seed(7); y <- round(cumsum(rnorm(40)), 8)
Y_VEC = np.array(
    [
        2.28724716, 1.09047548, 0.39618297, -0.01610998, -0.98678332, -1.93406327,
        -1.18592393, -1.30287915, -1.15022153, 1.03975658, 1.39674281, 4.11349459,
        6.39494652, 6.71896706, 8.61503413, 9.08271464, 8.18891391, 7.88158561,
        7.87676319, 8.86492734, 9.7046777, 10.41001953, 11.71598425, 10.32798804,
        11.6009049, 11.78509767, 12.53737757, 13.12912262, 12.14607002, 11.87000607,
        10.99915505, 11.7178656, 11.82851848, 11.75005171, 11.32956125, 10.76743537,
        11.76494882, 10.65981876, 10.51753093, 10.83252583,
    ]
)
MINW = 10


def test_sign_transform_shape():
    y = np.cumsum(np.random.default_rng(0).normal(size=30))
    ct = sign_transform(y)
    assert len(ct) == len(y)
    assert ct[0] == 0.0


def test_sign_demean_transform_matches_brute_force():
    rng = np.random.default_rng(1)
    y = np.cumsum(rng.normal(size=50))
    dy = np.diff(y)
    s = np.sign(dy)
    n = len(s)
    brute = [0.0]
    for t in range(1, n + 1):
        brute.append(sum(s[i - 1] - np.mean(s[:i]) for i in range(1, t + 1)))
    np.testing.assert_allclose(sign_demean_transform(y), brute, atol=1e-10)


def test_radf_sign_matches_r():
    res = radf_sign(Y_VEC, minw=MINW)
    assert res.sadf[0] == pytest.approx(0.6739661980, abs=1e-6)
    assert res.gsadf[0] == pytest.approx(2.0040132787, abs=1e-6)


def test_radf_sign_dm_matches_r():
    res = radf_sign_dm(Y_VEC, minw=MINW)
    assert res.sadf[0] == pytest.approx(3.6784167432, abs=1e-6)
    assert res.gsadf[0] == pytest.approx(3.6784167432, abs=1e-6)


def test_radf_sign_cv_against_published_table1():
    """Pivotal (RNG-agnostic) target -- Harvey, Leybourne & Zu (2020)'s
    Table 1 finite-T=200 row at minw/n = 0.1. Deliberately the T=200 row,
    not the T=Inf asymptotic one: the paper's own text documents sPSY's
    finite-sample critical values converging to the asymptotic limit much
    more slowly than sPWY's, so an exact finite-T match avoids that
    convergence ambiguity (see sign_based_finite_T_crosscheck.R/.py)."""
    cv = radf_sign_cv(n=200, minw=20, nrep=800, seed=1)
    published_sadf = np.array([2.405, 2.735, 3.434])
    published_gsadf = np.array([3.469, 3.901, 4.957])
    assert np.all(np.abs(cv.sadf_cv - published_sadf) < 0.5)
    assert np.all(np.abs(cv.gsadf_cv - published_gsadf) < 0.7)


def test_radf_sign_exact_heteroskedasticity_invariance():
    """The paper's central claim: sadf/gsadf are exactly invariant to
    (even wildly time-varying) volatility, since sign() strips magnitude."""
    rng = np.random.default_rng(7)
    n = 100
    raw_dy = rng.normal(size=n - 1)
    y_homo = np.cumsum(raw_dy)
    vol_pattern = np.concatenate([np.full(30, 0.1), np.full(40, 10.0), np.full(n - 71, 1.0)])
    y_hetero = np.cumsum(raw_dy * vol_pattern)

    r_homo = radf_sign(y_homo, minw=15)
    r_hetero = radf_sign(y_hetero, minw=15)
    np.testing.assert_allclose(r_homo.sadf, r_hetero.sadf)
    np.testing.assert_allclose(r_homo.gsadf, r_hetero.gsadf)


def test_radf_sign_dm_cv_shape_and_monotonic():
    cv = radf_sign_dm_cv(n=60, minw=15, nrep=100, seed=1)
    assert cv.sadf_cv.shape == (3,)
    assert np.all(np.diff(cv.sadf_cv) >= 0)
    np.testing.assert_allclose(cv.badf_cv[-1], cv.adf_cv)
