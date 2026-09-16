"""Tests for exuber.volatility -- the volatility-robust bubble tests.
Ports of exuber's R/radf_tt.R, R/radf_sign.R and R/radf_kp.R so far; see
docs/volatility-robustness.md (root repo) for the source papers. More
volatility-robust tests land here in follow-up commits.

Formula-exact reference numbers below were produced by feeding the SAME
deterministic input series to both R and this port (no RNG involved at
the formula level, so results agree to numpy/R floating-point precision,
not just approximately) -- see
docs/replication/volatility-robustness/radf_tt_validation.py,
sign_based_finite_T_crosscheck.py and radf_kp_validation.py (root repo)
for the fuller narrative version and the exact R commands used.

radf_kp()'s own tests need exuber._core (the compiled extension) since it
calls radf() internally -- like test_radf.py, these fail to run on a
machine without the extension built, and only run on CI.
"""

import numpy as np
import pytest

from exuber.radf import psy_minw
from exuber.volatility import (
    gls_dfstat_grid,
    kernel_purge,
    kernel_spot_vol,
    radf_kp,
    radf_sign,
    radf_sign_cv,
    radf_sign_dm,
    radf_sign_dm_cv,
    radf_tt,
    radf_tt_cv,
    sign_demean_transform,
    sign_transform,
    variance_profile,
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


def test_gls_dfstat_grid_matches_r():
    res = gls_dfstat_grid(Y_VEC, MINW)
    assert res["sadf"] == pytest.approx(1.2107378397, abs=1e-6)
    assert res["gsadf"] == pytest.approx(1.9637889135, abs=1e-6)


def test_variance_profile_matches_r():
    eta_grid, omega2 = variance_profile(Y_VEC, kernel="uniform")
    assert eta_grid[-1] == pytest.approx(1.0, abs=1e-6)
    assert eta_grid[0] == pytest.approx(0.0, abs=1e-6)
    assert omega2 == pytest.approx(0.8233400111, abs=1e-6)


def test_radf_tt_matches_r():
    res = radf_tt(Y_VEC, minw=MINW, kernel="uniform")
    assert res.adf[0] == pytest.approx(0.3401194521, abs=1e-6)
    assert res.sadf[0] == pytest.approx(1.1571296746, abs=1e-6)
    assert res.gsadf[0] == pytest.approx(1.8910439255, abs=1e-6)
    assert res.minw == MINW
    assert res.lag == 0


def test_radf_tt_cv_against_published_whitehouse():
    """Pivotal (RNG-agnostic) asymptotic target -- comparable directly to
    Whitehouse (2019)'s published STADF triple even with numpy's RNG."""
    cv = radf_tt_cv(n=300, minw=30, nrep=800, seed=555)
    published = np.array([2.319, 2.626, 3.223])
    assert np.all(np.abs(cv.sadf_cv - published) < 0.4)


def test_radf_tt_cv_badf_cv_identity():
    cv = radf_tt_cv(n=60, minw=15, nrep=100, seed=1)
    np.testing.assert_allclose(cv.badf_cv[-1], cv.adf_cv)


def test_radf_tt_default_minw_uses_psy_minw():
    res = radf_tt(Y_VEC)
    assert res.minw == psy_minw(len(Y_VEC))


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


R_KSV_SIGMA2 = np.array(
    [
        0.4959746477, 0.5312260867, 0.5716433342, 0.6169370612, 0.6874127672,
        0.8412310930, 1.1577525057, 1.6723606201, 2.3107950220, 2.8881768207,
        3.1903140074, 3.0969314316, 2.6568967882, 2.0450515634, 1.4477027141,
        0.9864934574, 0.7115504863, 0.6200852851, 0.6744228299, 0.8193921797,
        0.9912734704, 1.1188219752, 1.1404876822, 1.0422514405, 0.8736254828,
        0.7103030842, 0.5977433306, 0.5317693774, 0.4823425959, 0.4254286845,
        0.3600372811, 0.3091871957, 0.3044556076, 0.3582846686, 0.4454091397,
        0.5159624129, 0.5322132895, 0.4917189956, 0.4177513067,
    ]
)
R_KSV_H = 0.0533043701

R_PURGED = np.array(
    [
        -1.7873025622, -2.7815277011, -3.3347316743, -4.5671929336, -5.7535320421,
        -4.8328468384, -4.9568680837, -4.8312994075, -3.3989648256, -3.1994688263,
        -1.7867645834, -0.5703660507, -0.3766526498, 0.9732799329, 1.3858598183,
        0.3777725907, -0.0501407948, -0.0571310824, 1.2267423867, 2.1748043745,
        2.8717529025, 4.0401985161, 2.8275287339, 4.0251706963, 4.2266895665,
        5.1809673847, 5.9841391137, 4.6283810407, 4.2390555386, 2.9405290986,
        4.1438219706, 4.3633073113, 4.1957073655, 3.4324194005, 2.6141599989,
        3.8927306251, 2.4765274358, 2.2724582562, 2.8167023625,
    ]
)

R_KP_ADF = -0.9285832329
R_KP_SADF = 0.6113330426
R_KP_GSADF = 0.8288627249


def test_kernel_spot_vol_matches_r():
    sigma2, h = kernel_spot_vol(Y_VEC, kernel="gaussian")
    np.testing.assert_allclose(sigma2, R_KSV_SIGMA2, atol=1e-4)
    assert h == pytest.approx(R_KSV_H, abs=1e-6)


def test_kernel_purge_matches_r():
    purged = kernel_purge(Y_VEC, kernel="gaussian")
    np.testing.assert_allclose(purged, R_PURGED, atol=1e-4)


def test_radf_kp_matches_r():
    """Needs exuber._core (radf_kp() calls radf() internally) -- runs on
    CI only, like test_radf.py's own golden-fixture tests."""
    res = radf_kp(Y_VEC, minw=MINW)
    assert res.adf[0] == pytest.approx(R_KP_ADF, abs=1e-4)
    assert res.sadf[0] == pytest.approx(R_KP_SADF, abs=1e-4)
    assert res.gsadf[0] == pytest.approx(R_KP_GSADF, abs=1e-4)
