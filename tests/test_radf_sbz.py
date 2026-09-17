"""Tests for exuber.radf_sbz -- the WLS + kernel-volatility SBZ bubble
test. Port of exuber's R/radf_sbz.R; see docs/volatility-robustness.md
(root repo) for the source paper.

Formula-exact reference numbers below were produced by feeding the SAME
deterministic input series to both R and this port (no RNG involved at
the formula level, so results agree to numpy/R floating-point precision,
not just approximately) -- see
docs/replication/volatility-robustness/radf_sbz_validation.py (root
repo) for the fuller narrative version and the exact R commands used.

radf_sbz_union()'s own test needs exuber._core (the compiled extension)
since it calls radf()/the C++ engine internally for the classic supDF
statistic -- like test_radf.py, it fails to run on a machine without the
extension built, and only runs on CI. radf_sbz()/radf_sbz_cv() need no
extension (pure Python wls_dfstat_grid()/kernel_spot_vol()).
"""

import numpy as np
import pytest

from exuber._kernel_vol import kernel_spot_vol
from exuber.radf_sbz import radf_sbz, radf_sbz_cv, radf_sbz_union, wls_dfstat_grid

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

R_WLS_BADF_LAST3 = np.array([-0.0031701844, -0.0449919654, 0.0608511749])
R_WLS_SADF = 1.6435722442
R_WLS_GSADF = 1.7667532509
R_SBZ_ADF = 0.0608511749
R_SBZ_SADF = 1.6435722442
R_SBZ_GSADF = 1.7667532509


def test_wls_dfstat_grid_matches_r():
    sigma2, _ = kernel_spot_vol(Y_VEC, kernel="gaussian")
    res = wls_dfstat_grid(Y_VEC, sigma2, MINW)
    np.testing.assert_allclose(res["badf"][-3:], R_WLS_BADF_LAST3, atol=1e-4)
    assert res["sadf"] == pytest.approx(R_WLS_SADF, abs=1e-4)
    assert res["gsadf"] == pytest.approx(R_WLS_GSADF, abs=1e-4)


def test_radf_sbz_matches_r():
    """radf_sbz() itself needs no exuber._core -- it's built entirely on
    wls_dfstat_grid()/kernel_spot_vol() (pure Python), unlike radf_kp()."""
    res = radf_sbz(Y_VEC, minw=MINW)
    assert res.adf[0] == pytest.approx(R_SBZ_ADF, abs=1e-4)
    assert res.sadf[0] == pytest.approx(R_SBZ_SADF, abs=1e-4)
    assert res.gsadf[0] == pytest.approx(R_SBZ_GSADF, abs=1e-4)
    assert res.minw == MINW


def test_radf_sbz_cv_shape_and_monotonic():
    """radf_sbz_cv() also needs no exuber._core (its wild bootstrap DGP is
    pure Python, unlike radf_wb_cv()'s reuse of the compiled radf_stat())."""
    cv = radf_sbz_cv(Y_VEC, minw=MINW, nboot=60, seed=1)
    assert cv.sadf_cv.shape == (1, 3)
    assert np.all(np.diff(cv.sadf_cv[0]) >= 0)
    np.testing.assert_allclose(cv.badf_cv[-1, :, 0], cv.adf_cv[0])


def test_radf_sbz_cv_empirical_size_under_h0():
    """Empirical rejection rate under H0 (pure random walk) should be
    roughly nominal, not grossly oversized -- the same check that caught
    R's own off-by-one bootstrap-indexing bug (see
    docs/volatility-robustness.md, "SBZ", "found and fixed a real bug")."""
    rng = np.random.default_rng(13579)
    n, nrep, nboot = 100, 25, 80
    rejections = 0
    for _ in range(nrep):
        y = np.cumsum(rng.normal(size=n))
        res = radf_sbz(y, minw=15)
        cv = radf_sbz_cv(y, minw=15, nboot=nboot, seed=int(rng.integers(1_000_000_000)))
        if res.sadf[0] > cv.sadf_cv[0, 1]:
            rejections += 1
    # nominal 0.05; generous bound for a small nrep/nboot smoke test
    assert rejections / nrep < 0.25


def test_radf_sbz_union_runs_and_orders_correctly():
    """Needs exuber._core (supDF via the compiled radf_stat())."""
    y = np.cumsum(np.random.default_rng(1).normal(size=60))
    res = radf_sbz_union(y, minw=15, nboot=40, seed=1)
    assert res.U[0] >= res.supDF[0]  # U := max(supDF, ratio*supBZ)
    assert 0.0 <= res.p_supDF[0] <= 1.0
    assert 0.0 <= res.p_supBZ[0] <= 1.0
    assert 0.0 <= res.p_U[0] <= 1.0
