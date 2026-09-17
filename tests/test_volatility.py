"""Tests for exuber.volatility -- the volatility-robust bubble tests.
Ports of exuber's R/radf_tt.R, R/radf_sign.R, R/radf_kp.R, R/radf_sbz.R
and R/ssu_test.R; see docs/volatility-robustness.md (root repo) for the
source papers.

Formula-exact reference numbers below were produced by feeding the SAME
deterministic input series to both R and this port (no RNG involved at
the formula level, so results agree to numpy/R floating-point precision,
not just approximately) -- see
docs/replication/volatility-robustness/radf_tt_validation.py,
sign_based_finite_T_crosscheck.py, radf_kp_validation.py,
radf_sbz_validation.py and radf_ssu_validation.py (root repo) for the
fuller narrative version and the exact R commands used.

radf_kp()'s and radf_sbz_union()'s own tests need exuber._core (the
compiled extension) since they call radf()/the C++ engine internally --
like test_radf.py, these fail to run on a machine without the extension
built, and only run on CI.
"""

import numpy as np
import pytest

from exuber.radf import psy_minw
from exuber.volatility import (
    gls_dfstat_grid,
    kernel_purge,
    kernel_spot_vol,
    radf_kp,
    radf_sbz,
    radf_sbz_cv,
    radf_sbz_union,
    radf_sign,
    radf_sign_cv,
    radf_sign_dm,
    radf_sign_dm_cv,
    radf_tt,
    radf_tt_cv,
    sign_demean_transform,
    sign_transform,
    ssu_q,
    ssu_test,
    variance_profile,
    wls_dfstat_grid,
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


R_SSU_STAT_LAST3 = np.array([-1.4294408532, -1.4956945921, -1.5827171047])
R_SSU_SADF = 0.8058323864


def test_ssu_q_lookup():
    assert ssu_q(90) == pytest.approx(2.90)
    assert ssu_q(95) == pytest.approx(3.30)
    assert ssu_q(99) == pytest.approx(4.20)
    with pytest.raises(ValueError):
        ssu_q(80)


def test_ssu_test_matches_r():
    res = ssu_test(Y_VEC, minw=MINW, sig_lvl=95)
    np.testing.assert_allclose(res.stat[-3:, 0], R_SSU_STAT_LAST3, atol=1e-4)
    assert res.sadf[0] == pytest.approx(R_SSU_SADF, abs=1e-4)
    assert res.crit == pytest.approx(3.30)
    assert bool(res.detected[0]) == (R_SSU_SADF > 3.30)


def test_ssu_test_default_minw_matches_psy_minw():
    y = np.cumsum(np.random.default_rng(1).normal(size=120))
    res = ssu_test(y)
    assert res.minw == psy_minw(120)


def test_ssu_test_formula_matches_brute_force():
    """ssu_stat_path()'s bilinear cross-moment expansion vs. a brute-force
    per-window computation (two separately fitted OLS regressions plus a
    manual residual cross-moment), the same check test-ssu.R's own R
    validation uses."""
    from exuber.volatility import ssu_prefix_sums, ssu_stat_path

    rng = np.random.default_rng(2)
    n = 150
    y = np.cumsum(rng.normal(size=n))
    ps = ssu_prefix_sums(y)

    def brute_force_stat(hi: int) -> float:
        win = np.arange(hi)
        x1 = y[win]
        d1 = y[win + 1] - x1
        x2 = x1**2
        d2 = d1**2

        A1 = np.column_stack([np.ones(hi), x1])
        beta1, *_ = np.linalg.lstsq(A1, d1, rcond=None)
        eps_hat = d1 - A1 @ beta1

        A2 = np.column_stack([np.ones(hi), x2])
        beta2, *_ = np.linalg.lstsq(A2, d2, rcond=None)
        eta_hat = d2 - A2 @ beta2

        sigma2_eps = np.sum(eps_hat**2) / (hi - 2)
        sigma2_eta = np.sum(eta_hat**2) / (hi - 2)
        sigma2_epseta = np.sum(eps_hat * eta_hat) / (hi - 1)
        sigma_eps = np.sqrt(sigma2_eps)
        sigma_eta = np.sqrt(sigma2_eta)
        psi_hat = sigma2_epseta / (sigma_eps * sigma_eta)

        omega_hat = beta2[1]
        sxx2_c = np.sum((x2 - x2.mean()) ** 2)
        t_omega = omega_hat / np.sqrt(sigma2_eta / sxx2_c)

        num_corr = np.sum((x2 - x2.mean()) * d1)
        den_corr = np.sqrt(sxx2_c)
        correction = (psi_hat / sigma_eps) * num_corr / den_corr
        return (t_omega - correction) / np.sqrt(1 - psi_hat**2)

    for hi in (50, 80, 120, 149):
        fast = ssu_stat_path(ps, np.array([hi]))[0]
        manual = brute_force_stat(hi)
        assert fast == pytest.approx(manual, abs=1e-6)


def test_ssu_test_power_on_stochastic_coefficient_dgp():
    """SSU should have decent detection power on the alternative it's
    actually designed for (stochastic, not deterministic, explosive
    coefficient) -- Kurozumi & Nishi's own eq. 2 style DGP."""

    def make_stochastic_bubble(rng: np.random.Generator, n: int, te_frac: float = 0.5,
                                c1: float = 3.0, a: float = 4.0) -> np.ndarray:
        y = np.empty(n)
        y[0] = rng.normal()
        te = round(te_frac * n)
        for t in range(1, n):
            if t < te:
                y[t] = y[t - 1] + rng.normal()
            else:
                rho_t = 1 + c1 / n + a * rng.normal() / np.sqrt(n)
                y[t] = rho_t * y[t - 1] + rng.normal()
        return y

    rng = np.random.default_rng(2)
    n, nrep = 200, 30
    detections = sum(
        ssu_test(make_stochastic_bubble(rng, n), sig_lvl=95).detected[0] for _ in range(nrep)
    )
    power = detections / nrep
    assert power > 0.4  # KN's own MC reports ~85%; a loose lower bound for a small smoke test
