"""Unit tests for dating.py, ported from exuber's test-rootstamp.R /
test-pdc.R / test-recovery.R. Formula-exact checks use pytest.approx with
a tight tolerance (matches this project's convention in
test_datestamp.py etc.); Monte Carlo checks use loose bounds, per the
same finite-sample honesty this family's R tests document (see
docs/dating-and-root-inference.md).

radf_recovery()/radf_recovery_cv() themselves need radf() (the C++
extension); their end-to-end tests below will only pass where that's
built (CI), same as test_cv.py's radf_mc_cv/radf_wb_cv tests. The
crossing-detection logic is factored into _recovery_dates_from_bsadf(),
tested separately below with synthetic bsadf arrays -- no extension
needed, same split as test_datestamp.py testing datestamp() with a
hand-built RadfResult/RadfCv."""

import math

import numpy as np
import pytest

from exuber.datestamp import Episode
from exuber.dating import (
    RootstampEpisode,
    _dating_hlw_from_episodes,
    _hls_fit_series,
    _hls_model1,
    _hls_model4,
    _hls_model23,
    _hls_prefix_sums,
    _hls_segment_ssr,
    _hlw_local_to_global,
    _knp_find_break,
    _nw_spot_vol,
    _pdc_find_break,
    _pdc_regime_resid,
    _recovery_dates_from_bsadf,
    dating_hls,
    dating_hlw,
    dating_knp,
    dating_pdc,
    radf_recovery,
    radf_recovery_cv,
    rootstamp,
    rootstamp_episodes,
)

# -- rootstamp() --------------------------------------------------------


def test_cauchy_percentiles_match_student_t_df1():
    # standard Cauchy IS Student's t at df=1 -- pure math identity, exact
    published = [6.314, 12.7, 63.65674]
    computed = [
        math.tan(math.pi * (p - 0.5)) for p in (0.95, 0.975, 0.995)
    ]
    for c, p in zip(computed, published, strict=True):
        assert c == pytest.approx(p, rel=1e-3)


def _simulate_ar1(rho, n, seed):
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    e = rng.normal(size=n)
    for t in range(1, n):
        y[t] = rho * y[t - 1] + e[t]
    return y


def test_rootstamp_recovers_known_rho():
    y = _simulate_ar1(1.03, 150, seed=1)
    fit = rootstamp(y)
    assert fit.rho == pytest.approx(1.03, abs=0.01)
    assert fit.se > 0
    assert fit.n == 149


def test_rootstamp_doubling_time_consistent_with_rho():
    y = _simulate_ar1(1.03, 150, seed=1)
    fit = rootstamp(y)
    assert fit.doubling_time == pytest.approx(math.log(2) / math.log(fit.rho))
    # doubling time decreases in rho, so the CI bounds are flipped
    assert fit.doubling_time_ci[0] < fit.doubling_time < fit.doubling_time_ci[1]
    assert fit.rho_ci[0] < fit.rho < fit.rho_ci[1]


def test_rootstamp_coverage_plausible_at_moderate_t():
    rho_true, n = 1.05, 200
    rng = np.random.default_rng(24601)
    covered = 0
    for _ in range(300):
        y = np.zeros(n)
        e = rng.normal(size=n)
        for t in range(1, n):
            y[t] = rho_true * y[t - 1] + e[t]
        ci = rootstamp(y)
        if ci.rho_ci[0] <= rho_true <= ci.rho_ci[1]:
            covered += 1
    # not the nominal 95% -- finite-sample undercoverage is expected and
    # documented (docs/dating-and-root-inference.md); a generous band only.
    assert covered / 300 > 0.80


def test_rootstamp_cauchy_brackets_point_estimate_and_matches_eq27():
    y = _simulate_ar1(1.05, 150, seed=1)
    ci = rootstamp(y, type="cauchy")
    assert ci.rho_ci[0] < ci.rho < ci.rho_ci[1]

    q = math.tan(math.pi * (0.975 - 0.5))
    half_width = q * (ci.rho**2 - 1) / ci.rho**ci.n
    assert ci.rho_ci == pytest.approx((ci.rho - half_width, ci.rho + half_width))


def test_rootstamp_rejects_bad_args():
    y = _simulate_ar1(1.03, 50, seed=1)
    with pytest.raises(ValueError):
        rootstamp(y, type="bogus")
    with pytest.raises(ValueError):
        rootstamp(y, sig_lvl=30)


def test_rootstamp_episodes_matches_direct_call():
    ep = Episode(start=10, peak=15, end=30, duration=20, ongoing=False)
    rng = np.random.default_rng(7)
    y = np.cumsum(rng.normal(size=40))
    ds = {"series1": [ep]}

    out = rootstamp_episodes(y, ds)
    direct = rootstamp(y[ep.start : ep.end])

    assert list(out.keys()) == ["series1"]
    row = out["series1"][0]
    assert isinstance(row, RootstampEpisode)
    assert row.rho == pytest.approx(direct.rho)
    assert row.rho_lower == pytest.approx(direct.rho_ci[0])
    assert row.rho_upper == pytest.approx(direct.rho_ci[1])


def test_rootstamp_episodes_ongoing_slices_to_series_end():
    ep = Episode(start=5, peak=8, end=None, duration=10, ongoing=True)
    rng = np.random.default_rng(8)
    y = np.cumsum(rng.normal(size=20))
    out = rootstamp_episodes(y, {"series1": [ep]})
    direct = rootstamp(y[5:])
    assert out["series1"][0].rho == pytest.approx(direct.rho)


# -- dating_pdc() ---------------------------------------------------------


def test_pdc_find_break_matches_brute_force_rss_scan():
    rng = np.random.default_rng(123)
    y = np.cumsum(rng.normal(size=80))
    trim = 0.05
    break_idx, rss = _pdc_find_break(y, trim)

    n1 = len(y) - 1
    ylag, ycur = y[:n1], y[1 : n1 + 1]
    k_min = max(2, math.ceil(trim * n1))
    k_max = n1 - k_min

    def rss_ols(x, yv):
        beta = np.sum(x * yv) / np.sum(x * x)
        return np.sum((yv - beta * x) ** 2)

    ks = list(range(k_min, k_max + 1))
    rss_brute = [rss_ols(ylag[:k], ycur[:k]) + rss_ols(ylag[k:n1], ycur[k:n1]) for k in ks]
    best = ks[int(np.argmin(rss_brute))]

    assert break_idx == best
    assert rss == pytest.approx(min(rss_brute), abs=1e-8)


def test_pdc_find_break_weights_of_ones_is_a_no_op():
    rng = np.random.default_rng(123)
    y = np.cumsum(rng.normal(size=80))
    res_none = _pdc_find_break(y, 0.05)
    res_ones = _pdc_find_break(y, 0.05, weights=np.ones(len(y) - 1))
    assert res_none[0] == res_ones[0]
    assert res_none[1] == pytest.approx(res_ones[1], abs=1e-10)


def test_pdc_find_break_errors_on_series_too_short_for_trim():
    rng = np.random.default_rng(1)
    y = np.cumsum(rng.normal(size=5))
    with pytest.raises(ValueError, match="too short"):
        _pdc_find_break(y, trim=0.05)


def test_pdc_regime_resid_has_no_gaps_or_overlaps():
    rng = np.random.default_rng(1)
    y = np.cumsum(rng.normal(size=150))
    resid = _pdc_regime_resid(y, [50, 100])
    assert len(resid) == len(y) - 1
    assert np.all(np.isfinite(resid))


def test_dating_pdc_rejects_bad_args():
    rng = np.random.default_rng(1)
    y = np.cumsum(rng.normal(size=80))
    with pytest.raises(ValueError, match="3 or 4"):
        dating_pdc(y, regimes=5)
    with pytest.raises(ValueError, match="ols.*wls|type"):
        dating_pdc(y, type="gls")


def test_dating_pdc_errors_on_series_too_short_for_trim():
    rng = np.random.default_rng(1)
    y = np.cumsum(rng.normal(size=10))
    with pytest.raises(ValueError, match="too short"):
        dating_pdc(y, regimes=4)


def test_dating_pdc_multivariate_panel():
    rng = np.random.default_rng(1)
    y = np.cumsum(rng.normal(size=80))
    panel = np.column_stack([y, y + rng.normal(scale=0.1, size=len(y))])
    out = dating_pdc(panel, regimes=3)
    assert out.origination.shape == (2,)
    assert out.collapse.shape == (2,)
    assert out.recovery is None


def _three_regime_series(seed, n1_len=400, n2_len=200, n3_len=250):
    rng = np.random.default_rng(seed)
    regime1 = np.cumsum(rng.normal(size=n1_len, scale=1))
    regime2 = regime1[-1] * 1.08 ** np.arange(1, n2_len + 1) + np.cumsum(
        rng.normal(size=n2_len, scale=0.1)
    )
    peak = regime2[-1]
    rho3 = 0.5
    regime3 = np.zeros(n3_len)
    regime3[0] = rho3 * peak + rng.normal(scale=0.5)
    for t in range(1, n3_len):
        regime3[t] = rho3 * regime3[t - 1] + rng.normal(scale=0.5)
    return np.concatenate([regime1, regime2, regime3])


def test_dating_pdc_3regime_consistent_in_low_noise_limit():
    # convergence check: as noise shrinks / T grows, the estimate should
    # approach the truth -- distinguishes "asymptotically correct" from
    # "happens to pass one finite-sample tolerance" (see R validation notes).
    y = _three_regime_series(seed=4001)
    out = dating_pdc(y, regimes=3, trim=0.05)
    assert abs(out.origination[0] - 400) <= 2
    assert abs(out.collapse[0] - 600) <= 2


def test_dating_pdc_4regime_consistent_in_low_noise_limit():
    rng = np.random.default_rng(4)
    n1_len, n2_len, n3_len, n4_len = 400, 200, 250, 250
    regime1 = np.cumsum(rng.normal(size=n1_len, scale=0.3))
    regime2 = regime1[-1] * 1.08 ** np.arange(1, n2_len + 1) + np.cumsum(
        rng.normal(size=n2_len, scale=0.05)
    )
    peak = regime2[-1]
    rho3 = 0.5
    regime3 = np.zeros(n3_len)
    regime3[0] = rho3 * peak + rng.normal(scale=0.5)
    for t in range(1, n3_len):
        regime3[t] = rho3 * regime3[t - 1] + rng.normal(scale=0.5)
    regime4 = regime3[-1] + np.cumsum(rng.normal(size=n4_len, scale=0.3))
    y = np.concatenate([regime1, regime2, regime3, regime4])

    out = dating_pdc(y, regimes=4, trim=0.05)
    assert abs(out.origination[0] - 400) <= 3
    assert abs(out.collapse[0] - 600) <= 3
    assert abs(out.recovery[0] - 850) <= 5


def test_dating_pdc_wls_same_shape_as_ols():
    rng = np.random.default_rng(1)
    y = np.cumsum(rng.normal(size=80))
    out_ols = dating_pdc(y, regimes=3, trim=0.05, type="ols")
    out_wls = dating_pdc(y, regimes=3, trim=0.05, type="wls")
    assert out_ols.origination.shape == out_wls.origination.shape


def test_dating_pdc_wls_close_to_ols_under_homoskedasticity():
    y = _three_regime_series(seed=1, n1_len=150, n2_len=80, n3_len=100)
    out_ols = dating_pdc(y, regimes=3, trim=0.05, type="ols")
    out_wls = dating_pdc(y, regimes=3, trim=0.05, type="wls")
    assert abs(out_wls.origination[0] - out_ols.origination[0]) <= 5
    assert abs(out_wls.collapse[0] - out_ols.collapse[0]) <= 5


def test_dating_pdc_wls_beats_ols_under_a_volatility_burst():
    # KS (2023)'s own headline scenario: a volatility burst at the start of
    # the sample biases OLS's unweighted origination split; WLS should not.
    def run(seed):
        rng = np.random.default_rng(seed)
        n1_len, n2_len, n3_len = 150, 80, 100
        burst_len = round(0.2 * n1_len)
        e1 = np.concatenate(
            [rng.normal(size=burst_len, scale=4), rng.normal(size=n1_len - burst_len, scale=0.3)]
        )
        regime1 = np.cumsum(e1)
        regime2 = regime1[-1] * 1.07 ** np.arange(1, n2_len + 1) + np.cumsum(
            rng.normal(size=n2_len, scale=0.15)
        )
        peak = regime2[-1]
        rho3 = 0.5
        regime3 = np.zeros(n3_len)
        regime3[0] = rho3 * peak + rng.normal(scale=0.5)
        for t in range(1, n3_len):
            regime3[t] = rho3 * regime3[t - 1] + rng.normal(scale=0.5)
        y = np.concatenate([regime1, regime2, regime3])
        out_ols = dating_pdc(y, regimes=3, trim=0.05, type="ols")
        out_wls = dating_pdc(y, regimes=3, trim=0.05, type="wls")
        return abs(out_ols.origination[0] - n1_len), abs(out_wls.origination[0] - n1_len)

    errs = [run(s) for s in range(20)]
    mae_ols = np.mean([e[0] for e in errs])
    mae_wls = np.mean([e[1] for e in errs])
    # loose 2x margin, matching the R test's own guard against being brittle
    # to the exact numbers on a different RNG/BLAS.
    assert mae_wls < mae_ols / 2


def test_nw_spot_vol_returns_finite_positive_variance_same_length_as_input():
    rng = np.random.default_rng(1)
    e = rng.normal(size=100)
    sigma2, h = _nw_spot_vol(e)
    assert len(sigma2) == len(e)
    assert np.all(np.isfinite(sigma2))
    assert np.all(sigma2 > 0)
    assert h > 0


def test_nw_spot_vol_rejects_bad_kernel():
    with pytest.raises(ValueError):
        _nw_spot_vol(np.ones(20), kernel="triangular")


# -- radf_recovery() -------------------------------------------------------
# _recovery_dates_from_bsadf(): pure crossing-detection logic, no C++
# extension needed (see module docstring).


def _up_then_down(n_minw=8, up_at=2, down_at=5):
    bsadf = np.zeros((n_minw, 1))
    bsadf[up_at:down_at, 0] = 3.0
    cv = np.tile([1.0, 1.0, 1.0], (n_minw, 1))
    return bsadf, cv


def test_recovery_dates_up_then_down_crossing():
    bsadf, cv = _up_then_down(n_minw=8, up_at=2, down_at=5)
    res = _recovery_dates_from_bsadf(bsadf, cv, minw=5, lag=0, n=13, sig_lvl=95)

    assert res.detected[0]
    assert not res.censored[0]
    zadj = 5
    assert res.f_r[0] == 13 - 1 - (2 + zadj)
    assert res.f_c[0] == 13 - 1 - (5 + zadj)
    assert res.f_c[0] <= res.f_r[0]


def test_recovery_dates_censored_when_no_down_crossing():
    n_minw = 8
    bsadf = np.zeros((n_minw, 1))
    bsadf[2:, 0] = 3.0  # up-crossing, never comes back down
    cv = np.tile([1.0, 1.0, 1.0], (n_minw, 1))
    res = _recovery_dates_from_bsadf(bsadf, cv, minw=5, lag=0, n=13, sig_lvl=95)

    assert res.detected[0]
    assert res.censored[0]
    assert np.isnan(res.f_c[0])
    assert not np.isnan(res.f_r[0])


def test_recovery_dates_not_detected_when_never_crosses():
    n_minw = 8
    bsadf = np.zeros((n_minw, 1))
    cv = np.tile([1.0, 1.0, 1.0], (n_minw, 1))
    res = _recovery_dates_from_bsadf(bsadf, cv, minw=5, lag=0, n=13, sig_lvl=95)

    assert not res.detected[0]
    assert not res.censored[0]
    assert np.isnan(res.f_c[0])
    assert np.isnan(res.f_r[0])


def test_recovery_dates_rejects_bad_sig_lvl():
    bsadf, cv = _up_then_down()
    with pytest.raises(ValueError):
        _recovery_dates_from_bsadf(bsadf, cv, minw=5, lag=0, n=13, sig_lvl=80)


def test_recovery_dates_multiseries_independent():
    n_minw = 8
    bsadf = np.zeros((n_minw, 2))
    bsadf[2:5, 0] = 3.0  # series 0: up then down
    # series 1: never crosses
    cv = np.tile([1.0, 1.0, 1.0], (n_minw, 1))
    res = _recovery_dates_from_bsadf(bsadf, cv, minw=5, lag=0, n=13, sig_lvl=95)

    assert res.detected[0] and not res.detected[1]
    assert not np.isnan(res.f_r[0])
    assert np.isnan(res.f_r[1])


# radf_recovery()/radf_recovery_cv(): need the C++ extension (radf()),
# only pass where it's built -- see module docstring.


def test_radf_recovery_cv_shape():
    cv = radf_recovery_cv(n=100, minw=20, nrep=50, seed=1)
    assert cv.bsadf_cv.shape == (80, 3)
    assert cv.minw == 20
    assert cv.n == 100
    assert cv.iter == 50


def test_radf_recovery_runs_end_to_end():
    rng = np.random.default_rng(1)
    y = np.cumsum(rng.normal(size=100))
    with pytest.warns(UserWarning):
        out = radf_recovery(y, minw=20, nrep=50, seed=1)
    assert out.detected.dtype == bool
    assert out.censored.dtype == bool
    assert out.n == 100


def test_radf_recovery_rejects_bad_sig_lvl():
    rng = np.random.default_rng(1)
    y = np.cumsum(rng.normal(size=100))
    with pytest.raises(ValueError):
        radf_recovery(y, minw=20, nrep=50, sig_lvl=93)


def test_radf_recovery_f_c_never_exceeds_f_r():
    def run_once(seed):
        rng = np.random.default_rng(seed)
        n1, n2, n3 = 40, 25, 35
        expansion = 100 * 1.03 ** np.arange(1, n1 + 1) + np.cumsum(rng.normal(size=n1))
        collapse = expansion[-1] * 0.5 ** (np.arange(1, n2 + 1) / n2) + np.cumsum(
            rng.normal(size=n2)
        )
        recovery = collapse[-1] + np.cumsum(rng.normal(size=n3)) + np.arange(1, n3 + 1) * 0.5
        y = np.concatenate([expansion, collapse, recovery])
        with pytest.warns(UserWarning):
            out = radf_recovery(y, minw=15, nrep=50, seed=1)
        if not out.detected[0] or out.censored[0]:
            return None
        return out.f_c[0] <= out.f_r[0]

    results = [r for r in (run_once(s) for s in range(10)) if r is not None]
    assert all(results)


# -- dating_hls() -----------------------------------------------------------


def _ols_ssr(xseg: np.ndarray, zseg: np.ndarray) -> float:
    a = np.vstack([xseg, np.ones_like(xseg)]).T
    coef, *_ = np.linalg.lstsq(a, zseg, rcond=None)
    return float(np.sum((zseg - a @ coef) ** 2))


def test_hls_segment_ssr_matches_brute_force_ols():
    rng = np.random.default_rng(1)
    y = np.cumsum(rng.normal(size=40))
    ps = _hls_prefix_sums(y)
    n1 = len(y) - 1
    x_all, z_all = y[:n1], np.diff(y)

    for lo, hi in [(0, 10), (10, 25), (4, 39)]:
        manual = _hls_segment_ssr(ps, lo, hi, True)
        brute = _ols_ssr(x_all[lo:hi], z_all[lo:hi])
        assert manual == pytest.approx(brute, abs=1e-8)

    assert _hls_segment_ssr(ps, 0, 10, False) == pytest.approx(np.sum(z_all[:10] ** 2), abs=1e-8)


def test_hls_model1_matches_brute_force_grid_search():
    rng = np.random.default_rng(2)
    n = 30
    y = np.cumsum(rng.normal(size=n))
    ps = _hls_prefix_sums(y)
    tau1, ssr = _hls_model1(y, ps, trim=0.1)

    n1 = n - 1
    x, z = y[:n1], np.diff(y)
    k_min = max(2, math.ceil(0.1 * n1))
    best_ssr, best_tau1 = math.inf, None
    for t1 in range(k_min, n1 - k_min + 1):
        if y[n1] <= y[t1]:
            continue
        s = np.sum(z[:t1] ** 2) + _ols_ssr(x[t1:n1], z[t1:n1])
        if s < best_ssr:
            best_ssr, best_tau1 = s, t1

    assert tau1 == best_tau1
    assert ssr == pytest.approx(best_ssr, abs=1e-6)


def test_hls_model23_matches_brute_force_grid_search():
    rng = np.random.default_rng(3)
    n = 26
    y = np.cumsum(rng.normal(size=n))
    ps = _hls_prefix_sums(y)
    n1 = n - 1
    x, z = y[:n1], np.diff(y)
    k_min = max(2, math.ceil(0.1 * n1))

    def brute(right_fit):
        best_ssr, best = math.inf, None
        for t1 in range(k_min, n1 - 2 * k_min + 1):
            for t2 in range(t1 + k_min, n1 - k_min + 1):
                if y[t2] <= y[t1]:
                    continue
                if right_fit and y[t2] <= y[n1]:
                    continue
                ssr_mid = _ols_ssr(x[t1:t2], z[t1:t2])
                ssr_right = _ols_ssr(x[t2:n1], z[t2:n1]) if right_fit else np.sum(z[t2:n1] ** 2)
                s = np.sum(z[:t1] ** 2) + ssr_mid + ssr_right
                if s < best_ssr:
                    best_ssr, best = s, (t1, t2)
        return best, best_ssr

    for right_fit in (False, True):
        tau1, tau2, ssr = _hls_model23(y, ps, trim=0.1, right_fit=right_fit)
        (b_tau1, b_tau2), b_ssr = brute(right_fit)
        assert (tau1, tau2) == (b_tau1, b_tau2)
        assert ssr == pytest.approx(b_ssr, abs=1e-6)


def test_hls_model4_matches_brute_force_joint_grid_search():
    rng = np.random.default_rng(5)
    n = 24
    y = np.cumsum(rng.normal(size=n))
    ps = _hls_prefix_sums(y)
    tau1, tau2, tau3, ssr = _hls_model4(y, ps, trim=0.1)

    n1 = n - 1
    x, z = y[:n1], np.diff(y)
    k_min = max(2, math.ceil(0.1 * n1))
    best_ssr, best = math.inf, None
    for t1 in range(k_min, n1 - 3 * k_min + 1):
        for t2 in range(t1 + k_min, n1 - 2 * k_min + 1):
            if y[t2] <= y[t1]:
                continue
            for t3 in range(t2 + k_min, n1 - k_min + 1):
                if y[t2] <= y[t3]:
                    continue
                s = (
                    np.sum(z[:t1] ** 2)
                    + _ols_ssr(x[t1:t2], z[t1:t2])
                    + _ols_ssr(x[t2:t3], z[t2:t3])
                    + np.sum(z[t3:n1] ** 2)
                )
                if s < best_ssr:
                    best_ssr, best = s, (t1, t2, t3)

    assert (tau1, tau2, tau3) == best
    assert ssr == pytest.approx(best_ssr, abs=1e-6)


def test_hls_fit_series_restricted_to_models_2_and_4():
    # dating_hlw() restricts non-final windows to models {2, 4} -- verify
    # the shared helper actually honors a restricted model set.
    rng = np.random.default_rng(1)
    y = np.cumsum(rng.normal(size=80))
    fit = _hls_fit_series(y, trim=0.1, models=(2, 4))
    assert fit.model in (2, 4)
    assert np.isinf(fit.bic[0])  # model 1 not requested
    assert np.isinf(fit.bic[2])  # model 3 not requested


def test_dating_hls_end_to_end_well_formed():
    rng = np.random.default_rng(1)
    y = np.cumsum(rng.normal(size=100))
    out = dating_hls(y, trim=0.05)
    assert out.model[0] in (1, 2, 3, 4)
    assert out.bic.shape == (1, 4)
    assert not math.isnan(out.origination[0])


def test_dating_hls_na_pattern_matches_selected_model():
    rng = np.random.default_rng(1)
    y = np.cumsum(rng.normal(size=100))
    out = dating_hls(y, trim=0.05)
    m = out.model[0]
    if m == 1:
        assert math.isnan(out.collapse[0])
        assert math.isnan(out.recovery[0])
    elif m in (2, 3):
        assert math.isnan(out.recovery[0])
    assert not math.isnan(out.origination[0])


def test_dating_hls_multivariate_panel():
    rng = np.random.default_rng(1)
    y = np.cumsum(rng.normal(size=80))
    panel = np.column_stack([y, y + rng.normal(scale=0.1, size=len(y))])
    out = dating_hls(panel, trim=0.05)
    assert out.model.shape == (2,)
    assert out.bic.shape == (2, 4)


def _sim_hls_model4(seed, n1=60, n2=25, n3=25, n4=40, base=100.0, c_bubble=1.05):
    rng = np.random.default_rng(seed)
    unit1 = base + np.cumsum(rng.normal(size=n1))
    bubble = unit1[-1] * c_bubble ** np.arange(1, n2 + 1) + np.cumsum(rng.normal(size=n2))
    target = bubble[-1] * 0.5
    collapse = np.empty(n3)
    collapse[0] = bubble[-1] + rng.normal()
    for k in range(1, n3):
        collapse[k] = target + 0.85 * (collapse[k - 1] - target) + rng.normal()
    recovery = collapse[-1] + np.cumsum(rng.normal(size=n4))
    y = np.concatenate([unit1, bubble, collapse, recovery])
    return y, n1, n1 + n2


def test_dating_hls_recovers_model4_dgp_and_prefers_complex_models():
    results = []
    for seed in range(15):
        y, true_tau1, _true_tau2 = _sim_hls_model4(seed)
        out = dating_hls(y, trim=0.05)
        results.append((out.model[0], out.origination[0] - true_tau1))

    models = [m for m, _ in results]
    bias = [abs(b) for _, b in results]
    assert np.mean([m in (3, 4) for m in models]) > 0.5
    assert np.mean(bias) < 15


def test_dating_hls_h0_does_not_spuriously_prefer_model4():
    models = []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        y = 100 + np.cumsum(rng.normal(size=150))
        out = dating_hls(y, trim=0.05)
        models.append(out.model[0])
    assert np.mean([m == 4 for m in models]) < 0.3


# -- dating_hlw() -----------------------------------------------------------
# _dating_hlw_from_episodes(): pure window-construction + per-window HLS
# fitting, no radf()/datestamp()/the C++ extension needed (see module
# docstring). dating_hlw() itself needs radf()/radf_wb_cv()/datestamp(),
# only passes where the extension is built -- see module docstring, same
# split as radf_recovery() above.


def test_hlw_local_to_global_arithmetic():
    # R's own check: local_tau=5, s=21 (1-indexed) -> i_index=25, position=26.
    # s0 (0-indexed window start) = s - 1 = 20.
    assert _hlw_local_to_global(local_tau=5, s0=20) == 26


def test_dating_hlw_single_episode_reduces_to_dating_hls():
    y, _t1, _t2 = _sim_hls_model4(11)
    n = len(y)
    ep = Episode(start=0, peak=0, end=None, duration=n, ongoing=True)

    hlw_eps = _dating_hlw_from_episodes(y, [ep], n, trim=0.1)
    hls_out = dating_hls(y, trim=0.1)

    assert len(hlw_eps) == 1
    last = hlw_eps[0]
    assert last.model == hls_out.model[0]
    assert last.origination == hls_out.origination[0]
    assert last.collapse == hls_out.collapse[0]
    if math.isnan(hls_out.recovery[0]):
        assert math.isnan(last.recovery)
    else:
        assert last.recovery == hls_out.recovery[0]


def test_dating_hlw_no_episodes_returns_empty_list():
    y = np.cumsum(np.random.default_rng(1).normal(size=50))
    assert _dating_hlw_from_episodes(y, [], n=50, trim=0.1) == []


def _sim_two_bubbles(seed, n1a=50, n2a=20, n3a=30, n1b=50, n2b=20, n3b=30):
    rng = np.random.default_rng(seed)
    e1 = 100 + np.cumsum(rng.normal(size=n1a))
    b1 = e1[-1] * 1.05 ** np.arange(1, n2a + 1) + np.cumsum(rng.normal(size=n2a))
    u1 = b1[-1] + np.cumsum(rng.normal(size=n3a))
    e2 = u1[-1] + np.cumsum(rng.normal(size=n1b))
    b2 = e2[-1] * 1.05 ** np.arange(1, n2b + 1) + np.cumsum(rng.normal(size=n2b))
    u2 = b2[-1] + np.cumsum(rng.normal(size=n3b))
    y = np.concatenate([e1, b1, u1, e2, b2, u2])
    true1 = (n1a, n1a + n2a)
    true2 = (n1a + n2a + n3a + n1b, n1a + n2a + n3a + n1b + n2b)
    return y, true1, true2


def test_dating_hlw_two_windows_recovers_both_bubbles_accurately():
    # hand-built "PSY-detected" episodes (a few observations of slack on
    # either side of the true regime, as a real step-1 pass would give)
    # rather than relying on datestamp()'s own detection noise -- keeps
    # this test independent of the C++ extension while still exercising
    # the real window-construction + sequential-adjustment logic.
    y, true1, true2 = _sim_two_bubbles(1)
    n = len(y)
    ep1 = Episode(start=true1[0] - 5, peak=0, end=true1[1] + 5, duration=0, ongoing=False)
    ep2 = Episode(start=true2[0] - 5, peak=0, end=true2[1] + 5, duration=0, ongoing=False)

    eps = _dating_hlw_from_episodes(y, [ep1, ep2], n, trim=0.1)

    assert len(eps) == 2
    assert eps[0].origination < eps[1].origination
    assert abs(eps[0].origination - true1[0]) < 10
    assert abs(eps[1].origination - true2[0]) < 10


def test_dating_hlw_end_to_end_well_formed():
    rng = np.random.default_rng(11)
    n1, n2, n3, n4 = 60, 25, 25, 40
    unit1 = 100 + np.cumsum(rng.normal(size=n1))
    bubble = unit1[-1] * 1.05 ** np.arange(1, n2 + 1) + np.cumsum(rng.normal(size=n2))
    target = bubble[-1] * 0.5
    collapse = np.empty(n3)
    collapse[0] = bubble[-1] + rng.normal()
    for k in range(1, n3):
        collapse[k] = target + 0.85 * (collapse[k - 1] - target) + rng.normal()
    recovery = collapse[-1] + np.cumsum(rng.normal(size=n4))
    y = np.concatenate([unit1, bubble, collapse, recovery])

    out = dating_hlw(y, trim=0.1, nboot=199, seed=1)
    assert "series1" in out.episodes
    assert isinstance(out.episodes["series1"], list)


def test_dating_hlw_h0_returns_zero_windows():
    rng = np.random.default_rng(2)
    y = 100 + np.cumsum(rng.normal(size=150))
    out = dating_hlw(y, trim=0.1, nboot=199, seed=1)
    assert out.episodes["series1"] == []


# -- dating_knp() -------------------------------------------------------


def test_knp_find_break_omit_false_matches_brute_force():
    rng = np.random.default_rng(3)
    n = 26
    y = np.cumsum(rng.normal(size=n))
    tau1, tau2, ssr = _knp_find_break(y, trim=0.1, omit=False)

    n1 = n - 1
    x, z = y[:n1], np.diff(y)
    k_min = max(2, math.ceil(0.1 * n1))
    best_ssr, best = math.inf, None
    for t1 in range(k_min, n1 - 2 * k_min + 1):
        for t2 in range(t1 + k_min, n1 - k_min + 1):
            s = np.sum(z[:t1] ** 2) + _ols_ssr(x[t1:t2], z[t1:t2]) + np.sum(z[t2:n1] ** 2)
            if s < best_ssr:
                best_ssr, best = s, (t1, t2)

    assert (tau1, tau2) == best
    assert ssr == pytest.approx(best_ssr, abs=1e-6)


def test_knp_find_break_omit_true_matches_brute_force_with_residual_dropped():
    rng = np.random.default_rng(3)
    n = 26
    y = np.cumsum(rng.normal(size=n))
    tau1, tau2, ssr = _knp_find_break(y, trim=0.1, omit=True)

    n1 = n - 1
    x, z = y[:n1], np.diff(y)
    k_min = max(2, math.ceil(0.1 * n1))
    best_ssr, best = math.inf, None
    for t1 in range(k_min, n1 - 2 * k_min + 1):
        for t2 in range(t1 + k_min, n1 - k_min + 1):
            s = (
                np.sum(z[:t1] ** 2) + _ols_ssr(x[t1:t2], z[t1:t2]) + np.sum(z[t2:n1] ** 2)
                - z[t2] ** 2
            )
            if s < best_ssr:
                best_ssr, best = s, (t1, t2)

    assert (tau1, tau2) == best
    assert ssr == pytest.approx(best_ssr, abs=1e-6)


def test_dating_knp_end_to_end_well_formed():
    rng = np.random.default_rng(1)
    y = np.cumsum(rng.normal(size=150))
    out = dating_knp(y, trim=0.05)
    assert not math.isnan(out.origination[0])
    assert not math.isnan(out.collapse[0])
    assert not math.isnan(out.delta[0])


def _sim_knp(seed, t1=50, t2=90, t=200, delta=1.05):
    rng = np.random.default_rng(seed)
    y = np.zeros(t)
    for i in range(1, t1):
        y[i] = y[i - 1] + rng.normal()
    for i in range(t1, t2):
        y[i] = delta * y[i - 1] + rng.normal()
    y[t2] = y[t1 - 1] + rng.normal()
    for i in range(t2 + 1, t):
        y[i] = y[i - 1] + rng.normal()
    return y, t1, t2


def test_dating_knp_omission_correction_reproduces_theorem_1_and_2():
    # Theorem 1: naive (omit=False) tau1 tracks the true COLLAPSE date,
    # not the origination date. Theorem 2: the omission correction
    # substantially reduces the origination-date bias.
    def run(seed, omit):
        y, t1, t2 = _sim_knp(seed)
        tau1, _tau2, _ssr = _knp_find_break(y, trim=0.05, omit=omit)
        return tau1, t1, t2

    res_naive = [run(s, False) for s in range(20)]
    res_om = [run(s, True) for s in range(20)]

    bias_naive_t1 = np.mean([abs(tau1 - t1) for tau1, t1, _t2 in res_naive])
    bias_naive_t2 = np.mean([abs(tau1 - t2) for tau1, _t1, t2 in res_naive])
    bias_om_t1 = np.mean([abs(tau1 - t1) for tau1, t1, _t2 in res_om])

    assert bias_naive_t2 < bias_naive_t1
    assert bias_om_t1 < bias_naive_t1 / 2
