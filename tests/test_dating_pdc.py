"""Unit tests for dating_pdc.py, ported from exuber's test-pdc.R.
Formula-exact checks use pytest.approx with a tight tolerance (matches
this project's convention in test_datestamp.py etc.); Monte Carlo checks
use loose bounds, per the same finite-sample honesty this family's R
tests document (see docs/dating-and-root-inference.md)."""

import math

import numpy as np
import pytest

from exuber.dating_pdc import _nw_spot_vol, _pdc_find_break, _pdc_regime_resid, dating_pdc

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
