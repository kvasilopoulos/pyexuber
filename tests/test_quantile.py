"""Tests for exuber.quantile_test.quantile_test(). The point statistics
(tstat, tau, delta, and the internal helpers) are deterministic and
cross-checked bit-for-bit against R; the simulated critical value (`crit`)
uses numpy's RNG (see quantile_test.py's module docstring) so only its
shape/sanity is checked, not an exact match."""

import numpy as np
import pytest
from test_monitor import Y42

from exuber.quantile_test import (
    _bw_nrd0,
    _quantile_adf_tstat,
    _quantile_check_density,
    _quantile_regression_fit,
    quantile_test,
)


def test_quantile_adf_tstat_matches_r():
    assert _quantile_adf_tstat(Y42) == pytest.approx(-1.66093811600216, abs=1e-6)


def test_quantile_check_density_matches_r():
    dy = np.diff(Y42)
    b_tau, f_hat = _quantile_check_density(dy, 0.5)
    assert b_tau == pytest.approx(0.0898328865790818, abs=1e-8)
    assert f_hat == pytest.approx(0.369113058891683, abs=1e-6)


def test_bw_nrd0_matches_closed_form():
    # Cross-language RNG streams differ (numpy vs. R), so this checks
    # _bw_nrd0() against its own closed form (0.9 * min(sd, IQR/1.34) *
    # n^-0.2) recomputed independently, rather than an R dput() reference.
    rng = np.random.default_rng(0)
    x = rng.normal(size=50)
    h = _bw_nrd0(x)
    assert h > 0
    sd = np.std(x, ddof=1)
    iqr = np.percentile(x, 75) - np.percentile(x, 25)
    expected = 0.9 * min(sd, iqr / 1.34) * len(x) ** (-0.2)
    assert h == pytest.approx(expected)


def test_quantile_regression_fit_matches_ols_at_tau_half_for_symmetric_errors():
    # For symmetric, homoskedastic errors the tau=0.5 (median) QR fit and
    # the OLS fit should closely agree -- a sanity check on the IRLS
    # solver's convergence, not a formula-exact identity.
    rng = np.random.default_rng(1)
    n = 500
    x = rng.normal(size=n)
    y = 2.0 + 3.0 * x + rng.normal(size=n) * 0.5
    a, b = _quantile_regression_fit(y, x, 0.5)
    design = np.column_stack([np.ones(n), x])
    ols_beta = np.linalg.lstsq(design, y, rcond=None)[0]
    assert a == pytest.approx(ols_beta[0], abs=0.1)
    assert b == pytest.approx(ols_beta[1], abs=0.05)


def test_quantile_regression_fit_recovers_known_quantile_line():
    # y = x + c, c the tau-th quantile of iid noise added on top --
    # the QR fit at that tau should recover slope ~1 and the right offset.
    rng = np.random.default_rng(2)
    n = 2000
    x = rng.uniform(-5, 5, size=n)
    noise = rng.normal(size=n)
    y = x + noise
    tau = 0.75
    a, b = _quantile_regression_fit(y, x, tau)
    from statistics import NormalDist

    expected_a = NormalDist().inv_cdf(tau)
    assert b == pytest.approx(1.0, abs=0.05)
    assert a == pytest.approx(expected_a, abs=0.1)


def test_quantile_test_tstat_and_delta_match_r():
    res = quantile_test(Y42, tau=0.5, nrep=50, sig_lvl=95, seed=7)
    assert res.tstat[0] == pytest.approx(-1.1087521367351, abs=1e-4)
    assert res.delta[0] == pytest.approx(0.7832341878, abs=1e-6)
    assert res.tau[0] == 0.5


def test_quantile_test_optimal_tau_matches_r():
    res = quantile_test(Y42, tau="optimal", nrep=50, sig_lvl=95, seed=7)
    assert res.tau[0] == pytest.approx(0.8)
    assert res.tstat[0] == pytest.approx(-0.440485749709493, abs=1e-4)


def test_quantile_test_rejects_bad_tau():
    with pytest.raises(ValueError):
        quantile_test(Y42, tau=1.5, nrep=20)


def test_quantile_test_rejects_bad_sig_lvl():
    with pytest.raises(ValueError):
        quantile_test(Y42, sig_lvl=80, nrep=20)


def test_quantile_test_crit_shape_and_detected_consistency():
    res = quantile_test(Y42, tau=0.5, nrep=200, sig_lvl=95, seed=1)
    assert res.crit.shape == (1,)
    assert res.detected[0] == (res.tstat[0] > res.crit[0])
