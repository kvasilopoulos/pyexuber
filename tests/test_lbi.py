"""Tests for exuber.monitor.lbi_test()/monitor_lbi(). Fully deterministic
(no bootstrap/RNG), cross-checked bit-for-bit against R -- see
docs/replication/monitoring/radf_lbi*_validation.py."""

import numpy as np
import pytest
from test_monitor import Y42

from exuber.monitor import _bd_cusum_q, _bd_cusum_weights, lbi_test, monitor_lbi


def test_lbi_test_matches_r():
    res = lbi_test(Y42, sig_lvl=95)
    assert res.stat[0] == pytest.approx(0.025526221649211, abs=1e-8)
    assert res.crit == pytest.approx(1.64485362695147, abs=1e-10)
    assert not res.detected[0]


def test_lbi_test_telescoping_identity():
    # Breitung & Diegel's eq. 4 (their y_0 = 0 convention; here y[0] is the
    # series' own first point, so the general telescoping identity carries
    # an extra -y[0]^2 term): 2*sum(dy_t * y_{t-1}) == y[-1]^2 - y[0]^2 -
    # (n-1)*sigma_tilde^2, sigma_tilde^2 = mean(dy^2) (n-1 differences).
    rng = np.random.default_rng(2)
    y = np.cumsum(rng.normal(size=60))
    n = len(y)
    dy = np.diff(y)
    ylag = y[: n - 1]
    lhs = 2 * np.sum(dy * ylag)
    sigma2_tilde = np.mean(dy**2)
    rhs = y[-1] ** 2 - y[0] ** 2 - (n - 1) * sigma2_tilde
    assert lhs == pytest.approx(rhs, rel=1e-10)


def test_lbi_test_null_distribution_is_standard_normal():
    rng = np.random.default_rng(3)
    stats = np.array([lbi_test(np.cumsum(rng.normal(size=100))).stat[0] for _ in range(500)])
    assert stats.mean() == pytest.approx(0, abs=0.15)
    assert stats.std() == pytest.approx(1, abs=0.15)


def test_lbi_test_rejects_out_of_range_sig_lvl():
    with pytest.raises(ValueError):
        lbi_test(Y42, sig_lvl=30)
    with pytest.raises(ValueError):
        lbi_test(Y42, sig_lvl=100)


def test_bd_cusum_table_lookup():
    assert _bd_cusum_q(95) == pytest.approx(1.95)
    assert _bd_cusum_q(99.5) == pytest.approx(2.80)
    with pytest.raises(ValueError):
        _bd_cusum_q(93)


def test_bd_cusum_weights_sum_of_squares():
    # eq. 12's own discrete closed form: sum(w^2) == 1 exactly for c_bar=0,
    # and to within Riemann-sum approximation error for c_bar>0.
    w0 = _bd_cusum_weights(500, 0)
    assert np.sum(w0**2) == pytest.approx(1.0, abs=1e-12)
    w2 = _bd_cusum_weights(500, 2)
    assert np.sum(w2**2) == pytest.approx(1.0, abs=5e-3)


def test_monitor_lbi_mcusum_matches_r():
    res = monitor_lbi(Y42, r_star=0.5, c_bar=0, sig_lvl=95)
    assert res.t_star == 40
    assert res.boundary == pytest.approx(1.95)
    np.testing.assert_allclose(
        res.stat[-5:, 0],
        [0.5187423335, 0.6196912485, 0.6806364851, 0.5642336848, 0.4197078303],
        atol=1e-8,
    )
    assert np.isnan(res.alarm[0])


def test_monitor_lbi_wcusum_matches_r():
    res = monitor_lbi(Y42, r_star=0.5, c_bar=2, sig_lvl=95)
    np.testing.assert_allclose(
        res.stat[-5:, 0],
        [0.4699810166, 0.6453696898, 0.7566848661, 0.5331770251, 0.2414413064],
        atol=1e-8,
    )
    assert np.isnan(res.alarm[0])


def test_monitor_lbi_rejects_negative_c_bar():
    with pytest.raises(ValueError):
        monitor_lbi(Y42, c_bar=-1)


def test_monitor_lbi_alarm_never_before_t_star():
    rng = np.random.default_rng(4)
    for _ in range(10):
        y = np.cumsum(rng.normal(size=150))
        res = monitor_lbi(y, r_star=0.5)
        if not np.isnan(res.alarm[0]):
            assert res.alarm[0] >= res.t_star
