"""Tests for exuber.monitor.monitor_cusum(). Fully deterministic (no
bootstrap/RNG at all), so every case here is cross-checked bit-for-bit
against R -- see docs/replication/monitoring/radf_cusum*_validation.py
for the reproduction commands."""

import numpy as np
import pytest
from test_monitor import Y42

from exuber.monitor_cusum import _hb_cusum_finite_q, _one_sided_kernel_spot_vol, monitor_cusum


def test_hb_cusum_finite_table_lookup():
    # exuber/R/monitor_cusum.R's hb_cusum_finite_table, Table 8(i).
    assert _hb_cusum_finite_q(95, 40, 2) == pytest.approx(1.43)  # n snaps to 50
    assert _hb_cusum_finite_q(95, 100, 2) == pytest.approx(1.51)
    assert _hb_cusum_finite_q(90, 20, 10) == pytest.approx(2.02)


def test_hb_cusum_finite_invalid_sig_lvl_raises():
    with pytest.raises(ValueError):
        _hb_cusum_finite_q(93, 50, 2)


def test_cusum_stat_path_matches_brute_force():
    # Independent brute-force cross-check: recompute sigma_hat_t^2 and S_t
    # via separate diff()/sum() calls per monitoring point, mirroring
    # docs/monitoring.md's own "formula-exact check" for this statistic.
    y = Y42
    t_star = 40
    dy = np.diff(y)
    expected_s, expected_b = [], []
    for t in range(t_star, len(y)):
        sigma2 = np.sum(dy[:t] ** 2) / t
        s_t = (y[t] - y[t_star - 1]) / np.sqrt(sigma2)
        c_t = np.sqrt(4.6 + np.log((t + 1) / t_star))
        expected_s.append(s_t)
        expected_b.append(c_t * np.sqrt(t + 1))

    res = monitor_cusum(y, r_star=0.5, b_alpha=4.6, boundary="asymptotic")
    np.testing.assert_allclose(res.stat[:, 0], expected_s, atol=1e-10)
    np.testing.assert_allclose(res.boundary[:, 0], expected_b, atol=1e-10)


def test_monitor_cusum_standard_matches_r():
    res = monitor_cusum(Y42, r_star=0.5, b_alpha=4.6, boundary="asymptotic")
    assert res.t_star == 40
    np.testing.assert_allclose(
        res.stat[-5:, 0],
        [3.6727107022, 4.4016203054, 4.8602235379, 4.0370328378, 3.0016493394],
        atol=1e-8,
    )
    np.testing.assert_allclose(
        res.boundary[-5:, 0],
        [19.9594813397, 20.1153995614, 20.2704388473, 20.4246151364, 20.5779438828],
        atol=1e-8,
    )
    assert np.isnan(res.alarm[0])


def test_monitor_cusum_finite_boundary_matches_r():
    res = monitor_cusum(Y42, r_star=0.5, boundary="finite", sig_lvl=95)
    assert res.b_alpha == pytest.approx(1.43)
    np.testing.assert_allclose(
        res.stat[-5:, 0],
        [3.6727107022, 4.4016203054, 4.8602235379, 4.0370328378, 3.0016493394],
        atol=1e-8,
    )


def test_monitor_cusum_kernel_matches_r():
    res = monitor_cusum(Y42, r_star=0.5, type="kernel", h=20, kernel="gaussian")
    np.testing.assert_allclose(
        res.stat[-5:, 0],
        [4.2409795835, 5.0516103362, 5.5494862292, 4.607070108, 3.2258025215],
        atol=1e-8,
    )
    assert np.isnan(res.alarm[0])


def test_one_sided_kernel_spot_vol_is_causal_and_starts_at_one():
    # AHLTZ's own convention: sigma2_j := 1 for j <= h.
    rng = np.random.default_rng(0)
    dy = rng.normal(size=50)
    sigma2 = _one_sided_kernel_spot_vol(dy, h=20)
    np.testing.assert_allclose(sigma2[:20], 1.0)
    # Only depends on current + past lags: perturbing a future value must
    # not change an earlier spot-variance estimate.
    dy2 = dy.copy()
    dy2[30] += 100.0
    sigma2_2 = _one_sided_kernel_spot_vol(dy2, h=20)
    np.testing.assert_allclose(sigma2[:30], sigma2_2[:30])


def test_monitor_cusum_alarm_never_before_t_star():
    rng = np.random.default_rng(1)
    for _ in range(10):
        y = np.cumsum(rng.normal(size=150))
        res = monitor_cusum(y, r_star=0.5)
        if not np.isnan(res.alarm[0]):
            assert res.alarm[0] >= res.t_star


def test_monitor_cusum_rejects_bad_args():
    with pytest.raises(ValueError):
        monitor_cusum(Y42, boundary="bogus")
    with pytest.raises(ValueError):
        monitor_cusum(Y42, type="bogus")
    with pytest.raises(ValueError):
        monitor_cusum(Y42, kernel="bogus")
