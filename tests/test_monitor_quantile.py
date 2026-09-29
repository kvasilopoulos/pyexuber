"""Tests for exuber.monitor_quantile.monitor_quantile(). The point
statistics are deterministic and cross-checked against R (within the
IRLS-vs-simplex QR tolerance); the boundary simulation is pure numpy, so
the whole call runs without the compiled `_core` extension."""

import numpy as np
import pytest
from test_monitor import Y42

from exuber.monitor_quantile import (
    _qpsy_stat_path,
    _qpwy_stat_path,
    _quantile_boundary_sim,
    monitor_quantile,
)


def test_qpwy_stat_path_matches_r():
    minw = 15
    r_idx = np.arange(minw + 1, len(Y42) + 1)
    stat = _qpwy_stat_path(Y42, 0.5, r_idx)
    assert len(stat) == 65
    expected_tail = np.array(
        [-1.128241465, -1.2210267401, -1.2384527008, -1.2255226955, -1.1087521367]
    )
    np.testing.assert_allclose(stat[-5:], expected_tail, atol=1e-4)


def test_qpwy_stat_path_last_value_matches_quantile_test_tstat():
    # Structural check (Corollary 2's own claim, restated): at the full
    # sample window, QPWY_n(tau) is exactly quantile_test()'s own tstat.
    from exuber.quantile_test import quantile_test

    minw = 15
    r_idx = np.arange(minw + 1, len(Y42) + 1)
    stat = _qpwy_stat_path(Y42, 0.5, r_idx)
    qt = quantile_test(Y42, tau=0.5, nrep=10, sig_lvl=95, seed=1)
    assert stat[-1] == pytest.approx(qt.tstat[0], abs=1e-6)


def test_monitor_quantile_rejects_bad_args():
    with pytest.raises(ValueError):
        monitor_quantile(Y42, tau=1.5, nrep=5)
    with pytest.raises(ValueError):
        monitor_quantile(Y42, sig_lvl=80, nrep=5)


def test_quantile_boundary_sim_matches_brute_force():
    n, minw, delta = 30, 8, np.array([0.3, 0.9])
    for qpsy in (False, True):
        sim = _quantile_boundary_sim(n, minw, 2, delta, qpsy, np.random.default_rng(11))
        rng = np.random.default_rng(11)
        for i in range(2):
            e = rng.normal(size=n - 1)
            v = rng.normal(size=n - 1)
            x = np.concatenate([[0.0], np.cumsum(e)])[: n - 1]
            u = []
            for hi in range(minw, n):
                for lo in range(hi - minw + 1) if qpsy else [0]:
                    xb = x[lo:hi] - x[lo:hi].mean()
                    s = np.sqrt(np.sum(xb**2))
                    q = np.sum(xb * e[lo:hi]) / s
                    z = np.sum(xb * v[lo:hi]) / s
                    u.append(delta * q + np.sqrt(1 - delta**2) * z)
            np.testing.assert_allclose(sim[i], np.max(u, axis=0), atol=1e-8)


def test_boundary_treats_z_as_a_process():
    # delta = 0 leaves only Z; one shared z per path would make sup_r Z
    # exactly N(0,1), 95% quantile ~1.645
    sim = _quantile_boundary_sim(150, 20, 400, np.array([0.0]), False, np.random.default_rng(3))
    assert np.quantile(sim[:, 0], 0.95) > 2


def test_qpsy_grid_contains_qpwy_path():
    d = np.array([0.2, 0.8])
    wy = _quantile_boundary_sim(60, 12, 20, d, False, np.random.default_rng(5))
    sy = _quantile_boundary_sim(60, 12, 20, d, True, np.random.default_rng(5))
    assert np.all(sy >= wy - 1e-12)


def test_qpsy_stat_path_first_value_is_qpwy():
    minw = 15
    r_idx = np.arange(minw + 1, len(Y42) + 1)
    sy = _qpsy_stat_path(Y42, 0.7, r_idx, minw)
    assert sy[0] == pytest.approx(_qpwy_stat_path(Y42, 0.7, r_idx[:1])[0], abs=1e-12)
    assert np.all(sy >= _qpwy_stat_path(Y42, 0.7, r_idx) - 1e-12)


def test_monitor_quantile_structural_run():
    res = monitor_quantile(Y42, tau=0.5, minw=15, nrep=50, sig_lvl=95, seed=1)
    assert res.stat.shape == (65, 1)
    assert res.boundary.shape == (1,)
    assert -1 <= res.delta[0] <= 1
    res2 = monitor_quantile(Y42[:40], tau=0.5, minw=10, nrep=20, seed=1, type="qpsy")
    assert res2.type == "qpsy"
    assert res2.stat.shape == (30, 1)
    with pytest.raises(ValueError):
        monitor_quantile(Y42, type="psy")
    with pytest.warns(UserWarning, match="oversized"):
        monitor_quantile(Y42[:40], tau=0.8, minw=10, nrep=20, seed=1, type="qpsy")


def test_monitor_quantile_alarm_never_before_minw():
    rng = np.random.default_rng(6)
    for _ in range(5):
        y = np.cumsum(rng.normal(size=100))
        res = monitor_quantile(y, tau=0.5, minw=20, nrep=50, seed=1)
        if not np.isnan(res.alarm[0]):
            assert res.alarm[0] >= 20
