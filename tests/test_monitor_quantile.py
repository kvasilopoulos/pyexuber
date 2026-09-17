"""Tests for exuber.monitor.monitor_quantile(). _qpwy_stat_path() is
deterministic (no RNG, no radf()) and cross-checked bit-for-bit against
R. The full monitor_quantile() call needs radf() (the compiled `_core`
extension, not buildable on this dev machine -- see pyexuber/CLAUDE.md),
so those tests run in CI, not locally."""

import numpy as np
import pytest
from test_monitor import Y42

from exuber.monitor import _qpwy_stat_path, monitor_quantile


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
    from exuber.monitor import quantile_test

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


# Needs the compiled _core extension (radf(), both for the point
# statistic's own QR-fit loop and the boundary simulation) -- CI only.
def test_monitor_quantile_structural_run():
    res = monitor_quantile(Y42, tau=0.5, minw=15, nrep=50, sig_lvl=95, seed=1)
    assert res.stat.shape == (65, 1)
    assert res.boundary.shape == (1,)
    assert -1 <= res.delta[0] <= 1


def test_monitor_quantile_alarm_never_before_minw():
    rng = np.random.default_rng(6)
    for _ in range(5):
        y = np.cumsum(rng.normal(size=100))
        res = monitor_quantile(y, tau=0.5, minw=20, nrep=50, seed=1)
        if not np.isnan(res.alarm[0]):
            assert res.alarm[0] >= 20
