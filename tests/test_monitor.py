"""Tests for exuber.monitor. See docs/replication/monitoring/*.py for the
standalone R-cross-checked validation scripts this test file wires into
pytest (the reference numbers below are duplicated from those scripts'
own dput() output, not re-derived here)."""

import numpy as np
import pytest

from exuber.monitor import (
    _hb_fluc_q,
    _kurozumi_gsadf_q,
    _kurozumi_gsadf_stat,
    _kurozumi_sadf_q,
    monitor,
)

# y <- cumsum(rnorm(80)) with set.seed(42) in R (exuber-project root):
# Rscript -e 'set.seed(42); y <- cumsum(rnorm(80)); dput(y)'
Y42 = np.array(
    [
        1.3709584471,
        0.8062602758,
        1.1693886871,
        1.802251292,
        2.2065196152,
        2.1003950991,
        3.6119170965,
        3.5172580581,
        5.535681772,
        5.4729676729,
        6.7778373272,
        9.0644827199,
        7.6756220188,
        7.3968332519,
        7.2635119156,
        7.8994623136,
        7.6152093922,
        4.9587539713,
        2.5182870427,
        3.8384003885,
        3.5317617944,
        1.7504533604,
        1.5785360046,
        2.7932107038,
        4.6884041651,
        4.2579350335,
        4.0006656507,
        2.2375025655,
        2.6975999203,
        2.0576050444,
        2.5130551676,
        3.2178925048,
        4.2529960268,
        3.6440696514,
        4.1490247747,
        2.4320160956,
        1.6475570873,
        0.7966494931,
        -1.6175581569,
        -1.58143555,
        -1.3754369498,
        -1.7364942483,
        -0.9783310126,
        -1.7050358397,
        -3.0733168841,
        -2.6404988582,
        -3.4518920344,
        -2.0077907727,
        -2.4392369753,
        -1.7835890919,
        -1.4616638267,
        -2.2455027676,
        -0.6697752478,
        -0.0268759421,
        0.0628847045,
        0.3394354518,
        1.0187242679,
        1.1085571544,
        -1.8845329287,
        -1.5996499752,
        -1.9668846179,
        -1.7816540531,
        -1.1998303257,
        0.1999065016,
        -0.5273855579,
        0.7751570742,
        1.1110051939,
        2.1495112926,
        3.0702398609,
        3.7911180238,
        2.7479990852,
        2.6578126986,
        3.2813308606,
        2.3278075028,
        1.7849786883,
        2.3659751859,
        3.1341539238,
        3.5979215123,
        2.7121452149,
        1.6123643163,
    ]
)


def test_kurozumi_sadf_table_lookup_matches_table1():
    # exuber/R/monitor.R's kurozumi_table1, transcribed from Kurozumi
    # (2020) Table 1.
    assert _kurozumi_sadf_q(95, 1) == pytest.approx(1.0381)
    assert _kurozumi_sadf_q(95, 1.2) == pytest.approx(1.0381)  # snaps to s_bar=1
    assert _kurozumi_sadf_q(95, 3) == pytest.approx(1.3330)
    assert _kurozumi_sadf_q(95, 5) == pytest.approx(1.4255)
    assert _kurozumi_sadf_q(90, 1) == pytest.approx(0.6946)
    assert _kurozumi_sadf_q(99, 1) == pytest.approx(1.6474)


def test_kurozumi_sadf_invalid_sig_lvl_raises():
    with pytest.raises(ValueError):
        _kurozumi_sadf_q(93, 1)


def test_kurozumi_gsadf_table_lookup():
    assert _kurozumi_gsadf_q(95, 1, 0.4) == pytest.approx(1.8081)
    assert _kurozumi_gsadf_q(95, 1, 0.8) == pytest.approx(2.3330)
    assert _kurozumi_gsadf_q(95, 1, 0.6) == pytest.approx(1.8081)  # snaps to 0.4


def test_hb_fluc_table_lookup():
    assert _hb_fluc_q(95, 40, 2) == pytest.approx(4.19)  # n snaps to 50
    assert _hb_fluc_q(95, 100, 2) == pytest.approx(4.50)
    assert _hb_fluc_q(90, 20, 10) == pytest.approx(4.12)


def test_hb_fluc_invalid_sig_lvl_raises():
    with pytest.raises(ValueError):
        _hb_fluc_q(93, 50, 2)


def test_kurozumi_gsadf_stat_matches_brute_force_ols():
    # Independent brute-force cross-check: for every monitoring point t
    # and window start k1, the with-intercept ADF t-statistic via a
    # direct OLS fit (np.linalg.lstsq), max'd over k1 -- same structural
    # check exuber/R/monitor.R's own validation performed against lm().
    y = Y42
    t_star, s0 = 40, 0.4
    n = len(y)
    dy = np.diff(y)
    ylag = y[: n - 1]
    k1_max = max(int(np.floor(t_star * s0)), 1)

    expected = []
    for t in range(t_star - 1, n - 1):  # 0-indexed window ends
        best = -np.inf
        for k1 in range(k1_max):
            yy = ylag[k1 : t + 1]
            dd = dy[k1 : t + 1]
            x_mat = np.column_stack([np.ones(len(yy)), yy])
            beta, _, _, _ = np.linalg.lstsq(x_mat, dd, rcond=None)
            resid = dd - x_mat @ beta
            dof = len(yy) - 2
            sigma2 = np.sum(resid**2) / dof
            xtx_inv = np.linalg.inv(x_mat.T @ x_mat)
            se = np.sqrt(sigma2 * xtx_inv[1, 1])
            tstat = beta[1] / se
            best = max(best, tstat)
        expected.append(best)

    actual = _kurozumi_gsadf_stat(y, t_star, s0)
    np.testing.assert_allclose(actual, expected, atol=1e-8)


def test_monitor_kurozumi_gsadf_s0_matches_r():
    # Rscript docs/replication/monitoring/radf_monitor_gsadf_s0_validation.py's
    # own R-side dput() reference (see that script for the exact command).
    mon = monitor(Y42, r_star=0.5, minw=15, boundary="kurozumi", s0=0.4, sig_lvl=95)
    assert mon.t_star == 40
    expected_boundary_tail = np.array(
        [1.3819348577, 1.3826566781, 1.3833643719, 1.3840584815, 1.3847395186]
    )
    expected_stat_tail = np.array(
        [-1.5282745861, -1.5179824638, -1.5031546411, -1.5618190616, -1.5957597584]
    )
    np.testing.assert_allclose(mon.boundary[-5:], expected_boundary_tail, atol=1e-8)
    np.testing.assert_allclose(mon.stat[-5:, 0], expected_stat_tail, atol=1e-8)
    assert np.isnan(mon.alarm[0])


def test_monitor_rejects_unsupported_boundary():
    with pytest.raises(ValueError):
        monitor(Y42, boundary="bootstrap")


def test_monitor_kurozumi_s0_alarm_never_before_t_star():
    rng = np.random.default_rng(0)
    for _ in range(10):
        y = np.cumsum(rng.normal(size=150))
        mon = monitor(y, r_star=0.5, minw=20, boundary="kurozumi", s0=0.4)
        if not np.isnan(mon.alarm[0]):
            assert mon.alarm[0] >= mon.t_star


# The default (s0=0) and boundary="fluc" paths call radf(), which needs the
# compiled _core extension -- not buildable on this dev machine (see
# pyexuber/CLAUDE.md), so these run in CI, not locally.
def test_monitor_kurozumi_s0_default_matches_r():
    mon = monitor(Y42, r_star=0.5, minw=15, boundary="kurozumi", sig_lvl=95)
    assert mon.t_star == 40
    assert mon.boundary[0] == pytest.approx(1.0381)
    expected_stat_tail = np.array(
        [-1.5922477876, -1.5806550833, -1.5647907562, -1.6255928209, -1.660938116]
    )
    np.testing.assert_allclose(mon.stat[-5:, 0], expected_stat_tail, atol=1e-8)
    assert np.isnan(mon.alarm[0])


def test_monitor_fluc_matches_r():
    mon = monitor(Y42, r_star=0.5, minw=15, boundary="fluc", sig_lvl=95)
    assert mon.t_star == 40
    assert mon.boundary[0] == pytest.approx(4.19)
    expected_stat_tail = np.array(
        [-1.5922477876, -1.5806550833, -1.5647907562, -1.6255928209, -1.660938116]
    )
    np.testing.assert_allclose(mon.stat[-5:, 0], expected_stat_tail, atol=1e-8)
    assert np.isnan(mon.alarm[0])
