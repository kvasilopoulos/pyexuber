"""Tests for exuber.contagion_reg. Several checks mirror docs/replication/
multivariate/radf_contagion_validation.py (same checks, re-implemented
here rather than imported -- that script lives in the umbrella repo, one
level up from pyexuber's own repo boundary, so pyexuber's CI (which only
checks out `kvasilopoulos/pyexuber`) can't reach it)."""

import numpy as np
import pytest

from exuber.contagion_reg import (
    _contagion_bandwidth_cv,
    _contagion_fixed_window_beta,
    _contagion_loocv_sse,
    _contagion_nw_delta2,
    contagion_reg,
)

# -- contagion_reg() ----------------------------------------------------------


def test_contagion_fixed_window_beta_matches_lstsq():
    """eq. 1 vs. brute-force lstsq -- mirrors
    docs/replication/multivariate/radf_contagion_validation.py's check 1."""
    rng = np.random.default_rng(1)
    n, S = 150, 50
    core = np.cumsum(rng.normal(size=n))
    t_core, beta_core = _contagion_fixed_window_beta(core, S)
    pos = {int(t): i for i, t in enumerate(t_core)}

    for t_check in (60, 100, 150):
        win = core[t_check - S : t_check]
        design = np.column_stack([np.ones(S - 1), win[:-1]])
        beta_lstsq, *_ = np.linalg.lstsq(design, win[1:], rcond=None)
        assert abs(beta_core[pos[t_check]] - beta_lstsq[1]) < 1e-8


def test_contagion_nw_delta2_matches_manual_wls():
    """eq. 6 vs. a manual weighted-least-squares ratio."""
    rng = np.random.default_rng(1)
    n, S = 150, 50
    core = np.cumsum(rng.normal(size=n))
    y = 0.5 * core + np.cumsum(rng.normal(scale=0.5, size=n))
    t_core, beta_core = _contagion_fixed_window_beta(core, S)
    t_j, beta_j = _contagion_fixed_window_beta(y, S)
    d, r_test, h_test = 2, np.array([0.5]), 0.2

    fast = _contagion_nw_delta2(t_core, beta_core, t_j, beta_j, n, r_test, h_test, d)[0]

    bcore_c = beta_core - beta_core.mean()
    bj_c = beta_j - beta_j.mean()
    pos = {int(t): i for i, t in enumerate(t_core)}
    idx = np.array([pos.get(int(t) - d, -1) for t in t_j])
    valid = idx >= 0
    s2 = t_j[valid]
    bjc = bj_c[valid]
    csh = bcore_c[idx[valid]]
    w = np.exp(-0.5 * ((s2 / n - r_test[0]) / h_test) ** 2) / np.sqrt(2 * np.pi) / h_test
    manual = np.sum(w * bjc * csh) / np.sum(w * csh**2)

    assert abs(fast - manual) < 1e-10


def test_contagion_bandwidth_cv_interior_optimum():
    """eq. 7: LOOCV-picked bandwidth is no worse than either H_T endpoint."""
    rng = np.random.default_rng(1)
    n, S, d = 150, 50, 2
    core = np.cumsum(rng.normal(size=n))
    y = 0.5 * core + np.cumsum(rng.normal(scale=0.5, size=n))
    t_core, beta_core = _contagion_fixed_window_beta(core, S)
    t_j, beta_j = _contagion_fixed_window_beta(y, S)

    h_opt = _contagion_bandwidth_cv(t_core, beta_core, t_j, beta_j, n, d)
    m = len(beta_j)
    H_T = (m ** (-1 / 2), m ** (-1 / 10))
    sse_opt = _contagion_loocv_sse(h_opt, t_core, beta_core, t_j, beta_j, n, d)
    sse_lo = _contagion_loocv_sse(H_T[0], t_core, beta_core, t_j, beta_j, n, d)
    sse_hi = _contagion_loocv_sse(H_T[1], t_core, beta_core, t_j, beta_j, n, d)
    assert sse_opt <= sse_lo + 1e-8
    assert sse_opt <= sse_hi + 1e-8


def test_contagion_reg_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        contagion_reg(np.zeros(10), np.zeros(9))


def test_contagion_reg_shapes_and_planted_signal():
    """Directional sensible-behavior check, averaged over several seeds
    (a single draw is noisy at this sample size -- the R/Python
    validation scripts both average over 15 reps for the same reason): a
    satellite series whose local persistence tracks the core's own shows
    a wider mean estimated delta_2(r) range than an independent series."""
    n, S, nrep = 150, 50, 8
    planted_range = np.empty(nrep)
    indep_range = np.empty(nrep)
    out_planted = None
    for i in range(nrep):
        rng = np.random.default_rng(1000 + i)
        core = np.cumsum(rng.normal(size=n))
        y_planted = np.empty(n)
        y_planted[:10] = rng.normal(size=10)
        for t in range(10, n):
            local_rho = 0.5 + 0.4 * np.tanh((core[max(t - 3, 0)] - core[max(t - 13, 0)]) / 5)
            y_planted[t] = local_rho * y_planted[t - 1] + rng.normal()
        y_indep = np.cumsum(rng.normal(size=n))

        out_planted = contagion_reg(y_planted, core, S=S, d=3, h=0.3)
        out_indep = contagion_reg(y_indep, core, S=S, d=3, h=0.3)
        planted_range[i] = np.ptp(out_planted.delta2)
        indep_range[i] = np.ptp(out_indep.delta2)

    assert out_planted.delta2.shape == (100,)
    assert out_planted.h == 0.3
    assert planted_range.mean() > indep_range.mean()
