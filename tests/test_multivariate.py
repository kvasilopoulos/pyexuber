"""Tests for exuber.multivariate. Several checks mirror docs/replication/
multivariate/*.py (same checks, re-implemented here rather than imported
-- those scripts live in the umbrella repo, one level up from pyexuber's
own repo boundary, so pyexuber's CI (which only checks out
`kvasilopoulos/pyexuber`) can't reach them; see those scripts' module
docstrings for full narrative, R reference numbers, and (for
radf_common_validation.py) why that boundary exists)."""

import numpy as np
import pytest

from exuber.cv import radf_mc_cv
from exuber.multivariate import (
    RadfCommonCv,
    RadfCommonResult,
    _coexplosive_select_lag,
    _coexplosive_stat,
    _contagion_bandwidth_cv,
    _contagion_fixed_window_beta,
    _contagion_loocv_sse,
    _contagion_nw_delta2,
    _pca,
    cobubble_test,
    contagion_reg,
    radf_common,
    radf_common_cv,
)
from exuber.radf import psy_minw


def test_pca_formula_exact():
    """_pca() (radf_common()'s PCA step, factored out so it's testable
    without the compiled _core extension) against an independent
    eigendecomposition of the sample covariance matrix."""
    rng = np.random.default_rng(0)
    n, nc, r = 60, 5, 2
    x = np.cumsum(rng.normal(size=(n, nc)), axis=0)

    loadings, scores, explained = _pca(x, r)

    xc = x - x.mean(axis=0)
    cov = (xc.T @ xc) / (n - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    for j in range(r):
        sign = np.sign(np.dot(loadings[:, j], eigvecs[:, j])) or 1.0
        diff = float(np.max(np.abs(loadings[:, j] - sign * eigvecs[:, j])))
        assert diff < 1e-10, f"loadings column {j}: diff={diff:.2e}"

    assert np.allclose(explained, eigvals[:r] / np.sum(eigvals), atol=1e-10)
    assert np.allclose(scores, xc @ loadings, atol=1e-10)


def test_radf_common_rejects_single_series():
    x = np.cumsum(np.random.default_rng(0).normal(size=(50, 1)), axis=0)
    with pytest.raises(ValueError):
        radf_common(x)


def test_radf_common_shapes_and_loadings():
    rng = np.random.default_rng(2)
    n, nc = 80, 4
    panel = np.cumsum(rng.normal(size=(n, nc)), axis=0)
    minw = psy_minw(n)

    res = radf_common(panel, minw=minw, r=2)

    assert isinstance(res, RadfCommonResult)
    assert res.adf.shape == (1,)
    assert res.bsadf.shape == (n - minw, 1)
    assert res.loadings.shape == (nc, 2)
    assert res.explained_variance_ratio.shape == (2,)
    # Columns of loadings are orthonormal (PCA property, independent of radf()).
    np.testing.assert_allclose(res.loadings.T @ res.loadings, np.eye(2), atol=1e-10)


def test_radf_common_cv_shape_and_n_field():
    cv = radf_common_cv(n=50, N=3, nrep=40, seed=5)

    assert isinstance(cv, RadfCommonCv)
    assert cv.N == 3
    assert cv.method == "Monte Carlo"
    assert cv.gsadf_cv.shape == (3,)
    n_minw = 50 - cv.minw
    assert cv.bsadf_cv.shape == (n_minw, 3)
    for arr in (cv.adf_cv, cv.sadf_cv, cv.gsadf_cv):
        assert np.all(np.diff(arr) >= 0)


def test_radf_common_cv_grows_with_n():
    """The independent-validation finding docs/multivariate.md documents:
    Theorem 4.3's claim that the null is asymptotically independent of
    panel width N does NOT hold at practical N -- the true null quantile
    *grows* with N. R reference numbers (n=100, nrep=300, seed=42):
    N=4 95% gsadf_cv=2.39, N=20 95% gsadf_cv=3.48 (see
    docs/replication/multivariate/radf_common_validation.py)."""
    cv_small = radf_common_cv(n=60, N=4, nrep=150, seed=1)
    cv_large = radf_common_cv(n=60, N=20, nrep=150, seed=1)
    assert cv_large.gsadf_cv[1] > cv_small.gsadf_cv[1]


def test_radf_common_cv_not_same_as_plain_mc_cv():
    """The whole point of radf_common_cv(): it must NOT collapse to
    radf_mc_cv() -- see docs/multivariate.md's independent-validation
    finding this file quotes above."""
    n, minw = 80, psy_minw(80)
    common_cv = radf_common_cv(n=n, N=15, minw=minw, nrep=80, seed=7)
    plain_cv = radf_mc_cv(n=n, minw=minw, nrep=80, seed=7)

    assert common_cv.gsadf_cv[1] > plain_cv.gsadf_cv[1]


# -- cobubble_test() ---------------------------------------------------------


def test_coexplosive_stat_formula_exact():
    """_coexplosive_stat() vs. an independent brute-force computation
    (separate lstsq call, manual loop-based cumulative sum) -- mirrors
    docs/replication/multivariate/radf_cobubble_validation.py's check 1."""
    rng = np.random.default_rng(1)
    tn = 100
    x = rng.normal(size=tn)
    y = 2 + 0.5 * x + rng.normal(size=tn)
    lag = 2

    S, _resid, _sigma2, _n = _coexplosive_stat(y, x, lag)

    lo, hi = max(lag, 0), tn + min(lag, 0)
    yy, xx = y[lo:hi], x[lo - lag : hi - lag]
    design = np.column_stack([np.ones(len(yy)), xx])
    beta, *_ = np.linalg.lstsq(design, yy, rcond=None)
    e_brute = yy - design @ beta
    n_brute = len(e_brute)
    sigma2_brute = np.sum(e_brute**2) / n_brute
    running = 0.0
    s_manual_sum = 0.0
    for e in e_brute:
        running += e
        s_manual_sum += running**2
    s_brute = s_manual_sum / (sigma2_brute * n_brute**2)

    assert abs(S - s_brute) < 1e-8


def test_cobubble_test_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        cobubble_test(np.zeros(10), np.zeros(9))


def test_cobubble_test_lag_recovery():
    """True lag baked into the DGP is recovered by lag selection --
    smaller/faster version of the validation script's check 5."""
    true_lag = 3
    for seed in range(1, 6):
        rng = np.random.default_rng(seed)
        tn, te = 150, 90
        ex = np.cumsum(rng.normal(size=te))
        expl = ex[-1] * 1.06 ** np.arange(1, tn - te + 1) + np.cumsum(
            rng.normal(scale=0.3, size=tn - te)
        )
        x = np.concatenate([ex, expl])
        y = np.full(tn, np.nan)
        for t in range(true_lag, tn):
            y[t] = 1 + 0.8 * x[t - true_lag] + rng.normal(scale=0.5)
        y[:true_lag] = x[:true_lag] + rng.normal(scale=0.5, size=true_lag)
        assert _coexplosive_select_lag(y, x, range(-6, 7)) == true_lag


def test_cobubble_test_rejects_independent_bubbles():
    """Power check: y and x have independent explosive episodes (not
    co-explosive) -- should reject co-explosivity."""
    rng = np.random.default_rng(42)
    tn, te = 150, 90
    ex = np.cumsum(rng.normal(size=te))
    expl_x = ex[-1] * 1.05 ** np.arange(1, tn - te + 1) + np.cumsum(
        rng.normal(scale=0.3, size=tn - te)
    )
    x = np.concatenate([ex, expl_x])
    ey = np.cumsum(rng.normal(size=te))
    expl_y = ey[-1] * 1.05 ** np.arange(1, tn - te + 1) + np.cumsum(
        rng.normal(scale=0.3, size=tn - te)
    )
    y = np.concatenate([ey, expl_y])

    out = cobubble_test(y, x, lag=0, nboot=199, seed=1)
    assert out.reject


def test_cobubble_test_not_reject_co_explosive_pair():
    """A genuinely co-explosive pair (y driven by x contemporaneously plus
    iid noise): co-explosivity should typically not be rejected."""
    rng = np.random.default_rng(7)
    tn, te = 150, 90
    ex = np.cumsum(rng.normal(size=te))
    expl = ex[-1] * 1.05 ** np.arange(1, tn - te + 1) + np.cumsum(
        rng.normal(scale=0.3, size=tn - te)
    )
    x = np.concatenate([ex, expl])
    y = 1 + 0.8 * x + rng.normal(size=tn)

    out = cobubble_test(y, x, lag=0, nboot=199, seed=1)
    assert not out.reject


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
