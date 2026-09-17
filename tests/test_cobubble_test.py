"""Tests for exuber.cobubble_test. Several checks mirror docs/replication/
multivariate/radf_cobubble_validation.py (same checks, re-implemented
here rather than imported -- that script lives in the umbrella repo, one
level up from pyexuber's own repo boundary, so pyexuber's CI (which only
checks out `kvasilopoulos/pyexuber`) can't reach it)."""

import numpy as np
import pytest

from exuber.cobubble_test import _coexplosive_select_lag, _coexplosive_stat, cobubble_test

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
