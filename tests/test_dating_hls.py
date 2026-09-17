"""Unit tests for dating_hls.py (and the shared _hls_common.py helpers it
exercises), ported from exuber's test-dating_hls.R."""

import math

import numpy as np
import pytest

from exuber._hls_common import (
    _hls_fit_series,
    _hls_model1,
    _hls_model4,
    _hls_model23,
    _hls_prefix_sums,
    _hls_segment_ssr,
)
from exuber.dating_hls import dating_hls

# -- dating_hls() -----------------------------------------------------------


def _ols_ssr(xseg: np.ndarray, zseg: np.ndarray) -> float:
    a = np.vstack([xseg, np.ones_like(xseg)]).T
    coef, *_ = np.linalg.lstsq(a, zseg, rcond=None)
    return float(np.sum((zseg - a @ coef) ** 2))


def test_hls_segment_ssr_matches_brute_force_ols():
    rng = np.random.default_rng(1)
    y = np.cumsum(rng.normal(size=40))
    ps = _hls_prefix_sums(y)
    n1 = len(y) - 1
    x_all, z_all = y[:n1], np.diff(y)

    for lo, hi in [(0, 10), (10, 25), (4, 39)]:
        manual = _hls_segment_ssr(ps, lo, hi, True)
        brute = _ols_ssr(x_all[lo:hi], z_all[lo:hi])
        assert manual == pytest.approx(brute, abs=1e-8)

    assert _hls_segment_ssr(ps, 0, 10, False) == pytest.approx(np.sum(z_all[:10] ** 2), abs=1e-8)


def test_hls_model1_matches_brute_force_grid_search():
    rng = np.random.default_rng(2)
    n = 30
    y = np.cumsum(rng.normal(size=n))
    ps = _hls_prefix_sums(y)
    tau1, ssr = _hls_model1(y, ps, trim=0.1)

    n1 = n - 1
    x, z = y[:n1], np.diff(y)
    k_min = max(2, math.ceil(0.1 * n1))
    best_ssr, best_tau1 = math.inf, None
    for t1 in range(k_min, n1 - k_min + 1):
        if y[n1] <= y[t1]:
            continue
        s = np.sum(z[:t1] ** 2) + _ols_ssr(x[t1:n1], z[t1:n1])
        if s < best_ssr:
            best_ssr, best_tau1 = s, t1

    assert tau1 == best_tau1
    assert ssr == pytest.approx(best_ssr, abs=1e-6)


def test_hls_model23_matches_brute_force_grid_search():
    rng = np.random.default_rng(3)
    n = 26
    y = np.cumsum(rng.normal(size=n))
    ps = _hls_prefix_sums(y)
    n1 = n - 1
    x, z = y[:n1], np.diff(y)
    k_min = max(2, math.ceil(0.1 * n1))

    def brute(right_fit):
        best_ssr, best = math.inf, None
        for t1 in range(k_min, n1 - 2 * k_min + 1):
            for t2 in range(t1 + k_min, n1 - k_min + 1):
                if y[t2] <= y[t1]:
                    continue
                if right_fit and y[t2] <= y[n1]:
                    continue
                ssr_mid = _ols_ssr(x[t1:t2], z[t1:t2])
                ssr_right = _ols_ssr(x[t2:n1], z[t2:n1]) if right_fit else np.sum(z[t2:n1] ** 2)
                s = np.sum(z[:t1] ** 2) + ssr_mid + ssr_right
                if s < best_ssr:
                    best_ssr, best = s, (t1, t2)
        return best, best_ssr

    for right_fit in (False, True):
        tau1, tau2, ssr = _hls_model23(y, ps, trim=0.1, right_fit=right_fit)
        (b_tau1, b_tau2), b_ssr = brute(right_fit)
        assert (tau1, tau2) == (b_tau1, b_tau2)
        assert ssr == pytest.approx(b_ssr, abs=1e-6)


def test_hls_model4_matches_brute_force_joint_grid_search():
    rng = np.random.default_rng(5)
    n = 24
    y = np.cumsum(rng.normal(size=n))
    ps = _hls_prefix_sums(y)
    tau1, tau2, tau3, ssr = _hls_model4(y, ps, trim=0.1)

    n1 = n - 1
    x, z = y[:n1], np.diff(y)
    k_min = max(2, math.ceil(0.1 * n1))
    best_ssr, best = math.inf, None
    for t1 in range(k_min, n1 - 3 * k_min + 1):
        for t2 in range(t1 + k_min, n1 - 2 * k_min + 1):
            if y[t2] <= y[t1]:
                continue
            for t3 in range(t2 + k_min, n1 - k_min + 1):
                if y[t2] <= y[t3]:
                    continue
                s = (
                    np.sum(z[:t1] ** 2)
                    + _ols_ssr(x[t1:t2], z[t1:t2])
                    + _ols_ssr(x[t2:t3], z[t2:t3])
                    + np.sum(z[t3:n1] ** 2)
                )
                if s < best_ssr:
                    best_ssr, best = s, (t1, t2, t3)

    assert (tau1, tau2, tau3) == best
    assert ssr == pytest.approx(best_ssr, abs=1e-6)


def test_hls_fit_series_restricted_to_models_2_and_4():
    # dating_hlw() restricts non-final windows to models {2, 4} -- verify
    # the shared helper actually honors a restricted model set.
    rng = np.random.default_rng(1)
    y = np.cumsum(rng.normal(size=80))
    fit = _hls_fit_series(y, trim=0.1, models=(2, 4))
    assert fit.model in (2, 4)
    assert np.isinf(fit.bic[0])  # model 1 not requested
    assert np.isinf(fit.bic[2])  # model 3 not requested


def test_dating_hls_end_to_end_well_formed():
    rng = np.random.default_rng(1)
    y = np.cumsum(rng.normal(size=100))
    out = dating_hls(y, trim=0.05)
    assert out.model[0] in (1, 2, 3, 4)
    assert out.bic.shape == (1, 4)
    assert not math.isnan(out.origination[0])


def test_dating_hls_na_pattern_matches_selected_model():
    rng = np.random.default_rng(1)
    y = np.cumsum(rng.normal(size=100))
    out = dating_hls(y, trim=0.05)
    m = out.model[0]
    if m == 1:
        assert math.isnan(out.collapse[0])
        assert math.isnan(out.recovery[0])
    elif m in (2, 3):
        assert math.isnan(out.recovery[0])
    assert not math.isnan(out.origination[0])


def test_dating_hls_multivariate_panel():
    rng = np.random.default_rng(1)
    y = np.cumsum(rng.normal(size=80))
    panel = np.column_stack([y, y + rng.normal(scale=0.1, size=len(y))])
    out = dating_hls(panel, trim=0.05)
    assert out.model.shape == (2,)
    assert out.bic.shape == (2, 4)


def _sim_hls_model4(seed, n1=60, n2=25, n3=25, n4=40, base=100.0, c_bubble=1.05):
    rng = np.random.default_rng(seed)
    unit1 = base + np.cumsum(rng.normal(size=n1))
    bubble = unit1[-1] * c_bubble ** np.arange(1, n2 + 1) + np.cumsum(rng.normal(size=n2))
    target = bubble[-1] * 0.5
    collapse = np.empty(n3)
    collapse[0] = bubble[-1] + rng.normal()
    for k in range(1, n3):
        collapse[k] = target + 0.85 * (collapse[k - 1] - target) + rng.normal()
    recovery = collapse[-1] + np.cumsum(rng.normal(size=n4))
    y = np.concatenate([unit1, bubble, collapse, recovery])
    return y, n1, n1 + n2


def test_dating_hls_recovers_model4_dgp_and_prefers_complex_models():
    results = []
    for seed in range(15):
        y, true_tau1, _true_tau2 = _sim_hls_model4(seed)
        out = dating_hls(y, trim=0.05)
        results.append((out.model[0], out.origination[0] - true_tau1))

    models = [m for m, _ in results]
    bias = [abs(b) for _, b in results]
    assert np.mean([m in (3, 4) for m in models]) > 0.5
    assert np.mean(bias) < 15


def test_dating_hls_h0_does_not_spuriously_prefer_model4():
    models = []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        y = 100 + np.cumsum(rng.normal(size=150))
        out = dating_hls(y, trim=0.05)
        models.append(out.model[0])
    assert np.mean([m == 4 for m in models]) < 0.3
