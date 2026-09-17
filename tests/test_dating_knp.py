"""Unit tests for dating_knp.py, ported from exuber's test-dating_knp.R."""

import math

import numpy as np
import pytest

from exuber.dating_knp import _knp_find_break, dating_knp

# -- dating_knp() -------------------------------------------------------


def _ols_ssr(xseg: np.ndarray, zseg: np.ndarray) -> float:
    a = np.vstack([xseg, np.ones_like(xseg)]).T
    coef, *_ = np.linalg.lstsq(a, zseg, rcond=None)
    return float(np.sum((zseg - a @ coef) ** 2))


def test_knp_find_break_omit_false_matches_brute_force():
    rng = np.random.default_rng(3)
    n = 26
    y = np.cumsum(rng.normal(size=n))
    tau1, tau2, ssr = _knp_find_break(y, trim=0.1, omit=False)

    n1 = n - 1
    x, z = y[:n1], np.diff(y)
    k_min = max(2, math.ceil(0.1 * n1))
    best_ssr, best = math.inf, None
    for t1 in range(k_min, n1 - 2 * k_min + 1):
        for t2 in range(t1 + k_min, n1 - k_min + 1):
            s = np.sum(z[:t1] ** 2) + _ols_ssr(x[t1:t2], z[t1:t2]) + np.sum(z[t2:n1] ** 2)
            if s < best_ssr:
                best_ssr, best = s, (t1, t2)

    assert (tau1, tau2) == best
    assert ssr == pytest.approx(best_ssr, abs=1e-6)


def test_knp_find_break_omit_true_matches_brute_force_with_residual_dropped():
    rng = np.random.default_rng(3)
    n = 26
    y = np.cumsum(rng.normal(size=n))
    tau1, tau2, ssr = _knp_find_break(y, trim=0.1, omit=True)

    n1 = n - 1
    x, z = y[:n1], np.diff(y)
    k_min = max(2, math.ceil(0.1 * n1))
    best_ssr, best = math.inf, None
    for t1 in range(k_min, n1 - 2 * k_min + 1):
        for t2 in range(t1 + k_min, n1 - k_min + 1):
            s = (
                np.sum(z[:t1] ** 2) + _ols_ssr(x[t1:t2], z[t1:t2]) + np.sum(z[t2:n1] ** 2)
                - z[t2] ** 2
            )
            if s < best_ssr:
                best_ssr, best = s, (t1, t2)

    assert (tau1, tau2) == best
    assert ssr == pytest.approx(best_ssr, abs=1e-6)


def test_dating_knp_end_to_end_well_formed():
    rng = np.random.default_rng(1)
    y = np.cumsum(rng.normal(size=150))
    out = dating_knp(y, trim=0.05)
    assert not math.isnan(out.origination[0])
    assert not math.isnan(out.collapse[0])
    assert not math.isnan(out.delta[0])


def _sim_knp(seed, t1=50, t2=90, t=200, delta=1.05):
    rng = np.random.default_rng(seed)
    y = np.zeros(t)
    for i in range(1, t1):
        y[i] = y[i - 1] + rng.normal()
    for i in range(t1, t2):
        y[i] = delta * y[i - 1] + rng.normal()
    y[t2] = y[t1 - 1] + rng.normal()
    for i in range(t2 + 1, t):
        y[i] = y[i - 1] + rng.normal()
    return y, t1, t2


def test_dating_knp_omission_correction_reproduces_theorem_1_and_2():
    # Theorem 1: naive (omit=False) tau1 tracks the true COLLAPSE date,
    # not the origination date. Theorem 2: the omission correction
    # substantially reduces the origination-date bias.
    def run(seed, omit):
        y, t1, t2 = _sim_knp(seed)
        tau1, _tau2, _ssr = _knp_find_break(y, trim=0.05, omit=omit)
        return tau1, t1, t2

    res_naive = [run(s, False) for s in range(20)]
    res_om = [run(s, True) for s in range(20)]

    bias_naive_t1 = np.mean([abs(tau1 - t1) for tau1, t1, _t2 in res_naive])
    bias_naive_t2 = np.mean([abs(tau1 - t2) for tau1, _t1, t2 in res_naive])
    bias_om_t1 = np.mean([abs(tau1 - t1) for tau1, t1, _t2 in res_om])

    assert bias_naive_t2 < bias_naive_t1
    assert bias_om_t1 < bias_naive_t1 / 2
