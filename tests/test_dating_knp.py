"""Unit tests for dating_knp.py, ported from exuber's test-dating_knp.R."""

import math

import numpy as np
import pytest

from exuber.dating_knp import _knp_dp, _knp_find_break, dating_knp

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


# -- KNP's multi-bubble dynamic programme (Section 3.2) -------------------


def _knp_brute(y: np.ndarray, breaks: int, trim: float, omit: bool) -> tuple[list[int], float]:
    n1 = len(y) - 1
    x, z = y[:n1], np.diff(y)
    k_min = max(2, math.ceil(trim * n1))

    def seg(j: int, lo: int, hi: int) -> float:
        if j % 2 == 0:
            return _ols_ssr(x[lo:hi], z[lo:hi])
        return float(np.sum(z[lo:hi] ** 2) - (z[lo] ** 2 if omit and j > 1 else 0.0))

    best: tuple[list[int], float] = ([], math.inf)

    def rec(taus: list[int]) -> None:
        nonlocal best
        if len(taus) == breaks:
            ends = [0, *taus, n1]
            ssr = sum(seg(r + 1, ends[r], ends[r + 1]) for r in range(breaks + 1))
            if ssr < best[1]:
                best = (taus, ssr)
            return
        start = (taus[-1] if taus else 0) + k_min
        for t in range(start, n1 - (breaks - len(taus)) * k_min + 1):
            rec([*taus, t])

    rec([])
    return best


def test_knp_dp_two_breaks_reproduces_single_bubble_search():
    y = np.cumsum(np.random.default_rng(11).normal(size=80))
    for omit in (True, False):
        tau, ssr = _knp_dp(y, 2, 0.05, omit)
        t1, t2, fssr = _knp_find_break(y, 0.05, omit)
        assert tau == [t1, t2]
        assert ssr == pytest.approx(fssr, abs=1e-10)


@pytest.mark.parametrize("breaks", [3, 4])
@pytest.mark.parametrize("omit", [True, False])
def test_knp_dp_matches_brute_force(breaks, omit):
    y = np.cumsum(np.random.default_rng(12).normal(size=28))
    tau, ssr = _knp_dp(y, breaks, 0.1, omit)
    b_tau, b_ssr = _knp_brute(y, breaks, 0.1, omit)
    assert tau == b_tau
    assert ssr == pytest.approx(b_ssr, abs=1e-8)


def test_dating_knp_multi_bubble_matches_r():
    from test_ssu_test import Y_VEC

    r3 = dating_knp(Y_VEC, trim=0.1, breaks=3)
    np.testing.assert_array_equal(r3.origination[:, 0], [9, 19])
    assert r3.collapse[0, 0] == 14 and np.isnan(r3.collapse[1, 0])
    np.testing.assert_allclose(r3.delta[:, 0], [0.8410590945, 0.5993879191], atol=1e-8)
    r4 = dating_knp(Y_VEC, trim=0.1, breaks=4)
    np.testing.assert_array_equal(r4.collapse[:, 0], [14, 23])
    np.testing.assert_allclose(r4.delta[:, 0], [0.8410590945, 1.0818531838], atol=1e-8)
    assert dating_knp(Y_VEC).origination.shape == (1,)
