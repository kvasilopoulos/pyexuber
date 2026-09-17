"""Tests for exuber.ssu_test -- the stochastic explosive-coefficient
(SSU) test. Port of exuber's R/ssu_test.R; see
docs/volatility-robustness.md (root repo) for the source paper.

Formula-exact reference numbers below were produced by feeding the SAME
deterministic input series to both R and this port (no RNG involved at
the formula level, so results agree to numpy/R floating-point precision,
not just approximately) -- see
docs/replication/volatility-robustness/radf_ssu_validation.py (root
repo) for the fuller narrative version and the exact R commands used.
"""

import numpy as np
import pytest

from exuber.radf import psy_minw
from exuber.ssu_test import ssu_prefix_sums, ssu_q, ssu_stat_path, ssu_test

# set.seed(7); y <- round(cumsum(rnorm(40)), 8)
Y_VEC = np.array(
    [
        2.28724716, 1.09047548, 0.39618297, -0.01610998, -0.98678332, -1.93406327,
        -1.18592393, -1.30287915, -1.15022153, 1.03975658, 1.39674281, 4.11349459,
        6.39494652, 6.71896706, 8.61503413, 9.08271464, 8.18891391, 7.88158561,
        7.87676319, 8.86492734, 9.7046777, 10.41001953, 11.71598425, 10.32798804,
        11.6009049, 11.78509767, 12.53737757, 13.12912262, 12.14607002, 11.87000607,
        10.99915505, 11.7178656, 11.82851848, 11.75005171, 11.32956125, 10.76743537,
        11.76494882, 10.65981876, 10.51753093, 10.83252583,
    ]
)
MINW = 10

R_SSU_STAT_LAST3 = np.array([-1.4294408532, -1.4956945921, -1.5827171047])
R_SSU_SADF = 0.8058323864


def test_ssu_q_lookup():
    assert ssu_q(90) == pytest.approx(2.90)
    assert ssu_q(95) == pytest.approx(3.30)
    assert ssu_q(99) == pytest.approx(4.20)
    with pytest.raises(ValueError):
        ssu_q(80)


def test_ssu_test_matches_r():
    res = ssu_test(Y_VEC, minw=MINW, sig_lvl=95)
    np.testing.assert_allclose(res.stat[-3:, 0], R_SSU_STAT_LAST3, atol=1e-4)
    assert res.sadf[0] == pytest.approx(R_SSU_SADF, abs=1e-4)
    assert res.crit == pytest.approx(3.30)
    assert bool(res.detected[0]) == (R_SSU_SADF > 3.30)


def test_ssu_test_default_minw_matches_psy_minw():
    y = np.cumsum(np.random.default_rng(1).normal(size=120))
    res = ssu_test(y)
    assert res.minw == psy_minw(120)


def test_ssu_test_formula_matches_brute_force():
    """ssu_stat_path()'s bilinear cross-moment expansion vs. a brute-force
    per-window computation (two separately fitted OLS regressions plus a
    manual residual cross-moment), the same check test-ssu.R's own R
    validation uses."""
    rng = np.random.default_rng(2)
    n = 150
    y = np.cumsum(rng.normal(size=n))
    ps = ssu_prefix_sums(y)

    def brute_force_stat(hi: int) -> float:
        win = np.arange(hi)
        x1 = y[win]
        d1 = y[win + 1] - x1
        x2 = x1**2
        d2 = d1**2

        A1 = np.column_stack([np.ones(hi), x1])
        beta1, *_ = np.linalg.lstsq(A1, d1, rcond=None)
        eps_hat = d1 - A1 @ beta1

        A2 = np.column_stack([np.ones(hi), x2])
        beta2, *_ = np.linalg.lstsq(A2, d2, rcond=None)
        eta_hat = d2 - A2 @ beta2

        sigma2_eps = np.sum(eps_hat**2) / (hi - 2)
        sigma2_eta = np.sum(eta_hat**2) / (hi - 2)
        sigma2_epseta = np.sum(eps_hat * eta_hat) / (hi - 1)
        sigma_eps = np.sqrt(sigma2_eps)
        sigma_eta = np.sqrt(sigma2_eta)
        psi_hat = sigma2_epseta / (sigma_eps * sigma_eta)

        omega_hat = beta2[1]
        sxx2_c = np.sum((x2 - x2.mean()) ** 2)
        t_omega = omega_hat / np.sqrt(sigma2_eta / sxx2_c)

        num_corr = np.sum((x2 - x2.mean()) * d1)
        den_corr = np.sqrt(sxx2_c)
        correction = (psi_hat / sigma_eps) * num_corr / den_corr
        return (t_omega - correction) / np.sqrt(1 - psi_hat**2)

    for hi in (50, 80, 120, 149):
        fast = ssu_stat_path(ps, np.array([hi]))[0]
        manual = brute_force_stat(hi)
        assert fast == pytest.approx(manual, abs=1e-6)


def test_ssu_test_power_on_stochastic_coefficient_dgp():
    """SSU should have decent detection power on the alternative it's
    actually designed for (stochastic, not deterministic, explosive
    coefficient) -- Kurozumi & Nishi's own eq. 2 style DGP."""

    def make_stochastic_bubble(rng: np.random.Generator, n: int, te_frac: float = 0.5,
                                c1: float = 3.0, a: float = 4.0) -> np.ndarray:
        y = np.empty(n)
        y[0] = rng.normal()
        te = round(te_frac * n)
        for t in range(1, n):
            if t < te:
                y[t] = y[t - 1] + rng.normal()
            else:
                rho_t = 1 + c1 / n + a * rng.normal() / np.sqrt(n)
                y[t] = rho_t * y[t - 1] + rng.normal()
        return y

    rng = np.random.default_rng(2)
    n, nrep = 200, 30
    detections = sum(
        ssu_test(make_stochastic_bubble(rng, n), sig_lvl=95).detected[0] for _ in range(nrep)
    )
    power = detections / nrep
    assert power > 0.4  # KN's own MC reports ~85%; a loose lower bound for a small smoke test
