"""The 2026-08 DGP extensions ported from exuber's R/sim.R. numpy's RNG is
not R's, so draws can't be compared; these pin the deterministic parts to
R's own output and check each process's defining property instead."""

import numpy as np
import pytest

from exuber.sim import (
    sim_coexplosive,
    sim_common,
    sim_falsebubble,
    sim_mar,
    sim_msbubble,
    sim_tree,
)


@pytest.mark.parametrize(
    ("shape", "tau_ref", "offset_ref"),
    [
        # R: p <- sim_falsebubble(40, shape=, seed=1);
        #    attr(p, "technology")[c(12,15,18,20,25,28,29)];
        #    (p - attr(p, "dividend") / 0.05)[c(1,10,20,30,40)]
        ("triangular", [0, 0.375, 0.75, 1, 0.375, 0, 0],
         [72.51267876, 107.85980756, 69.08133013, 8.4, 8.4]),
        ("gaussian", [0.1353352832, 0.4578333618, 0.8824969026, 1, 0.4578333618, 0.1353352832, 0],
         [86.32969625, 129.29453665, 82.81728064, 8.4, 8.4]),
    ],
)
def test_sim_falsebubble_deterministic_parts_match_r(shape, tau_ref, offset_ref):
    p, d, tau = sim_falsebubble(40, shape=shape, seed=1, return_components=True)
    np.testing.assert_allclose(tau[[11, 14, 17, 19, 24, 27, 28]], tau_ref, atol=1e-9)
    # price minus the dividend term is deterministic: drift constant + discounted hump
    np.testing.assert_allclose((p - d / 0.05)[[0, 9, 19, 29, 39]], offset_ref, atol=1e-7)


def test_sim_tree_respects_price_floor():
    y = sim_tree(2000, seed=3)
    assert y.shape == (2000,) and np.all(np.isfinite(y))
    assert y.min() >= 1 / (1 - 0.95) - 1e-9  # Corollary 1: Y_t >= eta / (1 - a)


def test_sim_mar_shape_and_t_innovations():
    y = sim_mar(300, dist="t", df=5, seed=1)
    assert y.shape == (300,) and np.all(np.isfinite(y))
    with pytest.raises(ValueError):
        sim_mar(10, dist="normal")


def test_sim_msbubble_regime_shares():
    with np.errstate(over="ignore"):  # b itself diverges this long -- see docstring
        b, s = sim_msbubble(200_000, seed=2, return_regime=True)
    assert set(np.unique(s)) == {1, 2}
    # stationary P(S=1) = (1 - p22) / (2 - p11 - p22) = 0.1 / 0.12
    assert abs((s == 1).mean() - 0.1 / 0.12) < 0.02
    assert b.shape == s.shape


def test_sim_coexplosive_lag_is_exact_without_noise():
    xy = sim_coexplosive(100, lag=5, sigma_y=0, seed=4)
    x, y = xy[:, 0], xy[:, 1]
    assert np.all(np.isnan(y[:5]))
    np.testing.assert_allclose(y[5:], x[:-5])
    lead = sim_coexplosive(100, lag=-3, sigma_y=0, seed=4)
    assert np.all(np.isnan(lead[-3:, 1]))


def test_sim_common_is_factor_plus_noise():
    x, f = sim_common(5, 100, sigma_e=0, seed=5, return_factor=True)
    assert x.shape == (100, 5)
    loadings = x[0] / f[0]
    assert np.all((loadings >= 0) & (loadings <= 2))
    np.testing.assert_allclose(x, np.outer(f, loadings))
