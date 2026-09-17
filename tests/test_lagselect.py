"""_lagselect.py is deterministic (no RNG) -- checked bit-for-bit against
R here. Reference numbers from
docs/replication/volatility-robustness/radf_wb_ps_validation.R (seed 11,
n=40) in the root exuber-project repo; duplicated as literals so this
test doesn't depend on that repo's layout."""

import numpy as np

from exuber._lagselect import adf_res, lag_select, unroot_adf, unroot_adf_null

# y <- cumsum(rnorm(40)) after set.seed(11) in R.
Y = np.array(
    [
        -0.5910311026, -0.5644367336, -2.0809898307, -3.4436431799, -2.2651540239,
        -3.1993053436, -1.8756996974, -1.2507819074, -1.2965048631, -2.3006254388,
        -3.1290586755, -3.4774104006, -5.0157037977, -5.2712690432, -6.4212140759,
        -6.4088871082, -6.6318566490, -5.7440850011, -6.3362402809, -6.9919583995,
        -7.6744760217, -7.6903342145, -8.1329389998, -7.7803815004, -7.7072109181,
        -7.7000521177, -7.8876522283, -8.6533528738, -8.8744096946, -9.8579982820,
        -10.9622823242, -11.9004325384, -11.2218082942, -12.7993061595, -13.6692446177,
        -13.1845675724, -13.3706202710, -11.8250655708, -12.4364456409, -12.7842021283,
    ]
)


def test_lag_select_matches_r():
    assert lag_select(Y, "aic", max_lag=5) == 5
    assert lag_select(Y, "bic", max_lag=5) == 4


def test_lag_select_max_lag_zero_is_a_noop():
    assert lag_select(Y, "aic", max_lag=0) == 0


def test_adf_res_matches_r():
    fit = adf_res(Y, adflag=2, type="fixed")
    np.testing.assert_allclose(fit.beta, [-0.2766891117, -0.0510771847, 0.0954335649], atol=1e-8)
    np.testing.assert_allclose(
        fit.res[:5],
        [-1.1659634957, 1.5303078394, -0.4672254328, 1.4401135170, 1.0583623423],
        atol=1e-8,
    )
    assert len(fit.res) == 37


def test_adf_res_type_criterion_resolves_adflag_via_lag_select():
    # adflag=5 is then max_lag for lag_select("bic", ...) == 4 (see above)
    fit_auto = adf_res(Y, adflag=5, type="bic")
    fit_fixed = adf_res(Y, adflag=4, type="fixed")
    np.testing.assert_array_equal(fit_auto.beta, fit_fixed.beta)


def test_unroot_adf_shapes():
    # nrow = len(x) - 1 - lag for both builders (one row lost to diff(),
    # `lag` more to the embed() truncation).
    yxmat = unroot_adf(Y, lag=2)
    assert yxmat.shape == (len(Y) - 3, 5)  # dy, ct, y, dy_lags1, dy_lags2
    yxmat0 = unroot_adf(Y, lag=0)
    assert yxmat0.shape == (len(Y) - 1, 3)  # dy, ct, y


def test_unroot_adf_null_shapes():
    yxmat = unroot_adf_null(Y, lag=2)
    assert yxmat.shape == (len(Y) - 3, 4)  # dy, ct, dy_lags1, dy_lags2
    yxmat0 = unroot_adf_null(Y, lag=0)
    assert yxmat0.shape == (len(Y) - 1, 2)  # dy, ct
