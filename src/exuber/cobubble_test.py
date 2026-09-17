"""Co-bubble test (Evripidou, Harvey, Leybourne & Sollis 2022). Ported
from exuber's R/cobubble_test.R -- see docs/multivariate.md for the full
evaluation this implements.

Tests whether two series that each contain an explosive episode are
"co-explosive": whether y_t - alpha - beta*x_{t-lag} is I(0) for some
lead/lag, i.e. a KPSS-type stationarity test (null = co-explosive) on
the residuals of y_t regressed on a constant and x_{t-lag} -- the
opposite testing direction from radf()'s right-tailed unit-root tests.
Critical values come from a wild bootstrap that reproduces the
residuals' own heteroskedasticity pattern (the null distribution
depends on it, so there's no fixed table -- Theorem 1/2).
"""

from dataclasses import dataclass

import numpy as np


def _coexplosive_stat_aligned(
    y: np.ndarray, xreg: np.ndarray
) -> tuple[float, np.ndarray, float, int]:
    """KPSS-type statistic S (eq. 3) on two already-aligned, equal-length
    vectors: sigma_y^-2 * n^-2 * sum_t (cumsum of OLS residuals up to t)^2,
    the OLS regression being y on a constant and xreg. Returns (S, resid,
    sigma2, n) -- resid/sigma2/n are reused by the wild bootstrap, which
    regresses a bootstrap y* on the SAME xreg (Remark 2: omitting xreg
    from the bootstrap regression gives a worse finite-sample match)."""
    n = len(y)
    if n < 3:
        raise ValueError("Series too short.")
    design = np.column_stack([np.ones(n), xreg])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    e = y - design @ beta
    sigma2 = float(np.mean(e**2))
    cs = np.cumsum(e)
    S = float(np.sum(cs**2) / (sigma2 * n**2))
    return S, e, sigma2, n


def _coexplosive_stat(
    y: np.ndarray, x: np.ndarray, lag: int = 0
) -> tuple[float, np.ndarray, float, int]:
    """Raw-index entry point: builds the (y_t, x_{t-lag}) pair over the
    overlapping valid range implied by `lag` and calls the aligned core."""
    tn = len(y)
    lo = max(lag, 0)
    hi = tn + min(lag, 0)
    idx_y = np.arange(lo, hi)
    idx_x = idx_y - lag
    return _coexplosive_stat_aligned(y[idx_y], x[idx_x])


def _coexplosive_select_lag(y: np.ndarray, x: np.ndarray, lags) -> int:
    """Section VI's i_hat = argmin_j sigma2_hat(j): a misspecified lag
    leaves a neglected explosive term in the residuals that inflates
    their variance, so the variance-minimizing lag recovers the true one."""
    lags = list(lags)
    sigma2s = [_coexplosive_stat(y, x, j)[2] for j in lags]
    return lags[int(np.argmin(sigma2s))]


@dataclass
class CobubbleTestResult:
    S: float
    lag: int
    cv: float
    p_value: float
    reject: bool
    sig_lvl: int
    nboot: int


def cobubble_test(
    y,
    x,
    lag: int | None = None,
    lag_grid=range(-6, 7),
    nboot: int = 499,
    sig_lvl: int = 95,
    seed: int | None = None,
) -> CobubbleTestResult:
    """Test for co-explosive behaviour between two series (Evripidou,
    Harvey, Leybourne & Sollis 2022).

    Tests whether y_t - alpha - beta*x_{t-lag} is stationary, i.e.
    whether the explosive dynamics in y and x are the same underlying
    phenomenon (possibly migrating between the two series with a lead or
    lag) rather than independent explosive episodes. Unlike radf() (a
    right-tailed test for explosiveness), this is a stationarity
    (KPSS-type) test: the null is co-explosivity. Critical values are a
    wild bootstrap of the residuals (there's no fixed table -- the null
    distribution depends on the residuals' heteroskedasticity pattern).

    y, x: equal-length sequences. x is the (candidate) explosive-episode
    regressor; y is tested for co-explosivity with x_{t-lag}.
    lag: the lead/lag i in x_{t-lag}. If None, estimated from lag_grid by
    minimizing the residual variance (Section VI's i_hat).
    lag_grid: candidate lags searched when lag is None.
    nboot: number of wild bootstrap replications.
    sig_lvl: one of 90, 95, 99 -- same convention as datestamp()'s sig_lvl.
    seed: optional seed for the bootstrap draws.
    """
    if sig_lvl not in (90, 95, 99):
        raise ValueError("sig_lvl must be one of 90, 95, 99")
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if len(y) != len(x):
        raise ValueError("'y' and 'x' must be the same length.")
    if nboot <= 2:
        raise ValueError("nboot must be greater than 2")

    if lag is None:
        lag = _coexplosive_select_lag(y, x, lag_grid)

    tn = len(y)
    lo = max(lag, 0)
    hi = tn + min(lag, 0)
    idx_y = np.arange(lo, hi)
    idx_x = idx_y - lag
    xreg = x[idx_x]

    S, resid, _sigma2, n = _coexplosive_stat_aligned(y[idx_y], xreg)

    rng = np.random.default_rng(seed)
    boot_S = np.empty(nboot)
    for b in range(nboot):
        ystar = rng.normal(size=n) * resid
        boot_S[b] = _coexplosive_stat_aligned(ystar, xreg)[0]

    cv = float(np.quantile(boot_S, sig_lvl / 100))
    p_value = float(np.mean(boot_S > S))

    return CobubbleTestResult(
        S=S,
        lag=int(lag),
        cv=cv,
        p_value=p_value,
        reject=bool(S > cv),
        sig_lvl=sig_lvl,
        nboot=nboot,
    )
