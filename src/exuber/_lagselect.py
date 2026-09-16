"""Shared OLS lag-selection / null-AR-fit subsystem. Port of exuber's
R/radf_wb.R (`adf_res`, `lag_select`, `lag_select_table`) and the two
regression-matrix builders in R/unroot.R (`unroot_adf`, `unroot_adf_null`)
that back them. This is internal plumbing shared by the Phillips-Shi wild
bootstrap (`radf_wb_ps_cv`) and the sieve bootstrap's automatic lag
selection (`radf_sb_cv(type="aic"/"bic")`) -- see cv.py.
"""

from dataclasses import dataclass

import numpy as np

from exuber._unroot import _embed


def unroot_adf(x: np.ndarray, lag: int) -> np.ndarray:
    """Port of R's unroot_adf(): columns dy, ct, y (level, lagged once),
    dy_lags1..lag. This is the ADF test regression matrix, used only for
    AIC/BIC lag-order selection (lag_select_table)."""
    x = np.asarray(x, dtype=float)
    x_lag = _embed(x, lag + 2)[:, 1]
    dx_embed = _embed(np.diff(x), lag + 1)
    dy = dx_embed[:, 0]
    ct = np.ones_like(dy)
    if lag == 0:
        return np.column_stack([dy, ct, x_lag])
    return np.column_stack([dy, ct, x_lag, dx_embed[:, 1:]])


def unroot_adf_null(x: np.ndarray, lag: int) -> np.ndarray:
    """Port of R's unroot_adf_null(): columns dy, ct, dy_lags1..lag -- no
    level term. This is the null AR(lag)-in-differences DGP regression
    fit by adf_res() and resampled by the Phillips-Shi wild bootstrap."""
    x = np.asarray(x, dtype=float)
    dx_embed = _embed(np.diff(x), lag + 1)
    dy = dx_embed[:, 0]
    ct = np.ones_like(dy)
    if lag == 0:
        return np.column_stack([dy, ct])
    return np.column_stack([dy, ct, dx_embed[:, 1:]])


def _ols(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OLS fit of y on x (x already includes its intercept/`ct` column).
    Returns (coefficients, residuals)."""
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return beta, y - x @ beta


def _aic_bic(res: np.ndarray, n_params: int) -> tuple[float, float]:
    """R stats::AIC.lm/BIC.lm for a Gaussian OLS fit (literal port of
    stats:::logLik.lm's formula: -0.5*n*(log(2*pi) + 1 - log(n) + log(RSS)),
    df = n_params + 1 for the estimated residual variance)."""
    n = len(res)
    rss = float(np.sum(res**2))
    loglik = -0.5 * n * (np.log(2 * np.pi) + 1 - np.log(n) + np.log(rss))
    df = n_params + 1
    return -2 * loglik + 2 * df, -2 * loglik + np.log(n) * df


def lag_select_table(x: np.ndarray, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Port of R's lag_select_table(): AIC/BIC of the ADF regression
    (unroot_adf) for lag = 0..max_lag."""
    aic = np.empty(max_lag + 1)
    bic = np.empty(max_lag + 1)
    for lag in range(max_lag + 1):
        yxmat = unroot_adf(x, lag)
        _, res = _ols(yxmat[:, 0], yxmat[:, 1:])
        aic[lag], bic[lag] = _aic_bic(res, yxmat.shape[1] - 1)
    return aic, bic


def lag_select(x: np.ndarray, criterion: str = "aic", max_lag: int = 8) -> int:
    """Port of R's lag_select(): the lag in 0..max_lag minimizing AIC or
    BIC of the ADF regression. max_lag = 0 is a no-op returning 0 (matches
    R's early-return, minus its message)."""
    if criterion not in ("aic", "bic"):
        raise ValueError('criterion must be "aic" or "bic"')
    if max_lag == 0:
        return 0
    aic, bic = lag_select_table(x, max_lag)
    return int(np.argmin(aic if criterion == "aic" else bic))


@dataclass
class AdfRes:
    beta: np.ndarray
    res: np.ndarray


def adf_res(x: np.ndarray, adflag: int = 0, type: str = "fixed") -> AdfRes:
    """Port of R's adf_res(): fits the null AR(adflag)-in-differences model
    (unroot_adf_null) by OLS. When type != "fixed", `adflag` is instead the
    max_lag searched by lag_select() to resolve the actual lag order."""
    x = np.asarray(x, dtype=float)
    if type != "fixed":
        adflag = lag_select(x, criterion=type, max_lag=adflag)
    yxmat = unroot_adf_null(x, adflag)
    beta, res = _ols(yxmat[:, 0], yxmat[:, 1:])
    return AdfRes(beta=beta, res=res)
