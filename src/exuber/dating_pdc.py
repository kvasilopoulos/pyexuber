"""Sequential sample-splitting bubble dating: Pang, Du & Chong (2021) /
Kurozumi & Skrobotov (2023) sequential sample-splitting bubble dating
(3- or 4-regime, OLS or volatility-weighted WLS). Ported from exuber's
R/dating_pdc.R (see docs/dating-and-root-inference.md in the umbrella
repo for the full methodology and validation record).

Indexing note: dating_pdc()'s origination/collapse/recovery are
1-indexed row positions into `data` (matching R's own default
`idx = 1:n` when no date index is supplied) -- NOT the 0-indexed
convention datestamp()'s Episode uses. Kept 1-indexed deliberately so a
number reported here matches the equivalent R run bit-for-bit rather
than silently differing by one.
"""

from dataclasses import dataclass

import numpy as np

from exuber.radf import _to_2d_array

# -- Sequential sample-splitting bubble dating (PDC 2021 / KS 2023) ---------


def _pdc_find_break(
    y: np.ndarray, trim: float = 0.05, weights: np.ndarray | None = None
) -> tuple[int, float]:
    """Single-breakpoint no-intercept AR(1) sample split: the split count k
    (number of (y_{t-1}, y_t) pairs assigned to the left regime) minimizing
    the combined residual sum of squares of two no-intercept AR(1) fits,
    via O(T) cumulative sums. `weights` (length len(y)-1, one per pair)
    gives Kurozumi & Skrobotov (2023)'s WLS-weighted variant; None is plain
    OLS. Returns (break_idx, rss); `y[:break_idx]` is the left regime.
    """
    n1 = len(y) - 1
    ylag = y[:n1]
    ycur = y[1 : n1 + 1]
    w = np.ones(n1) if weights is None else np.asarray(weights, dtype=float)

    csxx = np.concatenate(([0.0], np.cumsum(w * ylag**2)))
    csxy = np.concatenate(([0.0], np.cumsum(w * ylag * ycur)))
    csyy = np.concatenate(([0.0], np.cumsum(w * ycur**2)))

    total_xx, total_xy, total_yy = csxx[n1], csxy[n1], csyy[n1]

    k_min = max(2, int(np.ceil(trim * n1)))
    k_max = n1 - k_min
    if k_min >= k_max:
        raise ValueError("Series too short for the requested 'trim' fraction.")
    ks = np.arange(k_min, k_max + 1)

    sxx_l, sxy_l, syy_l = csxx[ks], csxy[ks], csyy[ks]
    sxx_r, sxy_r, syy_r = total_xx - sxx_l, total_xy - sxy_l, total_yy - syy_l

    rss = (syy_l - sxy_l**2 / sxx_l) + (syy_r - sxy_r**2 / sxx_r)
    best = int(np.argmin(rss))
    return int(ks[best]), float(rss[best])


def _pdc_regime_resid(y: np.ndarray, breaks: list[int]) -> np.ndarray:
    """Fitted no-intercept-AR(1) residuals of the piecewise regime model
    implied by 1-indexed break counts `breaks` (same semantics as
    _pdc_find_break's break_idx), one per (y_{t-1}, y_t) pair."""
    n1 = len(y) - 1
    ylag = y[:n1]
    ycur = y[1 : n1 + 1]
    bounds = [0, *breaks, n1]

    resid = np.empty(n1)
    for i in range(len(bounds) - 1):
        idx = slice(bounds[i], bounds[i + 1])
        rho = np.sum(ylag[idx] * ycur[idx]) / np.sum(ylag[idx] ** 2)
        resid[idx] = ycur[idx] - rho * ylag[idx]
    return resid


def _nw_spot_vol(
    e: np.ndarray, kernel: str = "gaussian", h: float | None = None
) -> tuple[np.ndarray, float]:
    """Nadaraya-Watson spot-volatility estimator: nonparametric smooth of
    e^2 over t/T, with leave-one-out cross-validated bandwidth when `h` is
    not given. Returns (sigma2, h); sigma2 has the same length as `e`."""
    if kernel not in ("gaussian", "uniform"):
        raise ValueError("kernel must be 'gaussian' or 'uniform'")
    tn = len(e)
    t_grid = np.arange(1, tn + 1) / tn
    s = t_grid[1:]  # (2:Tn)/Tn -- t_grid without its first point
    e2 = e[1:] ** 2  # e_2..e_Tn squared

    if kernel == "gaussian":
        def kern(u):
            return np.exp(-(u**2) / 2) / np.sqrt(2 * np.pi)
    else:
        def kern(u):
            return (np.abs(u) <= 1) / 2.0

    def spot_vol_at(hh: float, drop0: bool = False) -> np.ndarray:
        diff = (s[None, :] - t_grid[:, None]) / hh
        w = kern(diff)
        if drop0:
            w[np.abs(s[None, :] - t_grid[:, None]) < np.sqrt(np.finfo(float).eps)] = 0.0
        wsum = w.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = (w * e2[None, :]).sum(axis=1) / wsum
        out[wsum <= 0] = np.mean(e2)
        return out

    if h is None:
        hl, hu = 1 / (2 * tn), 1 / 6
        grid = np.exp(np.linspace(np.log(hl), np.log(hu), 10))
        cv = np.array([np.mean((e2 - spot_vol_at(hh, drop0=True)[1:]) ** 2) for hh in grid])
        h = float(grid[np.argmin(cv)])

    sigma2 = spot_vol_at(h, drop0=False)
    return sigma2, h


@dataclass
class DatingPdcResult:
    origination: np.ndarray
    collapse: np.ndarray
    recovery: np.ndarray | None
    series_names: list[str] | None
    regimes: int
    type: str


def dating_pdc(
    data,
    regimes: int = 3,
    trim: float = 0.05,
    type: str = "ols",
    kernel: str = "gaussian",
    h: float | None = None,
) -> DatingPdcResult:
    """Sequential sample-splitting bubble dating (Pang, Du & Chong 2021;
    Kurozumi & Skrobotov 2023). Fits a fixed regime structure (unit-root,
    explosive, stationary-collapse, and optionally a final unit-root
    recovery regime) via sequential O(T) no-intercept AR(1) breakpoint
    search -- collapse first, then origination on the left subsample, then
    (regimes=4) recovery on the right subsample -- no joint grid search or
    BIC model selection.

    type="wls" adds Kurozumi & Skrobotov (2023)'s volatility correction:
    fit OLS first, smooth its fitted regime residuals' squares via a
    Nadaraya-Watson spot-volatility estimator, and re-run the same
    sequential search weighted by the inverse estimated spot variance.
    """
    if regimes not in (3, 4):
        raise ValueError("regimes must be 3 or 4")
    if type not in ("ols", "wls"):
        raise ValueError("type must be 'ols' or 'wls'")

    x, columns = _to_2d_array(data)
    nc = x.shape[1]
    names = columns or [f"series{i + 1}" for i in range(nc)]

    def fit_sequential(y: np.ndarray, weights_full: np.ndarray | None = None):
        def w_left(b: int):
            return None if weights_full is None else weights_full[: b - 1]

        def w_right(b: int):
            return None if weights_full is None else weights_full[b:]

        b2, _ = _pdc_find_break(y, trim, weights=weights_full)
        b1, _ = _pdc_find_break(y[:b2], trim, weights=w_left(b2))
        breaks = [b1, b2]
        b3 = None
        if regimes == 4:
            y_right = y[b2:]
            b3_rel, _ = _pdc_find_break(y_right, trim, weights=w_right(b2))
            b3 = b2 + b3_rel
            breaks = [b1, b2, b3]
        return b1, b2, b3, breaks

    origination = np.empty(nc)
    collapse = np.empty(nc)
    recovery = np.empty(nc) if regimes == 4 else None

    for j in range(nc):
        y = x[:, j]
        b1, b2, b3, breaks = fit_sequential(y)

        if type == "wls":
            resid = _pdc_regime_resid(y, breaks)
            sigma2, _ = _nw_spot_vol(resid, kernel=kernel, h=h)
            b1, b2, b3, breaks = fit_sequential(y, weights_full=1 / sigma2)

        origination[j] = b1
        collapse[j] = b2
        if regimes == 4:
            assert recovery is not None
            recovery[j] = b3

    return DatingPdcResult(
        origination=origination, collapse=collapse, recovery=recovery,
        series_names=names, regimes=regimes, type=type,
    )
