"""Dating and root inference: post-detection tools that operate on an
episode radf()/datestamp() has already flagged as explosive. Two families
so far, ported from exuber's R/rootstamp.R and R/dating_pdc.R (see
docs/dating-and-root-inference.md in the umbrella repo for the full
methodology and validation record):

  - rootstamp()/rootstamp_episodes(): Guo, Sun & Wang (2019) normal-t CI
    and Phillips-Magdalinos (2007) Cauchy CI for the explosive AR(1) root,
    plus implied doubling time.
  - dating_pdc(): Pang, Du & Chong (2021) / Kurozumi & Skrobotov (2023)
    sequential sample-splitting bubble dating (3- or 4-regime, OLS or
    volatility-weighted WLS).

Deviation from R: R's rootstamp.radf_obj(object, ds) reads its input
series back out of `object` (a radf_obj, which carries its own data via
mat()). Python's RadfResult doesn't retain the raw data (see
datestamp.py's docstring for the same gap), so rootstamp_episodes() here
takes the original `data` as an explicit argument instead.

Indexing note: dating_pdc()'s origination/collapse/recovery are 1-indexed
row positions into `data` (matching R's own default `idx = 1:n` when no
date index is supplied) -- NOT the 0-indexed convention datestamp()'s
Episode uses. Kept 1-indexed deliberately so a number reported here matches
the equivalent R run bit-for-bit rather than silently differing by one.
"""

import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

from exuber.datestamp import Episode
from exuber.radf import _to_2d_array

# -- Root inference (Guo, Sun & Wang 2019 / Phillips-Magdalinos 2007) -------


@dataclass
class RootstampEst:
    rho: float
    se: float
    t_stat: float
    n: int
    rho_ci: tuple[float, float]
    doubling_time: float
    doubling_time_ci: tuple[float, float]
    sig_lvl: int
    type: str


def rootstamp(y, sig_lvl: int = 95, type: str = "normal") -> RootstampEst:
    """Confidence interval and doubling time for an explosive AR(1) root.

    Fits the no-intercept AR(1) y_t = rho*y_{t-1} + e_t and reports rho
    with a CI (type="normal": Guo, Sun & Wang's asymptotically-standard-
    normal t-statistic Wald interval, valid under drift/weak dependence;
    type="cauchy": Phillips & Magdalinos's fixed-root eq. 27 Cauchy
    interval) and the implied doubling time log(2)/log(rho).
    """
    if type not in ("normal", "cauchy"):
        raise ValueError("type must be 'normal' or 'cauchy'")
    if not (50 <= sig_lvl < 100):
        raise ValueError("sig_lvl must be in [50, 100)")
    alpha = 1 - sig_lvl / 100

    y = np.asarray(y, dtype=float)
    y_lag = y[:-1]
    dy = np.diff(y)

    sxx = np.sum(y_lag**2)
    sxy = np.sum(y_lag * dy)
    beta = sxy / sxx
    res = dy - beta * y_lag
    n = len(dy)
    sigma2 = np.sum(res**2) / (n - 1)
    se = np.sqrt(sigma2 / sxx)
    rho = 1 + beta
    t_stat = beta / se

    if type == "normal":
        z = NormalDist().inv_cdf(1 - alpha / 2)
        half_width = z * se
    else:
        # standard Cauchy quantile function, tan(pi*(p - 1/2)) -- identical to
        # Student's t at df=1, which is how R's qcauchy()/qt(., df=1) is
        # verified in docs/dating-and-root-inference.md.
        q = math.tan(math.pi * (1 - alpha / 2 - 0.5))
        half_width = q * (rho**2 - 1) / rho**n
    rho_ci = (rho - half_width, rho + half_width)

    def dt(r: float) -> float:
        return np.log(2) / np.log(r)

    return RootstampEst(
        rho=rho, se=se, t_stat=t_stat, n=n, rho_ci=rho_ci,
        doubling_time=dt(rho), doubling_time_ci=(dt(rho_ci[1]), dt(rho_ci[0])),
        sig_lvl=sig_lvl, type=type,
    )


@dataclass
class RootstampEpisode:
    start: int
    end: int | None
    rho: float
    rho_lower: float
    rho_upper: float
    doubling_time: float
    doubling_time_lower: float
    doubling_time_upper: float


def rootstamp_episodes(
    data, ds: dict[str, list[Episode]], sig_lvl: int = 95, type: str = "normal"
) -> dict[str, list[RootstampEpisode]]:
    """Run rootstamp() on every episode of a datestamp() result `ds`,
    computed on `data` (the same array/DataFrame passed to radf())."""
    x, columns = _to_2d_array(data)
    nc = x.shape[1]
    names = columns or [f"series{i + 1}" for i in range(nc)]
    name_to_col = {name: j for j, name in enumerate(names)}

    out: dict[str, list[RootstampEpisode]] = {}
    for series_name, episodes in ds.items():
        if series_name not in name_to_col:
            continue  # e.g. a sieve-bootstrap "panel" entry with no matching column
        y = x[:, name_to_col[series_name]]
        rows = []
        for ep in episodes:
            end = ep.end if ep.end is not None else len(y)
            fit = rootstamp(y[ep.start : end], sig_lvl=sig_lvl, type=type)
            rows.append(
                RootstampEpisode(
                    start=ep.start, end=ep.end,
                    rho=fit.rho, rho_lower=fit.rho_ci[0], rho_upper=fit.rho_ci[1],
                    doubling_time=fit.doubling_time,
                    doubling_time_lower=fit.doubling_time_ci[0],
                    doubling_time_upper=fit.doubling_time_ci[1],
                )
            )
        out[series_name] = rows
    return out


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
