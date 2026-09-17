"""Dating and root inference: post-detection tools that operate on an
episode radf()/datestamp() has already flagged as explosive. Four
families so far, ported from exuber's R/rootstamp.R, R/dating_pdc.R,
R/radf_recovery.R and R/dating_hls.R (see docs/dating-and-root-
inference.md in the umbrella repo for the full methodology and
validation record):

  - rootstamp()/rootstamp_episodes(): Guo, Sun & Wang (2019) normal-t CI
    and Phillips-Magdalinos (2007) Cauchy CI for the explosive AR(1) root,
    plus implied doubling time.
  - dating_pdc(): Pang, Du & Chong (2021) / Kurozumi & Skrobotov (2023)
    sequential sample-splitting bubble dating (3- or 4-regime, OLS or
    volatility-weighted WLS).
  - radf_recovery()/radf_recovery_cv(): Phillips & Shi (2014) reverse-
    regression crisis-origination/market-recovery dating. Shipped with
    the same caveats as R: f_r (recovery) validates well, f_c (crisis
    origination) and the false-detection rate under H0 are noisier and
    not fully resolved -- see radf_recovery()'s docstring and
    docs/dating-and-root-inference.md's "Reverse-regression recovery
    dating" section.
  - dating_hls(): Harvey, Leybourne & Sollis (2017) SSR+BIC dating --
    four candidate regime-dummy models, jointly-fit breakpoint search,
    BIC model selection.

Deviation from R: R's rootstamp.radf_obj(object, ds) reads its input
series back out of `object` (a radf_obj, which carries its own data via
mat()). Python's RadfResult doesn't retain the raw data (see
datestamp.py's docstring for the same gap), so rootstamp_episodes() here
takes the original `data` as an explicit argument instead.

Indexing note: dating_pdc()'s and dating_hls()'s origination/collapse/
recovery are 1-indexed row positions into `data` (matching R's own
default `idx = 1:n` when no date index is supplied) -- NOT the 0-indexed
convention datestamp()'s Episode uses. Kept 1-indexed deliberately so a
number reported here matches the equivalent R run bit-for-bit rather
than silently differing by one. radf_recovery()'s f_c/f_r, by contrast,
ARE 0-indexed positions (the same convention as datestamp()'s Episode),
since they're derived directly from radf()'s own bsadf array the same
way datestamp() is.
"""

import math
import warnings
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

from exuber._unroot import unroot
from exuber.cv import PCNT
from exuber.datestamp import SIG_IDX, Episode
from exuber.radf import _to_2d_array, psy_minw, radf

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


# -- Reverse-regression recovery dating (Phillips & Shi 2014) ---------------

_RECOVERY_CAVEAT = (
    "Experimental. f_c and the overall false-detection rate are exploratory "
    "pending further validation; see radf_recovery()'s docstring."
)


@dataclass
class RadfRecoveryCv:
    bsadf_cv: np.ndarray  # (n_minw, 3), columns 90%/95%/99%
    minw: int
    n: int
    iter: int
    lag: int = 0


def _radf_recovery_mc(
    n: int, minw: int | None = None, nrep: int = 1000, seed: int | None = None, lag: int = 0
) -> tuple[np.ndarray, int]:
    from . import _core  # lazy: see radf.py's radf() for why

    minw = minw if minw is not None else psy_minw(n)
    rng = np.random.default_rng(seed)
    n_minw = n - minw - lag

    badf = np.empty((n_minw, nrep))
    for i in range(nrep):
        y = np.cumsum(rng.normal(size=n))[::-1]  # reversed null path -- see module docstring
        yxmat = unroot(y, lag=lag)
        result = _core.radf_stat(yxmat, minw, lag)
        badf[:, i] = result[:n_minw]
    return badf, minw


def radf_recovery_cv(
    n: int, minw: int | None = None, nrep: int = 1000, seed: int | None = None, lag: int = 0
) -> RadfRecoveryCv:
    """Monte Carlo critical values for radf_recovery()'s reverse-time bsadf
    statistic, calibrated to Phillips & Shi (2014)'s own (different from
    the forward test's) null limiting distribution: reversal induces an
    endogeneity between the reverse-time regressor and its own innovation
    with no forward-regression analogue (see this module's docstring), so
    reusing radf_mc_cv()'s forward critical values would be a mismatched
    boundary, not an approximation of known quality.
    """
    badf, minw = _radf_recovery_mc(n, minw, nrep, seed, lag)
    bsadf_cv = np.quantile(np.maximum.accumulate(badf, axis=0), PCNT, axis=1).T
    return RadfRecoveryCv(bsadf_cv=bsadf_cv, minw=minw, n=n, iter=nrep, lag=lag)


@dataclass
class RadfRecoveryResult:
    f_c: np.ndarray  # crisis-origination date, NaN if not identified
    f_r: np.ndarray  # market-recovery date, NaN if not identified
    detected: np.ndarray
    censored: np.ndarray
    series_names: list[str] | None
    sig_lvl: int
    minw: int
    lag: int
    n: int


def _recovery_dates_from_bsadf(
    bsadf_rev: np.ndarray,
    bsadf_cv: np.ndarray,
    minw: int,
    lag: int,
    n: int,
    sig_lvl: int = 95,
    series_names: list[str] | None = None,
) -> RadfRecoveryResult:
    """Phillips & Shi (2014) eq. 8-9 crossing rule, operating on an
    already-computed reverse-time bsadf array (kept separate from
    radf_recovery() so the crossing logic is testable without radf()/the
    C++ extension -- same split as datestamp() operating on a
    pre-computed RadfResult/RadfCv).

    Locates the first up-crossing of the reversal-calibrated boundary
    (market recovery, f_r) then the next down-crossing, searched only
    after the up-crossing (crisis origination in the original series,
    f_c) -- so f_c <= f_r always, by construction, whenever both are
    identified and uncensored. Positions are 0-indexed into the original
    (non-reversed) series, matching datestamp()'s Episode convention.
    """
    if sig_lvl not in SIG_IDX:
        raise ValueError("sig_lvl must be one of 90, 95, 99")
    sidx = SIG_IDX[sig_lvl]
    zadj = minw + lag
    n_minw, nc = bsadf_rev.shape
    names = series_names or [f"series{i + 1}" for i in range(nc)]

    f_c = np.full(nc, np.nan)
    f_r = np.full(nc, np.nan)
    detected = np.zeros(nc, dtype=bool)
    censored = np.zeros(nc, dtype=bool)

    for j in range(nc):
        exceed = bsadf_rev[:, j] > bsadf_cv[:, sidx]
        exceed_idx = np.where(exceed)[0]
        if len(exceed_idx) == 0:
            continue
        detected[j] = True
        g_e = exceed_idx[0]
        rev_pos_e = g_e + zadj
        f_r[j] = n - 1 - rev_pos_e

        not_exceed_after = np.where(~exceed[g_e:])[0]
        if len(not_exceed_after) == 0:
            censored[j] = True
            continue
        g_c = g_e + not_exceed_after[0]
        rev_pos_c = min(g_c + zadj, n - 1)
        f_c[j] = n - 1 - rev_pos_c

    return RadfRecoveryResult(
        f_c=f_c, f_r=f_r, detected=detected, censored=censored,
        series_names=names, sig_lvl=sig_lvl, minw=minw, lag=lag, n=n,
    )


def radf_recovery(
    data,
    minw: int | None = None,
    lag: int = 0,
    nrep: int = 1000,
    sig_lvl: int = 95,
    seed: int | None = None,
) -> RadfRecoveryResult:
    """Reverse-regression dating of crisis origination and market recovery
    (Phillips & Shi 2014). Reverses the series, runs radf()'s existing
    bsadf recursion on it, and locates the first up-crossing of a
    reversal-calibrated critical value boundary (market recovery, f_r)
    followed by the next down-crossing (crisis/collapse origination in
    the original series, f_c), then maps both back to the original time
    index. f_c <= f_r always, by construction, when both are identified.

    Caveats (validation status, see docs/dating-and-root-inference.md's
    "Reverse-regression recovery dating" for the full numbers): f_r
    validates well against synthetic collapse-then-recovery data. f_c
    shows a materially larger residual bias, and the empirical
    false-detection rate under a pure random-walk null is around 29% at
    n=100/minw=20/95% -- higher than comparable forward-test numbers
    elsewhere in this package. Treat f_c and the overall detection rate
    as exploratory pending further validation.
    """
    if sig_lvl not in SIG_IDX:
        raise ValueError("sig_lvl must be one of 90, 95, 99")
    warnings.warn(_RECOVERY_CAVEAT, stacklevel=2)

    x, columns = _to_2d_array(data)
    n = x.shape[0]
    minw = minw if minw is not None else psy_minw(n)

    x_rev = x[::-1, :]
    rev_fit = radf(x_rev, minw=minw, lag=lag)
    cv = radf_recovery_cv(n=n, minw=minw, nrep=nrep, seed=seed, lag=lag)

    return _recovery_dates_from_bsadf(
        rev_fit.bsadf, cv.bsadf_cv, minw=minw, lag=lag, n=n, sig_lvl=sig_lvl, series_names=columns
    )


# -- SSR/BIC bubble dating (Harvey, Leybourne & Sollis 2017) ----------------
#
# Replaces PSY's threshold-crossing rule with a model-based SSR-minimisation
# + BIC rule: four candidate regime-dummy regressions of Delta y_t on
# y_{t-1} (unit-root-to-end / unit-root-bubble-unit-root / unit-root-
# bubble-collapse / unit-root-bubble-collapse-unit-root), each fit by
# residual-sum-of-squares minimisation over candidate break fractions
# (jointly, per model), with BIC selecting among the four. Because the
# four models' dummy windows never overlap, each candidate partition's
# SSR is exactly the sum of independent per-segment closed-form OLS fits
# (a no-dummy segment has SSR = sum(Delta y_t^2); a dummy segment is a
# plain intercept+slope fit) -- a closed-form ratio of cumulative sums,
# the same style of O(1)-per-candidate lookup _pdc_find_break() uses, so
# the joint grid search needs no repeated regression fit, only prefix-sum
# differences (matches exuber's own R/dating_hls.R exactly).
#
# Indexing: breakpoints are internally 0-indexed "boundary counts" b in
# 0..n1 (n1 = len(y)-1) -- y[b] is the observation at that boundary (same
# convention _pdc_find_break() uses). dating_hls()'s *output*
# origination/collapse/recovery add 1 to match R's own dating_hls()
# output number (R: idx[b + 1L], i.e. R's 1-indexed position b+1) --
# matching dating_pdc()'s existing "R-bit-for-bit" convention, not
# datestamp()'s 0-indexed Episode convention.


@dataclass
class _HlsPrefixSums:
    cx: np.ndarray
    cx2: np.ndarray
    cz: np.ndarray
    cz2: np.ndarray
    cxz: np.ndarray
    n1: int


def _hls_prefix_sums(y: np.ndarray) -> _HlsPrefixSums:
    n1 = len(y) - 1
    x = y[:n1]
    z = np.diff(y)
    return _HlsPrefixSums(
        cx=np.concatenate(([0.0], np.cumsum(x))),
        cx2=np.concatenate(([0.0], np.cumsum(x**2))),
        cz=np.concatenate(([0.0], np.cumsum(z))),
        cz2=np.concatenate(([0.0], np.cumsum(z**2))),
        cxz=np.concatenate(([0.0], np.cumsum(x * z))),
        n1=n1,
    )


def _hls_segment_ssr(ps: _HlsPrefixSums, lo, hi, fit: bool):
    """SSR of the segment(s) with i-index in (lo, hi] (lo/hi may be
    scalars or equal-length/broadcastable arrays). fit=False: no active
    dummy, SSR = sum(z^2). fit=True: intercept + slope OLS of z on x."""
    sx = ps.cx[hi] - ps.cx[lo]
    sxx = ps.cx2[hi] - ps.cx2[lo]
    sz = ps.cz[hi] - ps.cz[lo]
    szz = ps.cz2[hi] - ps.cz2[lo]
    if not fit:
        return szz
    sxz = ps.cxz[hi] - ps.cxz[lo]
    n_seg = np.asarray(hi) - np.asarray(lo)
    b = (n_seg * sxz - sx * sz) / (n_seg * sxx - sx**2)
    a = (sz - b * sx) / n_seg
    return szz - a * sz - b * sxz


def _hls_model1(y: np.ndarray, ps: _HlsPrefixSums, trim: float) -> tuple[int | None, float]:
    """Model 1: unit root -> bubble to sample end. Sign constraint:
    y_T > y_tau1 (series ends above where the bubble started)."""
    n1 = ps.n1
    k_min = max(2, math.ceil(trim * n1))
    taus = np.arange(k_min, n1 - k_min + 1)
    taus = taus[y[n1] > y[taus]]
    if len(taus) == 0:
        return None, math.inf
    ssr = _hls_segment_ssr(ps, 0, taus, False) + _hls_segment_ssr(ps, taus, n1, True)
    best = int(np.argmin(ssr))
    return int(taus[best]), float(ssr[best])


def _hls_model23(
    y: np.ndarray, ps: _HlsPrefixSums, trim: float, right_fit: bool
) -> tuple[int | None, int | None, float]:
    """Model 2 (right_fit=False, post-bubble tail unfitted) / Model 3
    (right_fit=True, post-bubble tail fitted as its own collapse regime).
    Sign constraint: y_tau2 > y_tau1 always; right_fit also requires
    y_tau2 > y_T (the peak must exceed the fitted collapse's endpoint)."""
    n1 = ps.n1
    k_min = max(2, math.ceil(trim * n1))
    best_ssr, best_tau1, best_tau2 = math.inf, None, None
    tau1_max = n1 - 2 * k_min
    if tau1_max < k_min:
        return None, None, math.inf
    for tau1 in range(k_min, tau1_max + 1):
        tau2 = np.arange(tau1 + k_min, n1 - k_min + 1)
        valid = y[tau2] > y[tau1]
        if right_fit:
            valid = valid & (y[tau2] > y[n1])
        tau2 = tau2[valid]
        if len(tau2) == 0:
            continue
        ssr = (
            _hls_segment_ssr(ps, 0, tau1, False)
            + _hls_segment_ssr(ps, tau1, tau2, True)
            + _hls_segment_ssr(ps, tau2, n1, right_fit)
        )
        j = int(np.argmin(ssr))
        if ssr[j] < best_ssr:
            best_ssr, best_tau1, best_tau2 = float(ssr[j]), tau1, int(tau2[j])
    return best_tau1, best_tau2, best_ssr


def _hls_model4(
    y: np.ndarray, ps: _HlsPrefixSums, trim: float
) -> tuple[int | None, int | None, int | None, float]:
    """Model 4: unit root -> bubble -> collapse -> unit root recovery.
    Sign constraints: y_tau2 > y_tau1 and y_tau2 > y_tau3 (the peak must
    exceed both the bubble's start and the fitted collapse's endpoint)."""
    n1 = ps.n1
    k_min = max(2, math.ceil(trim * n1))
    best_ssr: float = math.inf
    best_tau1: int | None = None
    best_tau2: int | None = None
    best_tau3: int | None = None
    tau1_max = n1 - 3 * k_min
    if tau1_max < k_min:
        return None, None, None, math.inf
    for tau1 in range(k_min, tau1_max + 1):
        tau2_max = n1 - 2 * k_min
        for tau2 in range(tau1 + k_min, tau2_max + 1):
            if y[tau2] <= y[tau1]:
                continue
            tau3 = np.arange(tau2 + k_min, n1 - k_min + 1)
            tau3 = tau3[y[tau2] > y[tau3]]
            if len(tau3) == 0:
                continue
            ssr = (
                _hls_segment_ssr(ps, 0, tau1, False)
                + _hls_segment_ssr(ps, tau1, tau2, True)
                + _hls_segment_ssr(ps, tau2, tau3, True)
                + _hls_segment_ssr(ps, tau3, n1, False)
            )
            j = int(np.argmin(ssr))
            if ssr[j] < best_ssr:
                best_ssr = float(ssr[j])
                best_tau1, best_tau2, best_tau3 = tau1, tau2, int(tau3[j])
    return best_tau1, best_tau2, best_tau3, best_ssr


def _hls_bic(ssr: float, n: int, df: int) -> float:
    return n * math.log(ssr / n) + df * math.log(n)


_HLS_DFS = (3, 4, 6, 7)  # models 1-4


@dataclass
class HlsFit:
    model: int  # 1-4
    breaks: dict[str, int | None]  # "tau1"/"tau2"/"tau3" -> 0-indexed boundary count
    bic: np.ndarray  # length 4, inf for unrequested models


def _hls_fit_series(y: np.ndarray, trim: float, models: tuple[int, ...] = (1, 2, 3, 4)) -> HlsFit:
    """Fits the requested subset of HLS's four models to a single series
    `y` and BIC-selects among them. Shared by dating_hls() and (per-window,
    restricted to models (2, 4)) dating_hlw()."""
    n = len(y)
    ps = _hls_prefix_sums(y)
    bic = np.full(4, math.inf)
    fits: dict[int, tuple] = {}

    if 1 in models:
        tau1, ssr = _hls_model1(y, ps, trim)
        fits[1] = (tau1,)
        bic[0] = _hls_bic(ssr, n, _HLS_DFS[0])
    if 2 in models:
        tau1, tau2, ssr = _hls_model23(y, ps, trim, right_fit=False)
        fits[2] = (tau1, tau2)
        bic[1] = _hls_bic(ssr, n, _HLS_DFS[1])
    if 3 in models:
        tau1, tau2, ssr = _hls_model23(y, ps, trim, right_fit=True)
        fits[3] = (tau1, tau2)
        bic[2] = _hls_bic(ssr, n, _HLS_DFS[2])
    if 4 in models:
        tau1, tau2, tau3, ssr = _hls_model4(y, ps, trim)
        fits[4] = (tau1, tau2, tau3)
        bic[3] = _hls_bic(ssr, n, _HLS_DFS[3])

    jopt = int(np.argmin(bic)) + 1
    fit = fits[jopt]
    if jopt == 1:
        breaks = {"tau1": fit[0]}
    elif jopt in (2, 3):
        breaks = {"tau1": fit[0], "tau2": fit[1]}
    else:
        breaks = {"tau1": fit[0], "tau2": fit[1], "tau3": fit[2]}
    return HlsFit(model=jopt, breaks=breaks, bic=bic)


@dataclass
class DatingHlsResult:
    model: np.ndarray  # int per series, 1-4
    origination: np.ndarray  # NaN if not applicable
    collapse: np.ndarray
    recovery: np.ndarray
    bic: np.ndarray  # (nc, 4)
    series_names: list[str] | None
    trim: float
    n: int


def dating_hls(data, trim: float = 0.05) -> DatingHlsResult:
    """SSR/BIC bubble dating (Harvey, Leybourne & Sollis 2017). Fits four
    candidate regime-dummy regressions of Delta y_t on y_{t-1} (unit-
    root-to-end / unit-root-bubble-unit-root / unit-root-bubble-collapse
    / unit-root-bubble-collapse-unit-root), each by joint residual-sum-
    -of-squares minimisation over its candidate break fraction(s), and
    selects among them by BIC.

    Unlike datestamp() (threshold-crossing on the recursive BSADF
    statistic) or dating_pdc() (a fixed 3/4-regime structure with
    sequentially, not jointly, estimated breaks), this jointly searches
    breakpoints within each of four candidate regime structures and lets
    BIC pick the structure itself. Needs no critical values -- this is
    model selection, not a hypothesis test.
    """
    x, columns = _to_2d_array(data)
    n, nc = x.shape
    names = columns or [f"series{i + 1}" for i in range(nc)]

    model = np.empty(nc, dtype=int)
    origination = np.full(nc, np.nan)
    collapse = np.full(nc, np.nan)
    recovery = np.full(nc, np.nan)
    bic_mat = np.full((nc, 4), np.nan)

    for j in range(nc):
        fit = _hls_fit_series(x[:, j], trim, models=(1, 2, 3, 4))
        model[j] = fit.model
        bic_mat[j, :] = fit.bic
        b = fit.breaks
        tau1, tau2, tau3 = b.get("tau1"), b.get("tau2"), b.get("tau3")
        if tau1 is not None:
            origination[j] = tau1 + 1
        if tau2 is not None:
            collapse[j] = tau2 + 1
        if tau3 is not None:
            recovery[j] = tau3 + 1

    return DatingHlsResult(
        model=model, origination=origination, collapse=collapse, recovery=recovery,
        bic=bic_mat, series_names=names, trim=trim, n=n,
    )
