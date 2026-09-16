"""Multivariate bubble tests. Ports of exuber's R/radf_common.R (this
section) -- see docs/multivariate.md for the full evaluation each
function implements.
"""

from dataclasses import dataclass

import numpy as np

from exuber.cv import PCNT, RadfCv
from exuber.radf import RadfResult, _to_2d_array, psy_minw, radf

# -- Common-bubble detection via PCA + PSY (Chen, Phillips & Shi 2023) ------


@dataclass
class RadfCommonResult(RadfResult):
    """RadfResult plus the PCA info radf_common() extracted it from."""

    loadings: np.ndarray | None = None
    explained_variance_ratio: np.ndarray | None = None


def _pca(x: np.ndarray, r: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """First r principal components of x (centered, not scaled) via SVD --
    the numerical core of radf_common(), factored out so it's testable
    without the compiled _core extension radf() itself needs. Returns
    (loadings, scores, explained_variance_ratio): loadings is R's
    prcomp()$rotation, scores is prcomp()$x, restricted to r columns.
    """
    xc = x - x.mean(axis=0)
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    r = min(r, vt.shape[0])
    loadings = vt[:r].T
    scores = xc @ loadings
    explained = (s[:r] ** 2) / np.sum(s**2)
    return loadings, scores, explained


def radf_common(data, minw: int | None = None, r: int = 1) -> RadfCommonResult:
    """Common-bubble detection via PCA + PSY (Chen, Phillips & Shi 2023).

    Extracts the panel's first principal component (PCA via SVD on the
    centered panel) and runs the ordinary radf() on its scores -- the
    paper's own Theorem 4.2/footnote 3: PC1 is "sufficient... for the
    purpose of bubble identification." `r` > 1 only changes how many
    components are kept in `.loadings`/`.explained_variance_ratio` for
    inspection; detection always uses PC1.

    Use radf_common_cv(), NOT radf_mc_cv(), for critical values --
    radf_mc_cv() has no dependence on panel width N and was independently
    found to be badly undersized here once N grows past a handful of
    series (docs/multivariate.md, "Independent validation (2026-08-09)").
    """
    x, columns = _to_2d_array(data)
    if x.shape[1] < 2:
        raise ValueError("radf_common needs a panel of at least 2 series.")
    if np.isnan(x).any():
        raise ValueError("radf_common: data contains NaN.")

    loadings, scores, explained = _pca(x, r)
    res = radf(scores[:, 0], minw=minw)
    return RadfCommonResult(
        adf=res.adf,
        badf=res.badf,
        sadf=res.sadf,
        bsadf=res.bsadf,
        gsadf=res.gsadf,
        bsadf_panel=res.bsadf_panel,
        gsadf_panel=res.gsadf_panel,
        minw=res.minw,
        lag=res.lag,
        n=res.n,
        series_names=columns,
        loadings=loadings,
        explained_variance_ratio=explained,
    )


@dataclass
class RadfCommonCv(RadfCv):
    """RadfCv plus the panel width N its null was simulated at."""

    N: int = 0


def radf_common_cv(
    n: int, N: int, minw: int | None = None, nrep: int = 1000, seed: int | None = None
) -> RadfCommonCv:
    """Critical values for radf_common(), simulated under its OWN null.

    Theorem 4.3 of Chen, Phillips & Shi (2023) claims the PSY-on-PC1
    statistic's limiting null is asymptotically identical to plain
    univariate GSADF's (independent of panel width N) -- but independent
    validation found this does not hold at practical N: the true null
    quantile *grows* with N (the opposite of a naive reading of the
    paper's own finite-sample section), reaching more than double
    radf_mc_cv()'s 95% quantile at N=100. See docs/multivariate.md,
    "Independent validation (2026-08-09)", for the full finding.

    Simulates the null radf_common() actually needs: an N-column panel of
    *independent* random walks (no true common factor -- the sharpest
    possible null), extracted to PC1 and tested exactly as radf_common()
    does. `N` must match the panel radf_common() was actually run on --
    unlike radf_mc_cv(), this null distribution depends on it.
    """
    if n <= 5:
        raise ValueError("n must be greater than 5")
    if N <= 1:
        raise ValueError("N must be greater than 1")
    if nrep <= 0:
        raise ValueError("nrep must be positive")
    minw = minw if minw is not None else psy_minw(n)
    if minw <= 2:
        raise ValueError("minw must be greater than 2")

    rng = np.random.default_rng(seed)
    n_minw = n - minw

    adf = np.empty(nrep)
    sadf = np.empty(nrep)
    gsadf = np.empty(nrep)
    bsadf = np.empty((n_minw, nrep))

    for i in range(nrep):
        panel = np.cumsum(rng.normal(size=(n, N)), axis=0)
        res = radf_common(panel, minw=minw)
        adf[i] = res.adf[0]
        sadf[i] = res.sadf[0]
        gsadf[i] = res.gsadf[0]
        bsadf[:, i] = res.bsadf[:, 0]

    adf_cv = np.quantile(adf, PCNT)
    sadf_cv = np.quantile(sadf, PCNT)
    gsadf_cv = np.quantile(gsadf, PCNT)
    # Matches R's apply(bsadf_mat, 2, cummax) |> apply(1, quantile): cummax
    # of each replication's simulated bsadf path, THEN quantiles across
    # replications -- not the same construction as radf_mc_cv()'s own
    # (which cummaxes badf instead); ported as its own R source does it.
    bsadf_cv = np.quantile(np.maximum.accumulate(bsadf, axis=0), PCNT, axis=1).T
    asy_adf_crit = np.array([-0.44, -0.08, 0.6])
    badf_cv = np.tile(asy_adf_crit, (n_minw, 1))

    return RadfCommonCv(
        adf_cv=adf_cv,
        sadf_cv=sadf_cv,
        gsadf_cv=gsadf_cv,
        badf_cv=badf_cv,
        bsadf_cv=bsadf_cv,
        method="Monte Carlo",
        minw=minw,
        n=n,
        iter=nrep,
        N=N,
    )


# -- Co-bubble test (Evripidou, Harvey, Leybourne & Sollis 2022) ------------
#
# Tests whether two series that each contain an explosive episode are
# "co-explosive": whether y_t - alpha - beta*x_{t-lag} is I(0) for some
# lead/lag, i.e. a KPSS-type stationarity test (null = co-explosive) on
# the residuals of y_t regressed on a constant and x_{t-lag} -- the
# opposite testing direction from radf()'s right-tailed unit-root tests.
# Critical values come from a wild bootstrap that reproduces the
# residuals' own heteroskedasticity pattern (the null distribution
# depends on it, so there's no fixed table -- Theorem 1/2).


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


# -- Contagion regression (Greenaway-McGrevy & Phillips 2016) ---------------
#
# Section 2.6's functional (time-varying) coefficient regression: a
# fixed-width rolling-window AR(1) coefficient sequence for a "core"
# series and a "satellite" series y (eq. 1), related by a Nadaraya-Watson
# kernel regression at a chosen delay d (eq. 6), with the bandwidth
# selected by leave-one-out cross-validation (eq. 7). Minimum-viable
# subset: eq. 8's automatic delay search is not implemented -- call
# contagion_reg() once per candidate d and compare fit if that's needed.
# The source paper performs no formal inference on the coefficient itself
# (point estimation/plotting only), so there is no critical value here.


def _contagion_prefix_sums(
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    n1 = len(y) - 1
    x = y[:n1]
    z = y[1 : n1 + 1]
    cx = np.concatenate(([0.0], np.cumsum(x)))
    cx2 = np.concatenate(([0.0], np.cumsum(x**2)))
    cz = np.concatenate(([0.0], np.cumsum(z)))
    cxz = np.concatenate(([0.0], np.cumsum(x * z)))
    return cx, cx2, cz, cxz, n1


def _contagion_fixed_window_beta(y: np.ndarray, S: int) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-window OLS AR(1) slope sequence (eq. 1): window {t = s-S+1,
    ..., s} is S LEVEL observations, i.e. S-1 (response, lag) pairs.
    Returns (t, beta) parallel arrays: t is the window's own end date
    (1-indexed, matching R's convention), for every window-end
    s = S, ..., len(y)."""
    cx, cx2, cz, cxz, n1 = _contagion_prefix_sums(y)
    hi = np.arange(S, n1 + 1)
    lo = hi - (S - 1)
    Sx = cx[hi] - cx[lo]
    Sxx = cx2[hi] - cx2[lo]
    Sz = cz[hi] - cz[lo]
    Sxz = cxz[hi] - cxz[lo]
    n_seg = S - 1
    beta = (n_seg * Sxz - Sx * Sz) / (n_seg * Sxx - Sx**2)
    return hi + 1, beta


def _contagion_kernel_weights(s: np.ndarray, T_len: int, r: np.ndarray, h: float) -> np.ndarray:
    """eq. 6's Gaussian kernel weight K_hs(r) = (1/h)*K((s/T-r)/h), for
    every (position, evaluation point) pair -- rows are positions `s`,
    columns are evaluation points `r`."""
    z = (s[:, None] / T_len - r[None, :]) / h
    return np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi) / h


def _align_shifted(
    t_core: np.ndarray, bcore_c: np.ndarray, t_j: np.ndarray, bj_c: np.ndarray, d: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aligns beta_j at date s with beta_core at date s-d (both centered),
    dropping s where s-d falls outside beta_core's own date range --
    R's named-vector indexing, done here via an index lookup."""
    pos = {int(t): i for i, t in enumerate(t_core)}
    idx = np.array([pos.get(int(t) - d, -1) for t in t_j])
    valid = idx >= 0
    return t_j[valid], bj_c[valid], bcore_c[idx[valid]]


def _contagion_nw_delta2(
    t_core: np.ndarray,
    beta_core: np.ndarray,
    t_j: np.ndarray,
    beta_j: np.ndarray,
    T_len: int,
    r_grid: np.ndarray,
    h: float,
    d: int,
) -> np.ndarray:
    """eq. 6: delta_2j(r; h, d), the Nadaraya-Watson local-constant kernel
    regression coefficient -- a closed-form ratio of two kernel-weighted
    sums (a no-intercept WLS solution). Centering (beta-tilde) uses each
    series' own full range, before any delay shift."""
    bj_c = beta_j - beta_j.mean()
    bcore_c = beta_core - beta_core.mean()
    s, bj_c, core_shift = _align_shifted(t_core, bcore_c, t_j, bj_c, d)

    K = _contagion_kernel_weights(s, T_len, r_grid, h)
    num = K.T @ (bj_c * core_shift)
    den = K.T @ (core_shift**2)
    return num / den


def _contagion_loocv_sse(
    h: float,
    t_core: np.ndarray,
    beta_core: np.ndarray,
    t_j: np.ndarray,
    beta_j: np.ndarray,
    T_len: int,
    d: int,
) -> float:
    """eq. 7's leave-one-out SSE at bandwidth h, delay d: evaluates the
    LOO version of eq. 6 at r = s/m for each available s (excluding the
    point itself from the kernel sum), sums squared prediction errors."""
    bj_c = beta_j - beta_j.mean()
    bcore_c = beta_core - beta_core.mean()
    s, bj_c, core_shift = _align_shifted(t_core, bcore_c, t_j, bj_c, d)

    m = len(s)
    r = s / m
    K = _contagion_kernel_weights(s, T_len, r, h)
    np.fill_diagonal(K, 0.0)
    num = K.T @ (bj_c * core_shift)
    den = K.T @ (core_shift**2)
    pred = (num / den) * core_shift
    return float(np.sum((bj_c - pred) ** 2))


def _golden_section_min(f, lo: float, hi: float, tol: float = 1e-5, max_iter: int = 100) -> float:
    """Bounded 1-D minimization -- exuber's R source uses stats::optimize()
    (golden section + successive parabolic interpolation); numpy has no
    bounded scalar minimizer built in, so this is the plain golden-section
    half of that, sufficient for eq. 7's unimodal-in-practice LOOCV SSE."""
    phi = (np.sqrt(5) - 1) / 2
    a, b = lo, hi
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = f(d)
    return (a + b) / 2


def _contagion_bandwidth_cv(
    t_core: np.ndarray, beta_core: np.ndarray, t_j: np.ndarray, beta_j: np.ndarray,
    T_len: int, d: int,
) -> float:
    """eq. 7: bandwidth via 1-D bounded LOOCV, H_T = [m^-1/2, m^-1/10]."""
    m = len(beta_j)
    H_T = (m ** (-1 / 2), m ** (-1 / 10))
    return _golden_section_min(
        lambda hh: _contagion_loocv_sse(hh, t_core, beta_core, t_j, beta_j, T_len, d),
        H_T[0], H_T[1],
    )


@dataclass
class ContagionRegResult:
    beta_core: np.ndarray
    beta_j: np.ndarray
    h: float
    d: int
    r_grid: np.ndarray
    delta2: np.ndarray
    n: int
    S: int


def contagion_reg(
    y,
    core,
    S: int | None = None,
    d: int = 0,
    h: float | None = None,
    r_grid=None,
) -> ContagionRegResult:
    """Bubble contagion regression (Greenaway-McGrevy & Phillips 2016).

    Estimates the time-varying contagion coefficient: a fixed-window
    rolling AR(1) coefficient sequence for a "core" series and a
    "satellite" series y, related by a Nadaraya-Watson kernel regression
    at a chosen delay d -- how strongly, and how (time-varying), the
    core's local persistence transmits to y, d periods later.

    Minimum-viable subset: the fixed-window AR(1) sequence (eq. 1), the
    Nadaraya-Watson regression at a single supplied d (eq. 6), and
    leave-one-out cross-validated bandwidth selection (eq. 7). eq. 8's
    automatic delay search is not implemented -- call this once per
    candidate d and compare fit. Not a hypothesis test: the source paper
    performs no formal inference on the coefficient (no confidence bands,
    no significance test) -- neither does this, by design.

    y: satellite (dependent) series. core: reference series, same length.
    S: fixed rolling-window width (default floor(0.33*len(y))).
    d: non-negative integer delay (default 0).
    h: bandwidth; None selects it via leave-one-out CV (eq. 7).
    r_grid: evaluation points as fractions of the sample (default 100
    points over [0, 1]).
    """
    y = np.asarray(y, dtype=float)
    core = np.asarray(core, dtype=float)
    if len(y) != len(core):
        raise ValueError("'y' and 'core' must be the same length.")
    if d < 0:
        raise ValueError("d must be non-negative")
    T_len = len(y)
    S = S if S is not None else max(int(0.33 * T_len), 5)
    if S <= 2:
        raise ValueError("S must be greater than 2")
    r_grid = np.linspace(0, 1, 100) if r_grid is None else np.asarray(r_grid, dtype=float)

    t_j, beta_j = _contagion_fixed_window_beta(y, S)
    t_core, beta_core = _contagion_fixed_window_beta(core, S)

    if h is None:
        h = _contagion_bandwidth_cv(t_core, beta_core, t_j, beta_j, T_len, d)
    delta2 = _contagion_nw_delta2(t_core, beta_core, t_j, beta_j, T_len, r_grid, h, d)

    return ContagionRegResult(
        beta_core=beta_core, beta_j=beta_j, h=float(h), d=int(d), r_grid=r_grid,
        delta2=delta2, n=T_len, S=S,
    )
