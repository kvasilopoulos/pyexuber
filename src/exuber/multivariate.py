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
