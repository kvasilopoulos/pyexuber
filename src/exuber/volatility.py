"""Volatility-robust bubble tests. Ports of exuber's R/radf_tt.R,
R/radf_sign.R, R/radf_kp.R, R/radf_sbz.R, R/ssu_test.R -- see
docs/volatility-robustness.md (root repo) for the formulas, papers and
independent validation this ports.

RNG note: uses numpy's Generator, not R's RNG -- see sim.py's module
docstring; a given `seed` will not reproduce the same draws as the R
functions of the same name.
"""

from dataclasses import dataclass

import numpy as np

from exuber.cv import PCNT, RadfCv, _wb_dgp_hlst
from exuber.radf import RadfResult, _to_2d_array, psy_minw

# -- Shared no-intercept recursive-DF machinery (STADF + sign-based) --------


def gls_dfstat_grid(y: np.ndarray, minw: int) -> dict:
    """GLS-demeaned (no-intercept) recursive sup-ADF family, eq. (9) of
    Kurozumi, Skrobotov & Tsarev (2024). `y` is a levels series; internally
    demeaned by its first observation (not OLS-demeaned the way radf()'s
    own unroot()/exubercore regression is). Fully vectorized over the
    (r1, r2) grid via cumulative sums, same construction as exuber's R
    gls_dfstat_grid() -- see that function's own comments for the formula
    derivation. Returns the same badf/bsadf/adf/sadf/gsadf shape radf()
    does, one row per candidate window end (length n - minw)."""
    y = np.asarray(y, dtype=float)
    yc = y - y[0]
    n1 = len(yc) - 1
    dy = np.diff(yc)
    ylag = yc[:n1]

    if minw >= n1:
        raise ValueError("minw must be smaller than the number of observations minus one")

    csxx = np.concatenate(([0.0], np.cumsum(ylag**2)))
    csxy = np.concatenate(([0.0], np.cumsum(ylag * dy)))
    csyy = np.concatenate(([0.0], np.cumsum(dy**2)))

    b_idx = np.arange(minw, n1 + 1)  # window ends
    a_idx = np.arange(1, n1 + 1)  # window starts (exclusive lower bound)

    sxx = csxx[b_idx][:, None] - csxx[a_idx - 1][None, :]
    sxy = csxy[b_idx][:, None] - csxy[a_idx - 1][None, :]
    syy = csyy[b_idx][:, None] - csyy[a_idx - 1][None, :]
    length = (b_idx[:, None] - a_idx[None, :] + 1).astype(float)

    valid = length >= minw
    sxx = np.where(valid, sxx, np.nan)

    with np.errstate(invalid="ignore", divide="ignore"):
        beta = sxy / sxx
        ssr = syy - beta * sxy
        sigma2 = ssr / (length - 1)
        tstat = sxy / np.sqrt(sigma2 * sxx)

    valid = valid & np.isfinite(tstat)
    tstat = np.where(valid, tstat, np.nan)

    badf = tstat[:, 0].copy()
    adf = badf[-1]
    sadf = float(np.nanmax(badf))

    tstat_filled = np.where(valid, tstat, -np.inf)
    best_col = np.argmax(tstat_filled, axis=1)
    bsadf = tstat_filled[np.arange(tstat_filled.shape[0]), best_col]
    gsadf = float(np.max(bsadf))

    return {"badf": badf, "bsadf": bsadf, "adf": float(adf), "sadf": sadf, "gsadf": gsadf}


def _panel(bsadf: np.ndarray) -> tuple[np.ndarray, float]:
    bsadf_panel = bsadf.mean(axis=1)
    return bsadf_panel, float(bsadf_panel.max())


def _grid_to_radf_result(
    x: np.ndarray, minw: int, columns: list[str] | None, transform
) -> RadfResult:
    """Shared driver for radf_tt/radf_sign/radf_sign_dm: apply `transform`
    to each series, run gls_dfstat_grid(), assemble a RadfResult."""
    nr, nc = x.shape
    n_minw = nr - minw
    adf = np.empty(nc)
    sadf = np.empty(nc)
    gsadf = np.empty(nc)
    badf = np.empty((n_minw, nc))
    bsadf = np.empty((n_minw, nc))

    for j in range(nc):
        res = gls_dfstat_grid(transform(x[:, j]), minw)
        badf[:, j] = res["badf"]
        bsadf[:, j] = res["bsadf"]
        adf[j] = res["adf"]
        sadf[j] = res["sadf"]
        gsadf[j] = res["gsadf"]

    bsadf_panel, gsadf_panel = _panel(bsadf)
    return RadfResult(
        adf=adf, badf=badf, sadf=sadf, bsadf=bsadf, gsadf=gsadf,
        bsadf_panel=bsadf_panel, gsadf_panel=gsadf_panel,
        minw=minw, lag=0, n=nr, series_names=columns,
    )


def _mc_cv_grid(
    n: int, minw: int | None, nrep: int, seed: int | None, transform, method: str
) -> RadfCv:
    """Shared driver for radf_tt_cv/radf_sign_cv/radf_sign_dm_cv: simulate
    the pivotal null distribution of gls_dfstat_grid() applied to
    `transform(cumsum(normal))`."""
    minw = minw if minw is not None else psy_minw(n)
    rng = np.random.default_rng(seed)

    n_minw = n - minw
    adf = np.empty(nrep)
    sadf = np.empty(nrep)
    gsadf = np.empty(nrep)
    badf = np.empty((n_minw, nrep))
    bsadf = np.empty((n_minw, nrep))

    for i in range(nrep):
        y = np.cumsum(rng.normal(size=n))
        res = gls_dfstat_grid(transform(y), minw)
        badf[:, i] = res["badf"]
        bsadf[:, i] = res["bsadf"]
        adf[i] = res["adf"]
        sadf[i] = res["sadf"]
        gsadf[i] = res["gsadf"]

    return RadfCv(
        adf_cv=np.quantile(adf, PCNT),
        sadf_cv=np.quantile(sadf, PCNT),
        gsadf_cv=np.quantile(gsadf, PCNT),
        badf_cv=np.quantile(badf, PCNT, axis=1).T,
        bsadf_cv=np.quantile(bsadf, PCNT, axis=1).T,
        method=method, minw=minw, n=n, iter=nrep,
    )


# -- STADF / GSTADF (Kurozumi, Skrobotov & Tsarev 2024) ----------------------


def variance_profile(
    y: np.ndarray, kernel: str = "uniform", h: float | None = None
) -> tuple[np.ndarray, float]:
    """Nonparametric variance profile estimate, eq. (18)-(19) of Kurozumi,
    Skrobotov & Tsarev (2024): local (Nadaraya-Watson-type) no-intercept
    kernel regression of the time-varying AR(1) coefficient, truncated
    residuals, cumulative-sum-of-squares profile. Returns (eta_grid,
    omega2); eta_grid has length Tn + 1 (Tn = len(y) - 1)."""
    if kernel not in ("uniform", "gaussian"):
        raise ValueError("kernel must be 'uniform' or 'gaussian'")
    y = np.asarray(y, dtype=float)
    yc = y - y[0]
    tn = len(yc) - 1
    dy = np.diff(yc)
    ylag = yc[:tn]

    h = h if h is not None else tn ** (-2 / 5)
    idx = np.arange(1, tn + 1)

    def kern(u: np.ndarray) -> np.ndarray:
        if kernel == "uniform":
            return (np.abs(u) <= 1).astype(float)
        return np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi)

    delta = np.empty(tn)
    eps = np.empty(tn)
    for t in range(1, tn + 1):
        w = kern((idx - t) / (tn * h))
        sw_xx = np.sum(w * ylag**2)
        sw_xy = np.sum(w * ylag * dy)
        delta[t - 1] = sw_xy / sw_xx if sw_xx > 0 else 0.0
        eps[t - 1] = dy[t - 1] - delta[t - 1] * ylag[t - 1]

    # Truncation threshold psi_T (footnote 6): max local-window residual sd.
    win = max(2, round(0.1 * tn))
    n_starts = max(1, round(0.9 * tn))
    cbar = max(
        np.std(eps[a : min(a + win + 1, tn)], ddof=1)
        for a in range(n_starts)
        if min(a + win + 1, tn) - a > 1
    )
    psi_t = cbar * tn ** (1 / 7)

    eps_star = np.where(np.abs(eps) >= psi_t, 0.0, eps)
    cs = np.cumsum(eps_star**2)
    total = cs[-1]
    omega2 = total / tn
    eta_grid = np.concatenate(([0.0], cs)) / total
    return eta_grid, omega2


def _time_transform(y: np.ndarray, kernel: str, h: float | None) -> np.ndarray:
    eta_grid, _ = variance_profile(y, kernel=kernel, h=h)
    tn = len(y) - 1
    tgrid = np.linspace(0, 1, tn + 1)
    # Generalized inverse of the monotone, piecewise-linear eta_hat via
    # linear interpolation (exact since eta_hat is piecewise-linear on the
    # observation grid), matching R's approx(eta_grid, tgrid, xout = tgrid).
    s = np.clip(tgrid, 0.0, 1.0)
    g_of_s = np.interp(s, eta_grid, tgrid)
    resampled_idx = np.clip(np.round(g_of_s * tn).astype(int), 0, len(y) - 1)
    return y[resampled_idx]


def radf_tt(
    data, minw: int | None = None, kernel: str = "uniform", h: float | None = None
) -> RadfResult:
    """Time-transformed test for explosive bubbles under non-stationary
    volatility (STADF/GSTADF), Kurozumi, Skrobotov & Tsarev (2024): the
    series is time-deformed using a nonparametric variance-profile
    estimate, after which the usual (homoskedastic) recursive sup-ADF
    critical values apply -- no bootstrap needed. Pair with
    `radf_tt_cv()`.
    """
    x, columns = _to_2d_array(data)
    nr, nc = x.shape
    minw = minw if minw is not None else psy_minw(nr)

    transformed = np.column_stack(
        [_time_transform(x[:, j], kernel, h) for j in range(nc)]
    )
    return _grid_to_radf_result(transformed, minw, columns, lambda col: col)


def radf_tt_cv(
    n: int, minw: int | None = None, nrep: int = 2000, seed: int | None = None
) -> RadfCv:
    """Monte Carlo critical values for `radf_tt()`. Per Theorem 1 of
    Kurozumi, Skrobotov & Tsarev (2024), the null limiting distribution is
    pivotal (free of the volatility process), so this is simulated once
    under a plain random walk rather than per dataset. `sadf_cv` (STADF)
    can be checked against Whitehouse (2019)'s published asymptotic
    triple, quoted in the paper's footnote 4: for minw/n = 0.1, (10%, 5%,
    1%) = (2.319, 2.626, 3.223).
    """
    return _mc_cv_grid(n, minw, nrep, seed, lambda y: y, method="Time-Transformed MC")


# -- Sign-based sGSADF (Harvey, Leybourne & Zu 2020) -------------------------


def sign_transform(y: np.ndarray) -> np.ndarray:
    """C_t := sum_{i<=t} sign(Delta y_i) -- exactly invariant to the
    (even time-varying) volatility of Delta y_t."""
    y = np.asarray(y, dtype=float)
    return np.concatenate(([0.0], np.cumsum(np.sign(np.diff(y)))))


def sign_demean_transform(y: np.ndarray) -> np.ndarray:
    """HLZ (2020)'s second sign-based analogue (HLTZ 2025's s-bar):
    Ctilde_t := sum_{i=2}^t {sign(dy_i) - (i-1)^-1 * sum_{j=2}^i sign(dy_j)}
    -- an expanding-window (not per-window) demeaning, so the demeaning
    term at each i is simply C_i / (i - 1), an O(T) one-time transform."""
    y = np.asarray(y, dtype=float)
    s = np.sign(np.diff(y))
    n = len(s)
    cs = np.cumsum(s)
    return np.concatenate(([0.0], np.cumsum(s - cs / np.arange(1, n + 1))))


def radf_sign(data, minw: int | None = None) -> RadfResult:
    """Sign-based bubble test (sPWY/sPSY), Harvey, Leybourne & Zu (2020):
    the recursive sup-ADF statistic applied to the cumulated sign of the
    first differences, exactly invariant to (even time-varying)
    heteroskedasticity -- no bootstrap needed. Pair with `radf_sign_cv()`.
    """
    x, columns = _to_2d_array(data)
    minw = minw if minw is not None else psy_minw(x.shape[0])
    return _grid_to_radf_result(x, minw, columns, sign_transform)


def radf_sign_cv(
    n: int, minw: int | None = None, nrep: int = 2000, seed: int | None = None
) -> RadfCv:
    """Monte Carlo critical values for `radf_sign()`. Per Theorem 2 of
    Harvey, Leybourne & Zu (2020), pivotal -- simulated once under a plain
    random walk. `sadf_cv`/`gsadf_cv` (minw/n = 0.1) can be checked
    against the paper's Table 1 asymptotic sPWY/sPSY rows: (2.410, 2.734,
    3.248) and (2.933, 3.180, 3.655).
    """
    return _mc_cv_grid(n, minw, nrep, seed, sign_transform, method="Sign-Based MC")


def radf_sign_dm(data, minw: int | None = None) -> RadfResult:
    """Recursively demeaned sign-based bubble test (s-bar-PWY/s-bar-PSY),
    Harvey, Leybourne & Zu (2020)'s second sign-based analogue. Shares
    `radf_sign()`'s exact heteroskedasticity invariance and (per Harvey,
    Leybourne, Tatlow & Zu 2025) its level-shift robustness, under a
    strictly weaker assumption (no zero-median requirement). Pair with
    `radf_sign_dm_cv()`.
    """
    x, columns = _to_2d_array(data)
    minw = minw if minw is not None else psy_minw(x.shape[0])
    return _grid_to_radf_result(x, minw, columns, sign_demean_transform)


def radf_sign_dm_cv(
    n: int, minw: int | None = None, nrep: int = 2000, seed: int | None = None
) -> RadfCv:
    """Monte Carlo critical values for `radf_sign_dm()`. Pivotal like
    `radf_sign_cv()`, simulated once under a plain random walk."""
    return _mc_cv_grid(
        n, minw, nrep, seed, sign_demean_transform, method="Sign-Based MC (demeaned)"
    )


# -- Kernel spot-volatility estimator (shared: kernel-purge + SBZ) ----------


def nw_spot_vol(
    e: np.ndarray, kernel: str = "gaussian", h: float | None = None
) -> tuple[np.ndarray, float]:
    """Nadaraya-Watson kernel smoother of a squared innovation series `e`,
    with leave-one-out cross-validated bandwidth over [1/(2T), 1/6]
    (Harvey, Leybourne & Zu 2019, footnote 2) when `h` is not given.
    Returns (sigma2, h), sigma2 one value per t = 1..T (t/T grid)."""
    if kernel not in ("gaussian", "uniform"):
        raise ValueError("kernel must be 'gaussian' or 'uniform'")
    e = np.asarray(e, dtype=float)
    tn = len(e)
    s = np.arange(2, tn + 1) / tn  # i/T for i = 2..T
    e2 = e[1:] ** 2  # e_i^2 for i = 2..T, aligned with s
    t_grid = np.arange(1, tn + 1) / tn

    def kern(u: np.ndarray) -> np.ndarray:
        if kernel == "gaussian":
            return np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi)
        return (np.abs(u) <= 1).astype(float) / 2

    def spot_vol_at(hh: float, drop_self: bool = False) -> np.ndarray:
        out = np.empty(len(t_grid))
        for j, tg in enumerate(t_grid):
            w = kern((s - tg) / hh)
            if drop_self:
                self_idx = np.abs(s - tg) < np.sqrt(np.finfo(float).eps)
                w = np.where(self_idx, 0.0, w)
            wsum = w.sum()
            out[j] = np.mean(e2) if wsum <= 0 else np.sum(w * e2) / wsum
        return out

    if h is None:
        hl = 1 / (2 * tn)
        hu = 1 / 6
        grid = np.exp(np.linspace(np.log(hl), np.log(hu), 10))
        s_to_tgrid_idx = np.searchsorted(t_grid, s)  # s values sit exactly on t_grid
        cv = np.empty(len(grid))
        for k, hh in enumerate(grid):
            s2_loo = spot_vol_at(hh, drop_self=True)
            cv[k] = np.mean((e2 - s2_loo[s_to_tgrid_idx]) ** 2)
        h = float(grid[np.argmin(cv)])

    sigma2 = spot_vol_at(h, drop_self=False)
    return sigma2, h


def kernel_spot_vol(
    y: np.ndarray, kernel: str = "gaussian", h: float | None = None
) -> tuple[np.ndarray, float]:
    """Nonparametric spot-volatility estimator, eq. (6) of Harvey,
    Leybourne & Zu (2019): `nw_spot_vol()` applied to the raw series'
    first differences."""
    return nw_spot_vol(np.diff(np.asarray(y, dtype=float)), kernel=kernel, h=h)


# -- Kernel-purge test (Harvey, Leybourne, Taylor & Zu 2024) -----------------


def kernel_purge(y: np.ndarray, kernel: str = "gaussian", h: float | None = None) -> np.ndarray:
    """Kernel-purged transform (eq. 4-5): x_t = cumsum(Delta y_t /
    sigma_hat_t). Feed the result to `radf()` -- the purged statistic's
    null distribution is proven identical to the standard homoskedastic
    GSADF null (Theorem 1/Remark 3.2 of Harvey, Leybourne, Taylor & Zu
    2024), so no new critical-value machinery is needed."""
    y = np.asarray(y, dtype=float)
    tn = len(y) - 1
    h = h if h is not None else 0.1 * tn ** (-0.25)
    sigma2, _ = kernel_spot_vol(y, kernel=kernel, h=h)
    return np.cumsum(np.diff(y) / np.sqrt(sigma2))


def radf_kp(
    data, minw: int | None = None, kernel: str = "gaussian", h: float | None = None
) -> RadfResult:
    """Kernel-purged heteroskedasticity-robust PSY test, Harvey, Leybourne,
    Taylor & Zu (2024): "purges" unconditional heteroskedasticity by
    cumulating the series' first differences after dividing each by a
    kernel spot-volatility estimate, then runs the ordinary
    (with-intercept) `radf()` on the purged series. `radf_mc_cv()` applies
    directly to the result (see module docstring / Details above)."""
    from exuber.radf import radf

    x, columns = _to_2d_array(data)
    nc = x.shape[1]
    purged = np.column_stack([kernel_purge(x[:, j], kernel=kernel, h=h) for j in range(nc)])
    if columns is not None:
        result = radf(purged, minw=minw)
        result.series_names = columns
        return result
    return radf(purged, minw=minw)


# -- SBZ: WLS + kernel volatility (Harvey, Leybourne & Zu 2019) -------------


def wls_dfstat_grid(y: np.ndarray, sigma2: np.ndarray, minw: int) -> dict:
    """Feasible BZ statistic family (supBZ), Harvey, Leybourne & Zu
    (2019): a WLS (1/sigma2-weighted) no-intercept recursive
    Dickey-Fuller statistic on the GLS-demeaned series. `sigma2` has
    length len(y) - 1, one spot-variance estimate per differenced
    observation."""
    y = np.asarray(y, dtype=float)
    yc = y - y[0]
    n1 = len(yc) - 1
    dy = np.diff(yc)
    ylag = yc[:n1]
    w = 1.0 / sigma2

    csxx = np.concatenate(([0.0], np.cumsum(w * ylag**2)))
    csxy = np.concatenate(([0.0], np.cumsum(w * ylag * dy)))

    b_idx = np.arange(minw, n1 + 1)
    a_idx = np.arange(1, n1 + 1)

    sxx = csxx[b_idx][:, None] - csxx[a_idx - 1][None, :]
    sxy = csxy[b_idx][:, None] - csxy[a_idx - 1][None, :]
    length = (b_idx[:, None] - a_idx[None, :] + 1).astype(float)

    valid = length >= minw
    sxx = np.where(valid, sxx, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        tstat = sxy / np.sqrt(sxx)
    valid = valid & np.isfinite(tstat)
    tstat = np.where(valid, tstat, np.nan)

    badf = tstat[:, 0].copy()
    adf = badf[-1]
    sadf = float(np.nanmax(badf))

    tstat_filled = np.where(valid, tstat, -np.inf)
    best_col = np.argmax(tstat_filled, axis=1)
    bsadf = tstat_filled[np.arange(tstat_filled.shape[0]), best_col]
    gsadf = float(np.max(bsadf))

    return {"badf": badf, "bsadf": bsadf, "adf": float(adf), "sadf": sadf, "gsadf": gsadf}


def radf_sbz(
    data, minw: int | None = None, kernel: str = "gaussian", h: float | None = None
) -> RadfResult:
    """WLS/kernel-volatility bubble statistic (supBZ), Harvey, Leybourne &
    Zu (2019). Pair with `radf_sbz_cv()` for critical values, or
    `radf_sbz_union()` for the paper's own union-of-rejections test
    against the classic supDF statistic."""
    x, columns = _to_2d_array(data)
    nr, nc = x.shape
    minw = minw if minw is not None else psy_minw(nr)
    n_minw = nr - minw

    adf = np.empty(nc)
    sadf = np.empty(nc)
    gsadf = np.empty(nc)
    badf = np.empty((n_minw, nc))
    bsadf = np.empty((n_minw, nc))
    for j in range(nc):
        sigma2, _ = kernel_spot_vol(x[:, j], kernel=kernel, h=h)
        res = wls_dfstat_grid(x[:, j], sigma2, minw)
        badf[:, j] = res["badf"]
        bsadf[:, j] = res["bsadf"]
        adf[j] = res["adf"]
        sadf[j] = res["sadf"]
        gsadf[j] = res["gsadf"]

    bsadf_panel, gsadf_panel = _panel(bsadf)
    return RadfResult(
        adf=adf, badf=badf, sadf=sadf, bsadf=bsadf, gsadf=gsadf,
        bsadf_panel=bsadf_panel, gsadf_panel=gsadf_panel,
        minw=minw, lag=0, n=nr, series_names=columns,
    )


def radf_sbz_cv(
    data,
    minw: int | None = None,
    nboot: int = 499,
    kernel: str = "gaussian",
    h: float | None = None,
    seed: int | None = None,
) -> RadfCv:
    """Wild bootstrap (HLST 2016, the same algorithm as `radf_wb_cv()`)
    critical values for `radf_sbz()`'s supBZ statistic -- supBZ's own null
    distribution depends on the WLS weighting, so it needs its own
    data-dependent critical values, unlike the pivotal sign-based/STADF
    tests."""
    x, columns = _to_2d_array(data)
    nr, nc = x.shape
    minw = minw if minw is not None else psy_minw(nr)
    n_minw = nr - minw
    rng = np.random.default_rng(seed)

    adf_cv = np.empty((nc, 3))
    sadf_cv = np.empty((nc, 3))
    gsadf_cv = np.empty((nc, 3))
    badf_cv = np.empty((n_minw, 3, nc))
    bsadf_cv = np.empty((n_minw, 3, nc))

    for j in range(nc):
        yj = x[:, j]
        sigma2, _ = kernel_spot_vol(yj, kernel=kernel, h=h)

        boot_adf = np.empty(nboot)
        boot_sadf = np.empty(nboot)
        boot_gsadf = np.empty(nboot)
        boot_badf = np.empty((n_minw, nboot))
        boot_bsadf = np.empty((n_minw, nboot))
        for b in range(nboot):
            ystar = _wb_dgp_hlst(yj, dist_rad=False, rng=rng)
            res = wls_dfstat_grid(ystar, sigma2, minw)
            boot_adf[b] = res["adf"]
            boot_sadf[b] = res["sadf"]
            boot_gsadf[b] = res["gsadf"]
            boot_badf[:, b] = res["badf"]
            boot_bsadf[:, b] = res["bsadf"]

        adf_cv[j] = np.quantile(boot_adf, PCNT)
        sadf_cv[j] = np.quantile(boot_sadf, PCNT)
        gsadf_cv[j] = np.quantile(boot_gsadf, PCNT)
        badf_cv[:, :, j] = np.quantile(boot_badf, PCNT, axis=1).T
        bsadf_cv[:, :, j] = np.quantile(boot_bsadf, PCNT, axis=1).T

    return RadfCv(
        adf_cv=adf_cv, sadf_cv=sadf_cv, gsadf_cv=gsadf_cv,
        badf_cv=badf_cv, bsadf_cv=bsadf_cv,
        method="Wild Bootstrap (SBZ)", minw=minw, n=nr, iter=nboot,
        series_names=columns,
    )


@dataclass
class RadfSbzUnion:
    """Output of `radf_sbz_union()`: paired supDF/supBZ/U statistics,
    critical values and bootstrap p-values, one entry per series. Not a
    `RadfResult` -- U's value needs the joint bootstrap, see
    `radf_sbz_union()`'s own docstring."""

    supDF: np.ndarray
    supBZ: np.ndarray
    U: np.ndarray
    supDF_cv: np.ndarray
    supBZ_cv: np.ndarray
    U_cv: np.ndarray
    p_supDF: np.ndarray
    p_supBZ: np.ndarray
    p_U: np.ndarray
    minw: int
    n: int
    iter: int
    series_names: list[str] | None = None


def radf_sbz_union(
    data,
    minw: int | None = None,
    nboot: int = 499,
    kernel: str = "gaussian",
    h: float | None = None,
    seed: int | None = None,
) -> RadfSbzUnion:
    """SBZ weighted-least-squares bubble test with union-of-rejections,
    Harvey, Leybourne & Zu (2019): runs the HLST (2016) wild bootstrap
    *jointly* on the classic sup-ADF statistic (supDF) and the WLS/
    kernel-volatility statistic (supBZ), and combines them into the
    union statistic `U := max(supDF, (qDF/qBZ) * supBZ)`, using the
    bootstrap's own 95% quantiles for the qDF/qBZ scaling ratio (the
    paper's Section 2.3). The joint bootstrap draw is required for the
    union's size guarantee (Theorem 3) to hold, which is why this stays
    one bundled function rather than a statistic/critical-value pair --
    see `radf_sbz()`/`radf_sbz_cv()` for the supBZ-only route."""
    from exuber._unroot import unroot

    x, columns = _to_2d_array(data)
    nr, nc = x.shape
    minw = minw if minw is not None else psy_minw(nr)
    rng = np.random.default_rng(seed)
    pcnt_arr = np.array(PCNT)

    supDF_cv = np.empty((nc, 3))
    supBZ_cv = np.empty((nc, 3))
    U_cv = np.empty((nc, 3))
    p_supDF = np.empty(nc)
    p_supBZ = np.empty(nc)
    p_U = np.empty(nc)
    supDF_obs = np.empty(nc)
    supBZ_obs = np.empty(nc)
    U_obs = np.empty(nc)

    for j in range(nc):
        yj = x[:, j]
        sigma2, _ = kernel_spot_vol(yj, kernel=kernel, h=h)

        # Classic supDF via exubercore's own radf() engine, the same
        # pointer/statistic-slot convention _radf_wb_hlst() (cv.py) uses.
        from . import _core

        yxmat = unroot(yj)
        pointer = nr - minw
        result = _core.radf_stat(yxmat, minw, 0)
        supDF_obs[j] = result[pointer + 1]  # sadf slot
        supBZ_obs[j] = wls_dfstat_grid(yj, sigma2, minw)["sadf"]

        boot_df = np.empty(nboot)
        boot_bz = np.empty(nboot)
        for b in range(nboot):
            ystar = _wb_dgp_hlst(yj, dist_rad=False, rng=rng)
            yxstar = unroot(ystar)
            res_star = _core.radf_stat(yxstar, minw, 0)
            boot_df[b] = res_star[pointer + 1]
            boot_bz[b] = wls_dfstat_grid(ystar, sigma2, minw)["sadf"]

        supDF_cv[j] = np.quantile(boot_df, pcnt_arr)
        supBZ_cv[j] = np.quantile(boot_bz, pcnt_arr)
        p_supDF[j] = np.mean(boot_df > supDF_obs[j])
        p_supBZ[j] = np.mean(boot_bz > supBZ_obs[j])

        ratio = supDF_cv[j, 1] / supBZ_cv[j, 1]  # index 1 == 95%
        U_boot = np.maximum(boot_df, ratio * boot_bz)
        U_obs[j] = max(supDF_obs[j], ratio * supBZ_obs[j])
        U_cv[j] = np.quantile(U_boot, pcnt_arr)
        p_U[j] = np.mean(U_boot > U_obs[j])

    return RadfSbzUnion(
        supDF=supDF_obs, supBZ=supBZ_obs, U=U_obs,
        supDF_cv=supDF_cv, supBZ_cv=supBZ_cv, U_cv=U_cv,
        p_supDF=p_supDF, p_supBZ=p_supBZ, p_U=p_U,
        minw=minw, n=nr, iter=nboot, series_names=columns,
    )


# -- SSU: stochastic explosive-coefficient test (Kurozumi & Nishi 2025) -----

_SSU_LEVELS = (90, 95, 99)
_SSU_CRIT = (2.90, 3.30, 4.20)


def ssu_q(sig_lvl: float) -> float:
    """Kurozumi & Nishi (2025) Table I's published SSU asymptotic critical
    value (their own 10,000-rep Monte Carlo) -- no simulation needed."""
    for lvl, crit in zip(_SSU_LEVELS, _SSU_CRIT, strict=True):
        if abs(sig_lvl - lvl) < 1e-8:
            return crit
    raise ValueError(
        f"sig_lvl must be one of {_SSU_LEVELS} (Kurozumi & Nishi (2025)'s "
        "Table I only tabulates these significance levels)"
    )


def ssu_prefix_sums(y: np.ndarray) -> dict:
    """All cumulative sums SSU's closed form needs, built from x1 =
    y_{t-1} (level lag) and d1 = Delta y_t (difference); every term in the
    ADF regression (eq. 6), the SSU regression (eq. 7), and the
    bias-correction cross-moment reduces to a window sum of one of these
    twelve products."""
    y = np.asarray(y, dtype=float)
    n1 = len(y) - 1
    x1 = y[:n1]
    d1 = y[1 : n1 + 1] - x1

    def mk(v: np.ndarray) -> np.ndarray:
        return np.concatenate(([0.0], np.cumsum(v)))

    return {
        "n1": n1,
        "x1": mk(x1), "x1_2": mk(x1**2), "x1_3": mk(x1**3), "x1_4": mk(x1**4),
        "d1": mk(d1), "d1_2": mk(d1**2), "d1_3": mk(d1**3), "d1_4": mk(d1**4),
        "d1x1": mk(d1 * x1),
        "d1_2x1_2": mk(d1**2 * x1**2),
        "d1x1_2": mk(d1 * x1**2),
        "x1d1_2": mk(x1 * d1**2),
    }


def ssu_stat_path(ps: dict, hi_idx: np.ndarray) -> np.ndarray:
    """t^{omega,c}_{0,r2} (SSU's own r1 = 0, fixed) for every candidate r2
    in `hi_idx` -- see exuber's R/ssu_test.R for the derivation of the
    bilinear cross-moment expansion this implements."""
    hi_idx = np.asarray(hi_idx)

    def s(name: str) -> np.ndarray:
        return ps[name][hi_idx]

    length = hi_idx.astype(float)

    sx1 = s("x1")
    sx1x1 = s("x1_2")
    sd1 = s("d1")
    sd1x1 = s("d1x1")
    sd1d1 = s("d1_2")
    delta_hat = (length * sd1x1 - sx1 * sd1) / (length * sx1x1 - sx1**2)
    mu1_hat = (sd1 - delta_hat * sx1) / length
    ssr6 = sd1d1 - mu1_hat * sd1 - delta_hat * sd1x1
    sigma2_eps = ssr6 / (length - 2)

    sx2 = sx1x1
    sx2x2 = s("x1_4")
    sd2 = sd1d1
    sd2x2 = s("d1_2x1_2")
    sd2d2 = s("d1_4")
    omega_hat = (length * sd2x2 - sx2 * sd2) / (length * sx2x2 - sx2**2)
    mu2_hat = (sd2 - omega_hat * sx2) / length
    ssr7 = sd2d2 - mu2_hat * sd2 - omega_hat * sd2x2
    sigma2_eta = ssr7 / (length - 2)
    sx2x2_c = sx2x2 - sx2**2 / length
    t_omega = omega_hat / np.sqrt(sigma2_eta / sx2x2_c)

    sd1d2 = s("d1_3")
    sd1x2 = s("d1x1_2")
    sx1x2 = s("x1_3")
    sx1d2 = s("x1d1_2")
    sum_eh = (
        sd1d2 - mu2_hat * sd1 - omega_hat * sd1x2 - mu1_hat * sd2
        + length * mu1_hat * mu2_hat + mu1_hat * omega_hat * sx2
        - delta_hat * sx1d2 + delta_hat * mu2_hat * sx1 + delta_hat * omega_hat * sx1x2
    )
    sigma2_epseta = sum_eh / (length - 1)

    sigma_eps = np.sqrt(sigma2_eps)
    sigma_eta = np.sqrt(sigma2_eta)
    psi_hat = sigma2_epseta / (sigma_eps * sigma_eta)

    ybar2 = sx2 / length
    num_corr = sd1x2 - ybar2 * sd1
    den_corr = np.sqrt(sx2x2_c)

    correction = (psi_hat / sigma_eps) * num_corr / den_corr
    return (t_omega - correction) / np.sqrt(1 - psi_hat**2)


@dataclass
class SsuTestResult:
    """Output of `ssu_test()`: the recursive SSU statistic path, its
    published critical value, and the sup-statistic/detection outcome per
    series."""

    stat: np.ndarray  # (n - minw, nc)
    sadf: np.ndarray  # (nc,)
    crit: float
    detected: np.ndarray  # bool, (nc,)
    minw: int
    n: int
    sig_lvl: float
    series_names: list[str] | None = None


def ssu_test(data, minw: int | None = None, sig_lvl: float = 95) -> SsuTestResult:
    """Stochastic unit root bubble test (SSU), Kurozumi & Nishi (2025):
    tests for a *stochastic* (rather than deterministic) unit root in the
    squared first differences, bias-corrected against its dependence on
    the correlation with the plain ADF regression's innovations. Only the
    single-recursion SSU statistic is implemented (not GSSU, CUSUM/
    CUSUM-SQ, or the union-of-rejections procedure -- see
    docs/volatility-robustness.md, root repo, for the minimum-viable-
    subset scoping). The critical value is Table I's published constant,
    no simulation needed."""
    x, columns = _to_2d_array(data)
    n, nc = x.shape
    minw = minw if minw is not None else psy_minw(n)
    crit = ssu_q(sig_lvl)

    hi_idx = np.arange(minw, n)  # R: minw:(n - 1L), 1-indexed -> same values here
    stat = np.empty((len(hi_idx), nc))
    for j in range(nc):
        ps = ssu_prefix_sums(x[:, j])
        stat[:, j] = ssu_stat_path(ps, hi_idx)

    sadf = stat.max(axis=0)
    detected = sadf > crit

    return SsuTestResult(
        stat=stat, sadf=sadf, crit=crit, detected=detected,
        minw=minw, n=n, sig_lvl=sig_lvl, series_names=columns,
    )
