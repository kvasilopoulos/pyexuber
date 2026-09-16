"""Volatility-robust bubble tests. Port of exuber's R/radf_tt.R (STADF/
GSTADF, Kurozumi, Skrobotov & Tsarev 2024) so far -- see
docs/volatility-robustness.md (root repo) for the formulas, papers and
independent validation this ports. More volatility-robust tests
(kernel-purge, SBZ, sign-based, SSU) land here in follow-up commits.

RNG note: uses numpy's Generator, not R's RNG -- see sim.py's module
docstring; a given `seed` will not reproduce the same draws as the R
functions of the same name.
"""

import numpy as np

from exuber.cv import PCNT, RadfCv
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
