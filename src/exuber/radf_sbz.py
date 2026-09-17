"""SBZ: WLS + kernel volatility bubble test (Harvey, Leybourne & Zu
2019). Ported from exuber's R/radf_sbz.R -- see
docs/volatility-robustness.md (root repo) for the formulas, papers and
independent validation this ports.

RNG note: uses numpy's Generator, not R's RNG -- see sim.py's module
docstring; a given `seed` will not reproduce the same draws as R's
radf_sbz_cv()/radf_sbz_union().
"""

from dataclasses import dataclass

import numpy as np

from exuber._kernel_vol import kernel_spot_vol
from exuber._unroot import unroot
from exuber.cv import PCNT, RadfCv, _wb_dgp_hlst
from exuber.radf import RadfResult, _to_2d_array, psy_minw


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

    bsadf_panel = bsadf.mean(axis=1)
    gsadf_panel = float(bsadf_panel.max())
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
