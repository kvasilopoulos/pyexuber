"""Monte Carlo / bootstrap critical values. Ports of exuber's R/radf_mc.R,
R/radf_wb.R (both the HLST and Phillips-Shi wild bootstrap variants) and
R/radf_sb.R (sieve bootstrap).

RNG note: uses numpy's Generator, not R's RNG -- a given `seed` will not
reproduce the same draws as the R functions of the same name. See
exubercore's RNG design note / sim.py's module docstring. lag_select()/
adf_res() (_lagselect.py) are the exception: they're deterministic (no
RNG) and verified bit-for-bit against R -- see
docs/replication/volatility-robustness/radf_wb_ps_validation.R.
"""

from dataclasses import dataclass

import numpy as np

from exuber._lagselect import AdfRes, adf_res, lag_select
from exuber._unroot import _embed, unroot
from exuber.radf import _to_2d_array, psy_minw

PCNT = (0.9, 0.95, 0.99)


@dataclass
class RadfCv:
    adf_cv: np.ndarray
    sadf_cv: np.ndarray
    gsadf_cv: np.ndarray
    badf_cv: np.ndarray
    bsadf_cv: np.ndarray
    method: str
    minw: int
    n: int
    iter: int
    lag: int = 0
    series_names: list[str] | None = None


@dataclass
class RadfDistr:
    adf_distr: np.ndarray
    sadf_distr: np.ndarray
    gsadf_distr: np.ndarray
    method: str
    minw: int
    n: int
    iter: int


@dataclass
class RadfSbCv:
    """radf_sb_cv()'s output shape: panel-only (cross-sectional mean of the
    per-series BSADF paths, then quantiles of that and of its max) -- no
    per-series adf_cv/sadf_cv/badf_cv, unlike RadfCv."""

    gsadf_panel_cv: np.ndarray
    bsadf_panel_cv: np.ndarray
    method: str
    minw: int
    n: int
    iter: int
    lag: int
    series_names: list[str] | None = None


@dataclass
class RadfSbDistr:
    gsadf_panel_distr: np.ndarray
    method: str
    minw: int
    n: int
    iter: int
    lag: int


# -- Monte Carlo ------------------------------------------------------------


def _radf_mc(n: int, minw: int | None = None, nrep: int = 1000, seed: int | None = None) -> dict:
    from . import _core  # lazy: see radf.py's radf() for why

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
        yxmat = unroot(y)
        result = _core.radf_stat(yxmat, minw, 0)
        badf[:, i] = result[:n_minw]
        adf[i] = result[n_minw]
        sadf[i] = result[n_minw + 1]
        gsadf[i] = result[n_minw + 2]
        bsadf[:, i] = result[n_minw + 3 :]

    return {"adf": adf, "sadf": sadf, "gsadf": gsadf, "badf": badf, "bsadf": bsadf, "minw": minw}


def radf_mc_cv(
    n: int, minw: int | None = None, nrep: int = 1000, seed: int | None = None
) -> RadfCv:
    """Monte Carlo critical values for the recursive unit root tests."""
    r = _radf_mc(n, minw, nrep, seed)

    adf_cv = np.quantile(r["adf"], PCNT)
    sadf_cv = np.quantile(r["sadf"], PCNT)
    gsadf_cv = np.quantile(r["gsadf"], PCNT)

    # BSADF cv: quantiles (across replications) of the cumulative max of the
    # BADF path -- matches R's apply(badf, 2, cummax) |> apply(1, quantile).
    bsadf_cv = np.quantile(np.maximum.accumulate(r["badf"], axis=0), PCNT, axis=1).T

    # BADF cv is NOT simulated -- exuber hardcodes the PWY asymptotic values
    # here, constant across the window (same as the R source).
    asy_adf_crit = np.array([-0.44, -0.08, 0.6])
    badf_cv = np.tile(asy_adf_crit, (r["badf"].shape[0], 1))

    return RadfCv(
        adf_cv=adf_cv, sadf_cv=sadf_cv, gsadf_cv=gsadf_cv, badf_cv=badf_cv, bsadf_cv=bsadf_cv,
        method="Monte Carlo", minw=r["minw"], n=n, iter=nrep,
    )


def radf_mc_distr(
    n: int, minw: int | None = None, nrep: int = 1000, seed: int | None = None
) -> RadfDistr:
    """Monte Carlo distribution of the ADF/SADF/GSADF statistics."""
    r = _radf_mc(n, minw, nrep, seed)
    return RadfDistr(
        adf_distr=r["adf"], sadf_distr=r["sadf"], gsadf_distr=r["gsadf"],
        method="Monte Carlo", minw=r["minw"], n=n, iter=nrep,
    )


# -- Wild bootstrap (Harvey, Leybourne, Sollis & Taylor 2016) ---------------


def _wb_dgp_hlst(
    y: np.ndarray, dist_rad: bool, rng: np.random.Generator, dist_skew: bool = False
) -> np.ndarray:
    dy = np.diff(y)
    nr = len(dy)
    if dist_skew:
        # Hafner (2020), Step 1: w = u/sqrt(2) + (v^2-1)/2, u,v ~ iid N(0,1)
        # independent -- E[w]=0, E[w^2]=1, E[w^3]=1, a fixed right-skewed
        # multiplier for series with right-skewed return distributions
        # (crypto in the source paper), vs. dist_rad's symmetric two-point
        # multiplier.
        u = rng.normal(size=nr)
        v = rng.normal(size=nr)
        w = u / np.sqrt(2) + (v**2 - 1) / 2
    elif dist_rad:
        w = rng.choice([-1.0, 1.0], size=nr)
    else:
        w = rng.normal(size=nr)
    estar = np.cumsum(w * dy)
    return np.concatenate(([0.0], estar))


def _radf_wb_hlst(
    data,
    minw: int | None = None,
    nboot: int = 500,
    dist_rad: bool = False,
    dist_skew: bool = False,
    seed: int | None = None,
) -> dict:
    if dist_rad and dist_skew:
        raise ValueError("only one of 'dist_rad' and 'dist_skew' may be True")

    from . import _core  # lazy: see radf.py's radf() for why

    y, columns = _to_2d_array(data)
    nr, nc = y.shape
    minw = minw if minw is not None else psy_minw(nr)
    pointer = nr - minw
    rng = np.random.default_rng(seed)

    adf = np.empty((nboot, nc))
    sadf = np.empty((nboot, nc))
    gsadf = np.empty((nboot, nc))
    badf = np.empty((pointer, nboot, nc))
    bsadf = np.empty((pointer, nboot, nc))

    for j in range(nc):
        for i in range(nboot):
            ystar = _wb_dgp_hlst(y[:, j], dist_rad, rng, dist_skew)
            yxmat = unroot(ystar)
            result = _core.radf_stat(yxmat, minw, 0)
            badf[:, i, j] = result[:pointer]
            adf[i, j] = result[pointer]
            sadf[i, j] = result[pointer + 1]
            gsadf[i, j] = result[pointer + 2]
            bsadf[:, i, j] = result[pointer + 3 :]

    return {
        "adf": adf, "sadf": sadf, "gsadf": gsadf, "badf": badf, "bsadf": bsadf,
        "minw": minw, "n": nr, "series_names": columns,
    }


def radf_wb_cv(
    data,
    minw: int | None = None,
    nboot: int = 500,
    dist_rad: bool = False,
    dist_skew: bool = False,
    seed: int | None = None,
) -> RadfCv:
    """Wild bootstrap critical values (Harvey, Leybourne, Sollis & Taylor 2016).

    `dist_skew=True` uses Hafner (2020)'s fixed right-skewed multiplier
    distribution (`w = u/sqrt(2) + (v^2-1)/2`, u,v iid N(0,1)) instead of
    the default standard normal or (`dist_rad=True`) Rademacher one --
    appropriate when the series' own return distribution is notably
    right-skewed (e.g. cryptocurrency returns, the paper's own
    application). At most one of `dist_rad`/`dist_skew` may be True.
    """
    r = _radf_wb_hlst(data, minw, nboot, dist_rad, dist_skew, seed)

    adf_cv = np.quantile(r["adf"], PCNT, axis=0).T
    sadf_cv = np.quantile(r["sadf"], PCNT, axis=0).T
    gsadf_cv = np.quantile(r["gsadf"], PCNT, axis=0).T
    badf_cv = np.moveaxis(np.quantile(r["badf"], PCNT, axis=1), 0, 1)
    bsadf_cv = np.moveaxis(np.quantile(r["bsadf"], PCNT, axis=1), 0, 1)

    return RadfCv(
        adf_cv=adf_cv, sadf_cv=sadf_cv, gsadf_cv=gsadf_cv, badf_cv=badf_cv, bsadf_cv=bsadf_cv,
        method="Wild Bootstrap", minw=r["minw"], n=r["n"], iter=nboot,
        series_names=r["series_names"],
    )


def radf_wb_distr(
    data,
    minw: int | None = None,
    nboot: int = 500,
    dist_rad: bool = False,
    dist_skew: bool = False,
    seed: int | None = None,
) -> RadfDistr:
    """Wild bootstrap distribution of the ADF/SADF/GSADF statistics. See
    `radf_wb_cv()` for `dist_skew`."""
    r = _radf_wb_hlst(data, minw, nboot, dist_rad, dist_skew, seed)
    return RadfDistr(
        adf_distr=r["adf"], sadf_distr=r["sadf"], gsadf_distr=r["gsadf"],
        method="Wild Bootstrap", minw=r["minw"], n=r["n"], iter=nboot,
    )


# -- Wild bootstrap, Phillips & Shi (2020) variant ---------------------------


def _wb_dgp_ps(
    y: np.ndarray, fit: AdfRes, adflag: int, rng: np.random.Generator, tb: int | None
) -> np.ndarray:
    """Port of R's radf_wb_dgp_ps(): resamples the residuals of adf_res()'s
    null AR(adflag) fit to build one bootstrap replicate of y.

    Faithful port note: for adflag == 0, R's loop `for (i in
    (adflag+1):(nr-1))` never assigns `dyb[nr]` (R is 1-indexed, so the
    last element of `dyb` is left at its initial 0) -- reproduced here
    rather than "fixed", since it's exuber's actual bootstrap DGP.
    """
    beta, eps = fit.beta, fit.res
    dy = np.diff(y)
    nr = len(dy) if tb is None else tb - 1

    r_n = rng.integers(0, nr - adflag, size=nr - adflag)
    wn = rng.normal(size=nr)

    dyb = np.zeros(nr)
    if adflag > 0:
        dyb[:adflag] = dy[:adflag]
        x = np.zeros((nr - 1, adflag))
        for i in range(adflag, nr - 1):
            for k in range(1, adflag + 1):
                x[i, k - 1] = dyb[i - k]
            dyb[i] = x[i, :] @ beta[1:] + wn[i - adflag] * eps[r_n[i - adflag]]
    else:
        for i in range(0, nr - 1):
            dyb[i] = wn[i] * eps[r_n[i]]

    return np.cumsum(np.concatenate(([y[0]], dyb)))


def _radf_wb_ps(
    data,
    minw: int | None = None,
    nboot: int = 500,
    adflag: int = 0,
    type: str = "fixed",
    tb: int | None = None,
    seed: int | None = None,
) -> dict:
    from . import _core  # lazy: see radf.py's radf() for why

    y, columns = _to_2d_array(data)
    nr_full, nc = y.shape
    minw = minw if minw is not None else psy_minw(nr_full)
    nr = tb if tb is not None else nr_full
    pointer = nr - minw
    rng = np.random.default_rng(seed)

    adf = np.empty((nboot, nc))
    sadf = np.empty((nboot, nc))
    gsadf = np.empty((nboot, nc))
    badf = np.empty((pointer, nboot, nc))
    bsadf = np.empty((pointer, nboot, nc))

    for j in range(nc):
        # adf_res() is deterministic in y[:, j] -- fit once, reuse across
        # bootstrap draws (R recomputes it every draw for the same result).
        fit = adf_res(y[:, j], adflag=adflag, type=type)
        for i in range(nboot):
            ystar = _wb_dgp_ps(y[:, j], fit, adflag, rng, tb)
            yxmat = unroot(ystar)
            result = _core.radf_stat(yxmat, minw, 0)
            badf[:, i, j] = result[:pointer]
            adf[i, j] = result[pointer]
            sadf[i, j] = result[pointer + 1]
            gsadf[i, j] = result[pointer + 2]
            bsadf[:, i, j] = result[pointer + 3 :]

    return {
        "adf": adf, "sadf": sadf, "gsadf": gsadf, "badf": badf, "bsadf": bsadf,
        "minw": minw, "n": nr_full, "pointer": pointer, "series_names": columns,
    }


def radf_wb_ps_cv(
    data,
    minw: int | None = None,
    nboot: int = 500,
    adflag: int = 0,
    type: str = "fixed",
    tb: int | None = None,
    seed: int | None = None,
) -> RadfCv:
    """Wild bootstrap critical values, Phillips & Shi (2020) variant: fit a
    null AR model, resample its residuals. Unlike radf_wb_cv()'s HLST
    multiplier bootstrap, this supports a training-window boundary `tb`
    (what monitor() would use it for)."""
    r = _radf_wb_ps(data, minw, nboot, adflag, type, tb, seed)

    adf_cv = np.quantile(r["adf"], PCNT, axis=0).T
    sadf_cv = np.quantile(r["sadf"], PCNT, axis=0).T
    gsadf_cv = np.quantile(r["gsadf"], PCNT, axis=0).T
    badf_cv = np.moveaxis(np.quantile(r["badf"], PCNT, axis=1), 0, 1)
    bsadf_cv = np.moveaxis(np.quantile(r["bsadf"], PCNT, axis=1), 0, 1)

    if tb is not None:
        # Training-window mode: badf/bsadf cv collapse to the (constant)
        # adf/gsadf cv repeated over the full-sample pointer length.
        full_pointer = r["n"] - r["minw"]
        nc = sadf_cv.shape[0]
        badf_cv = np.empty((full_pointer, 3, nc))
        bsadf_cv = np.empty((full_pointer, 3, nc))
        for i in range(nc):
            badf_cv[:, :, i] = sadf_cv[i]
            bsadf_cv[:, :, i] = gsadf_cv[i]

    return RadfCv(
        adf_cv=adf_cv, sadf_cv=sadf_cv, gsadf_cv=gsadf_cv, badf_cv=badf_cv, bsadf_cv=bsadf_cv,
        method="Wild Bootstrap (Phillips-Shi)", minw=r["minw"], n=r["n"], iter=nboot,
        series_names=r["series_names"],
    )


def radf_wb_ps_distr(
    data,
    minw: int | None = None,
    nboot: int = 500,
    adflag: int = 0,
    type: str = "fixed",
    tb: int | None = None,
    seed: int | None = None,
) -> RadfDistr:
    """Distribution of the ADF/SADF/GSADF statistics under the Phillips &
    Shi (2020) wild bootstrap variant."""
    r = _radf_wb_ps(data, minw, nboot, adflag, type, tb, seed)
    return RadfDistr(
        adf_distr=r["adf"], sadf_distr=r["sadf"], gsadf_distr=r["gsadf"],
        method="Wild Bootstrap (Phillips-Shi)", minw=r["minw"], n=r["n"], iter=nboot,
    )


# -- Sieve bootstrap (Pavlidis et al. 2016 / Pedersen & Schuette 2020) ------


def _filter_recursive(x: np.ndarray, coefs: np.ndarray, init: np.ndarray) -> np.ndarray:
    """Port of R's stats::filter(x, coefs, method="recursive", init=init):
    y[i] = x[i] + coefs @ [y[i-1], ..., y[i-p]]. `init` is in reverse time
    order (init[0] = y[-1], the value immediately before x[0]), matching
    R's own convention for `init`."""
    p = len(coefs)
    state = list(init[:p])
    out = np.empty(len(x))
    for i, xi in enumerate(x):
        val = xi + float(np.dot(coefs, state))
        out[i] = val
        state = [val] + state[:-1]
    return out


def _radf_sb(
    data,
    minw: int | None,
    lag: int,
    nboot: int,
    type: str = "fixed",
    max_lag: int = 8,
    seed: int | None = None,
) -> dict:
    from . import _core  # lazy: see radf.py's radf() for why

    y, columns = _to_2d_array(data)
    nr, nc = y.shape
    minw = minw if minw is not None else psy_minw(nr)

    # Pedersen & Schuette (2020): automatic AIC/BIC lag selection, max
    # across the panel (radf_sb_'s bootstrap DGP loop assumes one common
    # lag order, matching radf()'s own single-`lag` API).
    if type != "fixed":
        lag = max(lag_select(y[:, j], criterion=type, max_lag=max_lag) for j in range(nc))

    pointer = nr - minw - lag

    initmat = np.zeros((nc, 1 + lag))
    resmat = np.zeros((nr - 2 - lag, nc))
    coefmat = np.zeros((nc, 2 + lag))

    for j in range(nc):
        dy = np.diff(y[:, j])
        ym = _embed(dy, lag + 2)
        design = np.column_stack([np.ones(ym.shape[0]), ym[:, 1:]])
        coef, *_ = np.linalg.lstsq(design, ym[:, 0], rcond=None)
        res = ym[:, 0] - design @ coef
        initmat[j, :] = ym[0, 1:]
        coefmat[j, :] = coef
        resmat[:, j] = res

    nres = resmat.shape[0]
    rng = np.random.default_rng(seed)

    bsadf_panel = np.empty((pointer, nboot))
    for i in range(nboot):
        # same time-resample index reused for every series in this draw,
        # preserving cross-sectional dependence (matches R's radf_sb_).
        boot_index = rng.integers(0, nres, size=nres)
        bsadf_boot = np.zeros(pointer)
        for j in range(nc):
            boot_res = resmat[boot_index, j]
            dboot_res = boot_res - boot_res.mean()
            # Prepend is initmat[j] reversed to forward-time order (R's own
            # `initmat[j, lag:1]` is short by one element for lag > 0 --
            # verified against R directly, see the validation script --
            # so this uses the full lag+1 reversal the algorithm needs).
            prepend = initmat[j, :][::-1]
            filtered = _filter_recursive(coefmat[j, 0] + dboot_res, coefmat[j, 1:], initmat[j, :])
            dy_boot = np.concatenate([prepend, filtered])
            y_boot = np.cumsum(np.concatenate(([y[0, j]], dy_boot)))
            yxmat_boot = unroot(y_boot, lag)
            aux_boot = _core.radf_stat(yxmat_boot, minw, lag)
            bsadf_boot += aux_boot[-pointer:]
        bsadf_panel[:, i] = bsadf_boot / nc

    gsadf_panel = bsadf_panel.max(axis=0)

    return {
        "bsadf_panel": bsadf_panel, "gsadf_panel": gsadf_panel,
        "minw": minw, "n": nr, "lag": lag, "series_names": columns,
    }


def radf_sb_cv(
    data,
    minw: int | None = None,
    lag: int = 0,
    nboot: int = 500,
    type: str = "fixed",
    max_lag: int = 8,
    seed: int | None = None,
) -> RadfSbCv:
    """Panel sieve bootstrap critical values (Pavlidis et al. 2016), with
    Pedersen & Schuette (2020)'s automatic AIC/BIC lag selection
    (`type="aic"/"bic"`) as a fix for fixed-lag size distortion under
    autocorrelated innovations."""
    r = _radf_sb(data, minw, lag, nboot, type, max_lag, seed)

    bsadf_cv = np.quantile(r["bsadf_panel"], PCNT, axis=1).T
    gsadf_cv = np.quantile(r["gsadf_panel"], PCNT)

    return RadfSbCv(
        gsadf_panel_cv=gsadf_cv, bsadf_panel_cv=bsadf_cv,
        method="Sieve Bootstrap", minw=r["minw"], n=r["n"], iter=nboot, lag=r["lag"],
        series_names=r["series_names"],
    )


def radf_sb_distr(
    data,
    minw: int | None = None,
    lag: int = 0,
    nboot: int = 500,
    type: str = "fixed",
    max_lag: int = 8,
    seed: int | None = None,
) -> RadfSbDistr:
    """Distribution of the panel GSADF statistic under the sieve
    bootstrap."""
    r = _radf_sb(data, minw, lag, nboot, type, max_lag, seed)
    return RadfSbDistr(
        gsadf_panel_distr=r["gsadf_panel"],
        method="Sieve Bootstrap", minw=r["minw"], n=r["n"], iter=nboot, lag=r["lag"],
    )
