"""Monte Carlo / wild bootstrap critical values. Ports of exuber's
R/radf_mc.R and R/radf_wb.R (HLST variant only -- see module docstring
in __init__.py for what's not yet ported).

RNG note: uses numpy's Generator, not R's RNG -- a given `seed` will not
reproduce the same draws as the R functions of the same name. See
exubercore's RNG design note / sim.py's module docstring.
"""

from dataclasses import dataclass

import numpy as np

from exuber._unroot import unroot
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


def _wb_dgp_hlst(y: np.ndarray, dist_rad: bool, rng: np.random.Generator) -> np.ndarray:
    dy = np.diff(y)
    nr = len(dy)
    w = rng.choice([-1.0, 1.0], size=nr) if dist_rad else rng.normal(size=nr)
    estar = np.cumsum(w * dy)
    return np.concatenate(([0.0], estar))


def _radf_wb_hlst(
    data, minw: int | None = None, nboot: int = 500, dist_rad: bool = False, seed: int | None = None
) -> dict:
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
            ystar = _wb_dgp_hlst(y[:, j], dist_rad, rng)
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
    data, minw: int | None = None, nboot: int = 500, dist_rad: bool = False, seed: int | None = None
) -> RadfCv:
    """Wild bootstrap critical values (Harvey, Leybourne, Sollis & Taylor 2016)."""
    r = _radf_wb_hlst(data, minw, nboot, dist_rad, seed)

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
    data, minw: int | None = None, nboot: int = 500, dist_rad: bool = False, seed: int | None = None
) -> RadfDistr:
    """Wild bootstrap distribution of the ADF/SADF/GSADF statistics."""
    r = _radf_wb_hlst(data, minw, nboot, dist_rad, seed)
    return RadfDistr(
        adf_distr=r["adf"], sadf_distr=r["sadf"], gsadf_distr=r["gsadf"],
        method="Wild Bootstrap", minw=r["minw"], n=r["n"], iter=nboot,
    )
