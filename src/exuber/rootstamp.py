"""Root inference: Guo, Sun & Wang (2019) normal-t CI and Phillips-
Magdalinos (2007) Cauchy CI for the explosive AR(1) root, plus implied
doubling time. Ported from exuber's R/rootstamp.R (see
docs/dating-and-root-inference.md in the umbrella repo for the full
methodology and validation record).

Deviation from R: R's rootstamp.radf_obj(object, ds) reads its input
series back out of `object` (a radf_obj, which carries its own data via
mat()). Python's RadfResult doesn't retain the raw data (see
datestamp.py's docstring for the same gap), so rootstamp_episodes() here
takes the original `data` as an explicit argument instead.
"""

import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

from exuber.datestamp import Episode
from exuber.radf import _to_2d_array


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
