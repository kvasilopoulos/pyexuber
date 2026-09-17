"""monitor_quantile() -- Wu, Shi & Wu (2025)'s QPWY recursive quantile
monitoring: the same per-window QR t-ratio as quantile_test(), computed
over an expanding window [0, r) (start fixed at 0, exactly radf()'s own
`badf` shape) instead of a single full-sample test. Ported from exuber's
R/monitor_quantile.R. Only QPWY (single recursion) is implemented, not
QPSY (their double recursion, additionally optimizing over the window
start too -- O(T^2) QR fits, not attempted).

The critical value reuses quantile_test()'s own decomposition: their
Corollary 1-2 identify the boundary-relevant null distribution's Q_{0,r}
component with exactly radf()'s own `badf` sequence under a simulated
null path -- one radf() call per Monte Carlo replicate gives the WHOLE
boundary path at once, no new simulation theory.

Indexing convention (differs from the R source, consistent with the rest
of pyexuber -- see datestamp.py): the "alarm" value returned here is a
0-indexed position into the original input array.

RNG note: uses numpy's Generator, not R's -- see quantile_test()'s own
module-level note.
"""

from dataclasses import dataclass

import numpy as np

from exuber._monitor_common import _assert_sig_lvl, _quantile_narm
from exuber.quantile_test import _quantile_check_density, _quantile_regression_fit
from exuber.radf import _to_2d_array, psy_minw, radf


def _qpwy_stat_path(y: np.ndarray, tau: float, r_idx: np.ndarray) -> np.ndarray:
    """QPWY_r(tau) for every window length r in `r_idx` -- window fixed at
    [0, r) (start=0, matching radf()'s own badf convention), mirroring
    quantile_test()'s own per-window QR t-ratio construction (eq. 18)
    exactly, just repeated over a growing window instead of the full
    sample."""
    out = np.empty(len(r_idx))
    for i, r in enumerate(r_idx):
        yy = y[:r]
        ylag = yy[:-1]
        yresp = yy[1:]
        dy = yresp - ylag

        _, alpha_hat = _quantile_regression_fit(yresp, ylag, tau)
        _, f_hat = _quantile_check_density(dy, tau)
        y_pzy = np.sum((ylag - np.mean(ylag)) ** 2)
        out[i] = (f_hat / np.sqrt(tau * (1 - tau))) * np.sqrt(y_pzy) * (alpha_hat - 1)
    return out


def _qpwy_boundary_sim(n: int, minw: int, nrep: int, rng: np.random.Generator) -> np.ndarray:
    """Simulated null Q_{0,r} paths: one radf() call per replicate gives
    the WHOLE expanding-window badf sequence at once (Corollary 2's own
    identification of Q_{0,r} with the ADF-family recursive t-statistic
    distribution) -- an (nrep, n - minw) matrix, matching badf's own shape."""
    q = np.empty((nrep, n - minw))
    for i in range(nrep):
        ysim = np.cumsum(rng.normal(size=n))
        result = radf(ysim, minw=minw, lag=0)
        q[i, :] = result.badf[:, 0]
    return q


@dataclass
class MonitorQuantileResult:
    stat: np.ndarray  # (n_mon, nc)
    boundary: np.ndarray  # (nc,), flat (supremum-calibrated, not per-r marginal)
    delta: np.ndarray  # (nc,)
    alarm: np.ndarray  # (nc,), NaN where no breach occurred
    n: int
    minw: int
    tau: float
    sig_lvl: float
    iter: int
    series_names: list[str] | None = None


def monitor_quantile(
    data,
    tau: float = 0.5,
    minw: int | None = None,
    nrep: int = 500,
    sig_lvl: float = 95,
    seed: int | None = None,
) -> MonitorQuantileResult:
    """Wu, Shi & Wu (2025)'s QPWY real-time monitoring: a quantile-
    regression (QR) analogue of PWY's own recursive ADF t-statistic,
    testing at a chosen conditional quantile `tau` over an expanding
    window [0, r) (start fixed at 0, exactly radf()'s own `badf`
    convention) rather than quantile_test()'s single full-sample test.

    Needs the compiled `_core` extension (via radf(), both for the point
    statistic's own O(T) QR-fit loop -- no closed form the way OLS has --
    and for simulating the boundary).

    A single flat boundary is used per series (not one value per r):
    controlling the first-crossing false-alarm rate requires calibrating
    against each simulated path's own supremum (mirrors radf_mc_cv()'s
    own sadf_cv construction), not the per-r marginal quantile, which
    would badly inflate the false-alarm rate.

    RNG note: uses numpy's Generator, not R's -- see quantile_test()'s
    own module-level note.
    """
    if not (0 < tau < 1):
        raise ValueError("tau must be in (0, 1)")
    _assert_sig_lvl(sig_lvl)
    x, columns = _to_2d_array(data)
    n, nc = x.shape
    minw = minw if minw is not None else psy_minw(n)
    if minw <= 2:
        raise ValueError("minw must be a positive integer greater than 2")
    r_idx = np.arange(minw + 1, n + 1)
    rng = np.random.default_rng(seed)

    q_paths = _qpwy_boundary_sim(n, minw, nrep, rng)
    z = rng.normal(size=nrep)

    stat_path = np.empty((len(r_idx), nc))
    delta = np.empty(nc)
    boundary = np.empty(nc)
    alarm = np.full(nc, np.nan)

    for j in range(nc):
        y = x[:, j]
        stat_path[:, j] = _qpwy_stat_path(y, tau, r_idx)

        dy_full = np.diff(y)
        psi = tau - (dy_full < _quantile_narm(dy_full, tau)).astype(float)
        delta_j = float(np.clip(np.corrcoef(dy_full, psi)[0, 1], -1, 1))
        delta[j] = delta_j

        u = np.sqrt(1 - delta_j**2) * z[:, None] + delta_j * q_paths
        sup_u = u.max(axis=1)
        boundary[j] = _quantile_narm(sup_u, sig_lvl / 100)

        breach = np.where(stat_path[:, j] > boundary[j])[0]
        if breach.size:
            alarm[j] = r_idx[breach[0]] - 1

    return MonitorQuantileResult(
        stat=stat_path,
        boundary=boundary,
        delta=delta,
        alarm=alarm,
        n=n,
        minw=minw,
        tau=tau,
        sig_lvl=sig_lvl,
        iter=nrep,
        series_names=columns,
    )
