"""monitor_quantile() -- Wu, Shi & Wu (2025)'s QPWY recursive quantile
monitoring: the same per-window QR t-ratio as quantile_test(), computed
over an expanding window [0, r) (start fixed at 0, exactly radf()'s own
`badf` shape) instead of a single full-sample test. Ported from exuber's
R/monitor_quantile.R. Only QPWY (single recursion) is implemented, not
QPSY (their double recursion, O(T^2) QR fits).

Critical values come from their Theorem 1 / Corollary 1-2: under the null
each window's t-ratio converges to delta*Q_{r1,r2} + sqrt(1-delta^2)*Z_{r1,r2},
Q the Dickey-Fuller t functional and Z its counterpart driven by an
independent Brownian motion. Z is N(0,1) for any one window but varies
across windows, so a monitoring boundary (a functional of the whole path)
must simulate it as a process -- the original port drew one z per
replicate, which oversized the test (see docs/alternative-paradigms.md).
_quantile_boundary_sim() simulates discretized Q and Z for every window
via prefix sums, O(1) per window, no QR fits and no radf() call.

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
from exuber.radf import _to_2d_array, psy_minw


def _quantile_window_stat(yy: np.ndarray, tau: float) -> float:
    """QR t-ratio on one window `yy` (WSW's QUr statistic, eq. 18),
    density estimated from the window's own first differences."""
    ylag = yy[:-1]
    yresp = yy[1:]
    _, alpha_hat = _quantile_regression_fit(yresp, ylag, tau)
    _, f_hat = _quantile_check_density(yresp - ylag, tau)
    y_pzy = np.sum((ylag - np.mean(ylag)) ** 2)
    return float((f_hat / np.sqrt(tau * (1 - tau))) * np.sqrt(y_pzy) * (alpha_hat - 1))


def _qpwy_stat_path(y: np.ndarray, tau: float, r_idx: np.ndarray) -> np.ndarray:
    """QPWY_r(tau) for every window length r in `r_idx` -- window fixed at
    [0, r) (start=0, matching radf()'s own badf convention)."""
    return np.array([_quantile_window_stat(y[:r], tau) for r in r_idx])


def _quantile_boundary_sim(
    n: int, minw: int, nrep: int, delta: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Simulated null path suprema, (nrep, len(delta)): for each replicate,
    sup over the monitoring path of delta*Q + sqrt(1-delta^2)*Z. Windows
    are over the n-1 regression pairs (y_{t-1}, dy_t); window (lo, hi] of
    pairs <-> y[lo:hi+1], so QPWY's window [0, r) is (0, r-1]."""
    npairs = n - 1
    hi = np.arange(minw, npairs + 1)[None, :]
    lo = np.zeros((1, 1), dtype=int)
    valid = hi - lo >= minw
    length = np.where(valid, hi - lo, 1)

    def win(cs: np.ndarray) -> np.ndarray:
        return cs[hi] - cs[lo]

    def mk(v: np.ndarray) -> np.ndarray:
        return np.concatenate([[0.0], np.cumsum(v)])

    out = np.empty((nrep, len(delta)))
    for i in range(nrep):
        e = rng.normal(size=npairs)
        v = rng.normal(size=npairs)
        x = np.concatenate([[0.0], np.cumsum(e)])[:npairs]  # y_{t-1}
        sx = win(mk(x))
        sxx = np.where(valid, win(mk(x**2)) - sx**2 / length, 1.0)
        q = (win(mk(x * e)) - sx * win(mk(e)) / length) / np.sqrt(sxx)
        z = (win(mk(x * v)) - sx * win(mk(v)) / length) / np.sqrt(sxx)
        for j, d in enumerate(delta):
            out[i, j] = (d * q + np.sqrt(1 - d**2) * z)[valid].max()
    return out


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

    A single flat boundary is used per series (not one value per r): the
    quantile of each simulated null path's own supremum (mirrors
    radf_mc_cv()'s own sadf_cv construction), which controls the
    first-crossing false-alarm rate.

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

    stat_path = np.empty((len(r_idx), nc))
    delta = np.empty(nc)
    alarm = np.full(nc, np.nan)

    for j in range(nc):
        y = x[:, j]
        stat_path[:, j] = _qpwy_stat_path(y, tau, r_idx)
        dy_full = np.diff(y)
        psi = tau - (dy_full < _quantile_narm(dy_full, tau)).astype(float)
        delta[j] = float(np.clip(np.corrcoef(dy_full, psi)[0, 1], -1, 1))

    sup_u = _quantile_boundary_sim(n, minw, nrep, delta, rng)
    boundary = np.array([_quantile_narm(sup_u[:, j], sig_lvl / 100) for j in range(nc)])
    for j in range(nc):
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
