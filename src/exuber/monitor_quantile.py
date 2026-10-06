"""monitor_quantile(): the QPWY and QPSY recursive quantile monitors of
Wu, Shi & Wu (2025). The statistic is the same per-window QR t-ratio as in
quantile_test(). QPWY computes it over an expanding window [0, r), which has
the shape of the `badf` sequence of radf(). QPSY also takes the supremum over
every window start, which has the shape of `bsadf`. Ported from exuber's
R/monitor_quantile.R.

The critical values come from Theorem 1 and Corollaries 1 and 2 of the paper.
Under the null, the t-ratio of each window converges to
delta*Q_{r1,r2} + sqrt(1-delta^2)*Z_{r1,r2}, where Q is the Dickey-Fuller t
functional and Z is its counterpart driven by an independent Brownian motion.
Z is N(0,1) for any single window but varies across windows. A monitoring
boundary is a functional of the whole path, so it must simulate Z as a
process. The original port drew one z for each replicate, which oversized the
test (see docs/alternative-paradigms.md). _quantile_boundary_sim() simulates
the discretized Q and Z for every window by prefix sums, at O(1) per window
and with no QR fits and no call to radf().

Indexing convention. This differs from the R source and is consistent with
the rest of pyexuber (see datestamp.py). The "alarm" value returned here is a
0-indexed position into the original input array.

Bootstrap boundary. boundary="bootstrap" implements Algorithm 1 of the paper
in place of the asymptotic limit. It resamples the centred first differences
with replacement, cumulates them into a null random walk and recomputes the
whole QPWY or QPSY path on it. The boundary is the quantile of the path
maxima over the replicates. The paper discards the first b = 100 draws to
remove the initialisation effect. The statistic regresses with an intercept,
so it does not depend on the level of the series, and with i.i.d. draws the
discarded values would change nothing, so the burn-in is left out. Each
replicate costs one full statistic path, O(T) QR fits for QPWY and O(T^2) for
QPSY.

Random numbers. The module uses numpy's Generator and not R's generator. See
the module-level note of quantile_test().
"""

import warnings
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
    """QPWY_r(tau) for every window length r in `r_idx`, with the window fixed at
    [0, r) (start=0, as in the badf convention of radf())."""
    return np.array([_quantile_window_stat(y[:r], tau) for r in r_idx])


def _qpsy_stat_path(y: np.ndarray, tau: float, r_idx: np.ndarray, minw: int) -> np.ndarray:
    """QPSY_r(tau, r0) for every window length r in `r_idx`, taking the supremum over
    window starts 0, ..., r - minw - 1. Every window keeps at least minw
    regression observations, which is the floor of the first window of
    QPWY."""
    return np.array(
        [max(_quantile_window_stat(y[r1:r], tau) for r1 in range(r - minw)) for r in r_idx]
    )


def _quantile_boundary_sim(
    n: int, minw: int, nrep: int, delta: np.ndarray, qpsy: bool, rng: np.random.Generator
) -> np.ndarray:
    """Simulated null path suprema, (nrep, len(delta)): for each replicate,
    sup over the monitoring path of delta*Q + sqrt(1-delta^2)*Z. Windows
    are over the n-1 regression pairs (y_{t-1}, dy_t); window (lo, hi] of
    pairs <-> y[lo:hi+1], so QPWY's window [0, r) is (0, r-1] and QPSY's
    [r1, r) is (r1, r-1]."""
    npairs = n - 1
    hi = np.arange(minw, npairs + 1)[None, :]
    lo = (np.zeros(1, dtype=int) if not qpsy else np.arange(npairs - minw + 1))[:, None]
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


def _quantile_boundary_boot(
    y: np.ndarray, tau: float, minw: int, nrep: int, qpsy: bool, rng: np.random.Generator
) -> np.ndarray:
    """Path maxima of the i.i.d. residual bootstrap (Algorithm 1 of the paper):
    resample the centred first differences, cumulate them, and recompute the
    whole statistic path. Returns `nrep` maxima."""
    n = len(y)
    r_idx = np.arange(minw + 1, n + 1)
    u = np.diff(y)
    u = u - u.mean()
    out = np.empty(nrep)
    for i in range(nrep):
        ystar = np.concatenate([[0.0], np.cumsum(rng.choice(u, size=n - 1, replace=True))])
        path = (
            _qpsy_stat_path(ystar, tau, r_idx, minw) if qpsy else _qpwy_stat_path(ystar, tau, r_idx)
        )
        out[i] = path.max()
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
    type: str = "qpwy"
    series_names: list[str] | None = None
    boundary_type: str = "asymptotic"


def monitor_quantile(
    data,
    tau: float = 0.5,
    minw: int | None = None,
    nrep: int = 500,
    sig_lvl: float = 95,
    seed: int | None = None,
    type: str = "qpwy",
    boundary: str = "asymptotic",
) -> MonitorQuantileResult:
    """Wu, Shi & Wu (2025)'s QPWY/QPSY real-time monitoring: quantile-
    regression (QR) analogues of PWY's and PSY's recursive ADF
    t-statistics, testing at a chosen conditional quantile `tau`.
    `type="qpwy"` uses the expanding window [0, r) (radf()'s own `badf`
    convention); `type="qpsy"` also sup's over every window start (`bsadf`).

    The point statistic needs real QR fits, which cost O(T) for QPWY and O(T^2)
    for QPSY. QPSY is slow: with the IRLS solver of this package it takes tens
    of seconds per series at n = 200.

    Each series gets a single flat boundary, and not one value for each r. It is
    the quantile of the supremum of each simulated null path, built in the
    same way as sadf_cv in radf_mc_cv(), and it controls the first-crossing
    false-alarm rate.

    `boundary="asymptotic"` (the default) simulates the limiting null
    distribution. It is well sized near the median (3.5-4.0% at a nominal 5%,
    Gaussian and t3, tau = 0.5), but the small early windows oversize it away
    from the median, QPSY badly even with Gaussian data (35% at tau = 0.9;
    t3: 21% at tau = 0.8, 44% at 0.9; n = 100). QPWY with t3: 7.5-8.5% at
    tau = 0.2/0.8, 12.5% at 0.9, n = 150. These are R's numbers; see
    docs/alternative-paradigms.md. Using type="qpsy" with tau away from 0.5
    and the asymptotic boundary issues a UserWarning.

    `boundary="bootstrap"` implements Algorithm 1 of Wu, Shi & Wu: it
    resamples the centred first differences, rebuilds the whole statistic path
    on each resample and takes the quantile of the path maxima. `nrep` is then
    the number of bootstrap replicates. It follows the finite-sample null
    distribution of the statistic and so corrects the oversizing above, but
    each replicate costs a full statistic path. With the IRLS solver of this
    package QPSY takes tens of seconds per replicate at n = 200, so use the
    bootstrap with QPSY only for short series or a small `nrep`.

    Random numbers: the function uses numpy's Generator and not R's generator.
    See the module-level note of quantile_test().
    """
    if type not in ("qpwy", "qpsy"):
        raise ValueError("type must be 'qpwy' or 'qpsy'")
    if boundary not in ("asymptotic", "bootstrap"):
        raise ValueError("boundary must be 'asymptotic' or 'bootstrap'")
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
        if type == "qpwy":
            stat_path[:, j] = _qpwy_stat_path(y, tau, r_idx)
        else:
            stat_path[:, j] = _qpsy_stat_path(y, tau, r_idx, minw)
        dy_full = np.diff(y)
        psi = tau - (dy_full < _quantile_narm(dy_full, tau)).astype(float)
        delta[j] = float(np.clip(np.corrcoef(dy_full, psi)[0, 1], -1, 1))

    if boundary == "asymptotic" and type == "qpsy" and abs(tau - 0.5) > 0.05:
        warnings.warn(
            "QPSY's asymptotic boundary is oversized away from the median in small samples "
            '(21% at tau = 0.8 with t3 data, n = 100, nominal 5%); use boundary="bootstrap" '
            "for a boundary that follows the finite-sample distribution, see the "
            "docstring of monitor_quantile.",
            stacklevel=2,
        )
    if boundary == "asymptotic":
        sup_u = _quantile_boundary_sim(n, minw, nrep, delta, type == "qpsy", rng)
        bound = np.array([_quantile_narm(sup_u[:, j], sig_lvl / 100) for j in range(nc)])
    else:
        bound = np.array(
            [
                _quantile_narm(
                    _quantile_boundary_boot(x[:, j], tau, minw, nrep, type == "qpsy", rng),
                    sig_lvl / 100,
                )
                for j in range(nc)
            ]
        )
    for j in range(nc):
        breach = np.where(stat_path[:, j] > bound[j])[0]
        if breach.size:
            alarm[j] = r_idx[breach[0]] - 1

    return MonitorQuantileResult(
        stat=stat_path,
        boundary=bound,
        delta=delta,
        alarm=alarm,
        n=n,
        minw=minw,
        tau=tau,
        sig_lvl=sig_lvl,
        iter=nrep,
        type=type,
        series_names=columns,
        boundary_type=boundary,
    )
