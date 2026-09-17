"""lbi_test() + monitor_lbi() -- Breitung & Diegel (2025)'s locally best
invariant (LBI) test: the static, full-sample statistic and its own
sequential (constant-boundary CUSUM) monitoring extension. Ported from
exuber's R/lbi_test.R (R keeps both functions in one file -- matches).
Both are the cheapest detectors in this package -- no bootstrap, no C++,
and (for lbi_test) not even a table lookup: the null distribution is
exactly standard normal (via stdlib's statistics.NormalDist, no scipy
needed).

Indexing convention (differs from the R source, consistent with the rest
of pyexuber -- see datestamp.py): every "alarm" value returned by
monitor_lbi() is a 0-indexed position into the original input array.
"""

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

from exuber._monitor_common import _assert_sig_lvl, _training_window
from exuber.radf import _to_2d_array


@dataclass
class LbiTestResult:
    stat: np.ndarray  # (nc,)
    crit: float
    detected: np.ndarray  # (nc,) bool
    n: int
    sig_lvl: float
    series_names: list[str] | None = None


def lbi_test(data, sig_lvl: float = 95) -> LbiTestResult:
    """Breitung & Diegel (2025)'s static locally best invariant (LBI) test
    for a bubble known/assumed to span the entire sample:
    LBI = (y_T - y_0) / (sigma_tilde * sqrt(T - 1)), sigma_tilde^2 the
    sample variance of first differences. Heteroskedasticity-robust by
    construction, standard normal null distribution -- no bootstrap, no
    simulation, no published table.

    `sig_lvl` (one-sided, right-tailed -- positive bubbles only) may be any
    value in [50, 100), since the critical value is a closed-form normal
    quantile.
    """
    _assert_sig_lvl(sig_lvl, choices=None)
    x, columns = _to_2d_array(data)
    n, nc = x.shape

    stat = np.empty(nc)
    for j in range(nc):
        y = x[:, j]
        dy = np.diff(y)
        sigma2_tilde = np.mean(dy**2)
        stat[j] = (y[-1] - y[0]) / np.sqrt(sigma2_tilde * (n - 1))

    crit = NormalDist().inv_cdf(sig_lvl / 100)
    detected = stat > crit

    return LbiTestResult(
        stat=stat, crit=crit, detected=detected, n=n, sig_lvl=sig_lvl, series_names=columns
    )


# Breitung & Diegel (2025) Table 1 (page 7 of the rendered PDF, 1,000,000
# Monte Carlo reps at T=10,000): one-sided asymptotic critical values for
# the constant-boundary sequential detector, shared by mCUSUM (c_bar=0)
# and wCUSUM (c_bar>0) alike (their own invariance argument: the running
# max of a time-changed Brownian motion has the same distribution
# regardless of c_bar).
_BD_CUSUM_TABLE = {90: 1.64, 95: 1.95, 97.5: 2.24, 99: 2.57, 99.5: 2.80}


def _bd_cusum_q(sig_lvl: float) -> float:
    if sig_lvl not in _BD_CUSUM_TABLE:
        raise ValueError(
            f"sig_lvl must be one of {sorted(_BD_CUSUM_TABLE)} "
            "(Breitung & Diegel (2025)'s Table 1 only tabulates these significance levels)."
        )
    return _BD_CUSUM_TABLE[sig_lvl]


def _bd_cusum_weights(t_m: int, c_bar: float) -> np.ndarray:
    """eq. 12's weight function at the monitoring period's discrete
    relative positions r_j = j/T_m, j = 1, ..., T_m. c_bar=0 (mCUSUM)
    short-circuits to the flat weight 1/sqrt(T_m) directly (the paper's
    own stated closed form for that limiting case, not a numerical
    approximation of it)."""
    if c_bar == 0:
        return np.full(t_m, 1 / np.sqrt(t_m))
    r = np.arange(1, t_m + 1) / t_m
    return np.sqrt(2 * c_bar / t_m) / np.sqrt(np.exp(2 * c_bar) - 1) * np.exp(c_bar * r)


@dataclass
class MonitorLbiResult:
    stat: np.ndarray  # (n_mon, nc)
    boundary: float
    t_star: int
    alarm: np.ndarray  # (nc,), NaN where no breach occurred
    n: int
    c_bar: float
    sig_lvl: float
    series_names: list[str] | None = None


def monitor_lbi(
    data, r_star: float = 0.5, c_bar: float = 0, sig_lvl: float = 95
) -> MonitorLbiResult:
    """Breitung & Diegel (2025)'s sequential (constant-boundary) extension
    of lbi_test()'s LBI statistic, for monitoring a series in real time
    when the bubble's start date is unknown: after a training window
    [0, T*) assumed free of exuberance, the (optionally exponentially
    weighted) partial sum of post-training first differences is compared
    against a constant boundary, flagging the first breach.

    `c_bar` (their eq. 12, >= 0) exponentially up-weights later (more
    bubble-like) monitoring observations; `0` (default) is the flat-weight
    "mCUSUM" variant, `> 0` is "wCUSUM" (the paper's own suggested default
    for a moderate power boost is `2`). Critical values (`sig_lvl`, one of
    90, 95, 97.5, 99, 99.5 -- Breitung & Diegel's Table 1) are the same for
    every `c_bar`.

    sigma_tilde^2 is estimated from the training window only (their own
    Section 4.2: "the training set ... is used for estimating nuisance
    parameters such as sigma^2"), consistent with lbi_test()'s own
    training-free full-sample sigma_tilde^2 for the static case.
    """
    if c_bar < 0:
        raise ValueError("c_bar must be >= 0")
    b_alpha = _bd_cusum_q(sig_lvl)
    x, columns = _to_2d_array(data)
    n, nc = x.shape
    t_star = _training_window(r_star, n)
    t_m = n - t_star
    w = _bd_cusum_weights(t_m, c_bar)

    stat_path = np.full((t_m, nc), np.nan)
    alarm = np.full(nc, np.nan)

    for j in range(nc):
        y = x[:, j]
        dy = np.diff(y)
        sigma2_tilde = np.mean(dy[: t_star - 1] ** 2)
        dy_mon = dy[t_star - 1 : n - 1]
        phi = np.cumsum(w * dy_mon) / np.sqrt(sigma2_tilde)
        stat_path[:, j] = phi
        breach = np.where(phi > b_alpha)[0]
        if breach.size:
            alarm[j] = t_star + breach[0]

    return MonitorLbiResult(
        stat=stat_path,
        boundary=b_alpha,
        t_star=t_star,
        alarm=alarm,
        n=n,
        c_bar=c_bar,
        sig_lvl=sig_lvl,
        series_names=columns,
    )
