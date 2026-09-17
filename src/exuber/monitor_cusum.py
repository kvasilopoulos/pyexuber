"""monitor_cusum() -- Homm & Breitung (2012)'s CUSUM real-time monitoring
(a structurally different statistic from monitor()'s recursive ADF-family
training-max -- a standardized running sum of first differences compared
against a closed-form asymptotic boundary, no bootstrap, no C++). Ported
from exuber's R/monitor_cusum.R. Astill, Harvey, Leybourne, Taylor & Zu
(2023)'s volatility-robust one-sided-kernel "CUSUMV" variant
(type="kernel") is included; their Corollary 1 establishes the same
boundary function controls the false-alarm rate for both.

Indexing convention (differs from the R source, consistent with the rest
of pyexuber -- see datestamp.py): every "alarm" value returned here is a
0-indexed position into the original input array.
"""

from dataclasses import dataclass

import numpy as np

from exuber._monitor_common import _HB_K_GRID, _HB_N_GRID, _sig_lvl_to_beta, _training_window
from exuber.radf import _to_2d_array

# Homm & Breitung (2012) Table 8(i): CUSUM finite-sample boundary constant
# b_{k,alpha}, transcribed from exuber/R/monitor_cusum.R's
# hb_cusum_finite_table. Same (n, alpha, k) grid as FLUC's Table 7.
_HB_CUSUM_FINITE_TABLE = {
    (100, 0.10): (0.92, 1.39, 1.62, 1.80, 1.92, 2.08, 2.19),
    (100, 0.05): (1.51, 2.14, 2.47, 2.73, 2.88, 3.20, 3.36),
    (100, 0.01): (2.86, 3.92, 4.57, 4.94, 5.30, 5.72, 6.02),
    (50, 0.10): (0.87, 1.31, 1.55, 1.70, 1.82, 1.97, 2.06),
    (50, 0.05): (1.43, 2.03, 2.34, 2.62, 2.80, 3.03, 3.12),
    (50, 0.01): (2.85, 3.87, 4.25, 4.77, 5.03, 5.47, 5.94),
    (20, 0.10): (0.81, 1.18, 1.43, 1.61, 1.75, 1.91, 2.02),
    (20, 0.05): (1.27, 1.96, 2.35, 2.53, 2.72, 3.00, 3.13),
    (20, 0.01): (2.61, 3.91, 4.49, 4.97, 5.16, 5.52, 5.74),
}


def _hb_cusum_finite_q(sig_lvl: float, n_train: int, k: float) -> float:
    """Same lookup-and-snap convention as monitor.py's _hb_fluc_q()."""
    beta = _sig_lvl_to_beta(sig_lvl)
    n_snap = min(_HB_N_GRID, key=lambda x: abs(n_train - x))
    k_snap = min(_HB_K_GRID, key=lambda x: abs(k - x))
    return _HB_CUSUM_FINITE_TABLE[(n_snap, beta)][_HB_K_GRID.index(k_snap)]


def _cusum_stat_path(y: np.ndarray, t_star: int, b_alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """HB's CUSUM statistic (eq. 26) and boundary (eq. 29) at every
    monitoring point t = t_star, ..., n-1 (0-indexed). sigma_hat_t^2 is the
    recursive (growing) sample variance of first differences up to t,
    re-estimated as new data arrives -- legitimate in real-time monitoring
    since only past/current data is used at each t."""
    n = len(y)
    dy = np.diff(y)
    cs_dy2 = np.cumsum(dy**2)

    t_idx = np.arange(t_star, n)  # 0-indexed monitoring positions
    sigma2_t = cs_dy2[t_idx - 1] / t_idx  # (t_idx+1-1) observations of dy, 1-indexed count == t_idx
    s_t = (y[t_idx] - y[t_star - 1]) / np.sqrt(sigma2_t)
    c_t = np.sqrt(b_alpha + np.log((t_idx + 1) / t_star))
    boundary_t = c_t * np.sqrt(t_idx + 1)
    return s_t, boundary_t


def _one_sided_kernel_spot_vol(dy: np.ndarray, h: int = 20, kernel: str = "gaussian") -> np.ndarray:
    """One-sided (backward-looking) Nadaraya-Watson kernel spot-variance
    estimator, Astill, Harvey, Leybourne, Taylor & Zu (2023)'s eq. 6-7: a
    FIXED weight vector applied as a one-sided moving weighted average of
    squared first differences. AHLTZ's own convention: sigma2_j := 1 for
    j <= h (the first h observations)."""
    u = np.arange(h + 1) / h
    w = np.exp(-(u**2) / 2) if kernel == "gaussian" else (np.abs(u) <= 1).astype(float) / 2
    w = w / w.sum()
    dy2 = dy**2
    sigma2 = np.full(len(dy), np.nan)
    for j in range(len(dy)):
        if j < h:
            sigma2[j] = 1.0
        else:
            sigma2[j] = np.dot(w, dy2[j - h : j + 1][::-1])
    return sigma2


def _cusum_stat_path_kernel(
    y: np.ndarray, t_star: int, b_alpha: float, h: int, kernel: str
) -> tuple[np.ndarray, np.ndarray]:
    """AHLTZ's modified (volatility-robust) CUSUM statistic, eq. 6: each
    first difference standardized by its own one-sided kernel
    spot-variance estimate before cumulating. Reuses cusum_stat_path()'s
    boundary formula unchanged (their Corollary 1)."""
    n = len(y)
    dy = np.diff(y)
    sigma2_dy = _one_sided_kernel_spot_vol(dy, h=h, kernel=kernel)
    weighted = dy / np.sqrt(sigma2_dy)
    cs = np.concatenate(([0.0], np.cumsum(weighted)))

    t_idx = np.arange(t_star, n)
    sv_t = cs[t_idx] - cs[t_star - 1]
    c_t = np.sqrt(b_alpha + np.log((t_idx + 1) / t_star))
    boundary_t = c_t * np.sqrt(t_idx + 1)
    return sv_t, boundary_t


@dataclass
class MonitorCusumResult:
    stat: np.ndarray  # (n_mon, nc)
    boundary: np.ndarray  # (n_mon, nc)
    t_star: int
    alarm: np.ndarray  # (nc,), NaN where no breach occurred
    n: int
    b_alpha: float
    series_names: list[str] | None = None


def monitor_cusum(
    data,
    r_star: float = 0.5,
    b_alpha: float = 4.6,
    boundary: str = "asymptotic",
    sig_lvl: float = 95,
    type: str = "standard",  # noqa: A002 (mirrors exuber's R argument name)
    h: int = 20,
    kernel: str = "gaussian",
) -> MonitorCusumResult:
    """Homm & Breitung (2012)'s CUSUM real-time monitoring: fix a training
    window [0, T*) assumed free of exuberance, then compare the
    standardized cumulative sum of post-training first differences,
    S_t = (y_t - y_{T*}) / sigma_hat_t, against a closed-form boundary
    c_t * sqrt(t), c_t = sqrt(b_alpha + log(t / T*)), flagging the first
    breach.

    `boundary = "asymptotic"` (default) uses `b_alpha` directly (HB's own
    one-sided asymptotic calibration for a 5% level, an upper bound via
    Chu, Stinchcombe & White (1996), typically conservative in finite
    samples). `boundary = "finite"` instead looks up HB's own finite-sample
    constant (their Table 8), requiring `sig_lvl` in {90, 95, 99}.

    `type = "kernel"` uses Astill, Harvey, Leybourne, Taylor & Zu (2023)'s
    volatility-robust "CUSUMV" modification: each first difference is
    standardized by its own one-sided kernel spot-variance estimate (their
    eq. 6-7, bandwidth `h`, default 20 -- their own empirically-recommended
    value) instead of a single running variance, before cumulating. Their
    Corollary 1 establishes the same boundary function still controls the
    false-alarm rate under time-varying volatility.
    """
    if kernel not in ("gaussian", "uniform"):
        raise ValueError("kernel must be 'gaussian' or 'uniform'")
    if boundary not in ("asymptotic", "finite"):
        raise ValueError("boundary must be 'asymptotic' or 'finite'")
    if type not in ("standard", "kernel"):
        raise ValueError("type must be 'standard' or 'kernel'")

    x, columns = _to_2d_array(data)
    n, nc = x.shape
    t_star = _training_window(r_star, n)

    if boundary == "finite":
        b_alpha = _hb_cusum_finite_q(sig_lvl, t_star, n / t_star)

    n_mon = n - t_star
    stat_path = np.full((n_mon, nc), np.nan)
    boundary_path = np.full((n_mon, nc), np.nan)
    alarm = np.full(nc, np.nan)

    for j in range(nc):
        y = x[:, j]
        if type == "kernel":
            s_t, b_t = _cusum_stat_path_kernel(y, t_star, b_alpha, h, kernel)
        else:
            s_t, b_t = _cusum_stat_path(y, t_star, b_alpha)
        stat_path[:, j] = s_t
        boundary_path[:, j] = b_t
        breach = np.where(s_t > b_t)[0]
        if breach.size:
            alarm[j] = t_star + breach[0]

    return MonitorCusumResult(
        stat=stat_path,
        boundary=boundary_path,
        t_star=t_star,
        alarm=alarm,
        n=n,
        b_alpha=b_alpha,
        series_names=columns,
    )
