"""Real-time monitoring for explosive bubbles. Ports of exuber's
R/monitor.R, R/monitor_cusum.R, R/lbi_test.R, R/quantile_test.R and
R/monitor_quantile.R. See docs/monitoring.md and
docs/alternative-paradigms.md for the full derivations; this module keeps
only the condensed formula citations needed to read the code.

Indexing convention (differs from the R source, consistent with the rest
of pyexuber -- see datestamp.py): every "alarm"/position value returned
here is a 0-indexed position into the original input array (`y[alarm]` is
the first monitoring observation that breaches the boundary), not an
R-style 1-indexed observation count.

RNG note (quantile_test/monitor_quantile only): uses numpy's Generator,
not R's RNG -- a given `seed` will not reproduce the same critical-value
draws as the R functions of the same name (see sim.py's module
docstring). The point statistics themselves (tstat, delta, badf/bsadf/
cusum paths) are fully deterministic and match R bit-for-bit.
"""

from dataclasses import dataclass

import numpy as np

from exuber.radf import RadfResult, _to_2d_array, psy_minw, radf

# ---------------------------------------------------------------------------
# Shared helpers (ports of exuber/R/utils-defensive.R's training_window(),
# assert_sig_lvl(), quantile_narm())
# ---------------------------------------------------------------------------


def _training_window(r_star: float, n: int, min_len: int = 3) -> int:
    """`r_star` is a fraction of the sample (< 1) or an observation count
    (>= 1); returns the training-window length T*, which must be at least
    `min_len` and leave at least one monitoring observation."""
    t_star = round(r_star * n) if r_star < 1 else int(r_star)
    if t_star < min_len:
        raise ValueError(
            f"Training window (r_star) is too short: needs at least {min_len} observations."
        )
    if t_star >= n:
        raise ValueError("Training window (r_star) must leave at least one monitoring observation.")
    return t_star


def _assert_sig_lvl(sig_lvl: float, choices: tuple[float, ...] | None = (90, 95, 99)) -> None:
    """Package-wide convention: `sig_lvl` is on the 0-100 scale. `choices`
    restricts to a tabulated set; None allows any value in [50, 100)."""
    if choices is None:
        if not (50 <= sig_lvl < 100):
            raise ValueError(
                "sig_lvl is on the 0-100 scale (e.g. 95, not 0.95) and should be in [50, 100)"
            )
        return
    if not any(abs(sig_lvl - c) < 1e-8 for c in choices):
        raise ValueError(f"sig_lvl must be one of {choices}")


def _quantile_narm(x: np.ndarray, prob: float) -> float:
    """quantile(), but NaN entries are dropped instead of propagating."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return float(np.quantile(x, prob))


# ---------------------------------------------------------------------------
# monitor() -- Phillips & Shi (2020) training/monitoring split (Family A),
# with Kurozumi (2020)'s closed-form SADF/GSADF_{s0} boundary and Homm &
# Breitung (2012)'s FLUC boundary. `boundary = "bootstrap"` (Phillips & Shi
# 2020's own wild-bootstrap boundary) is NOT ported: it needs
# radf_wb_ps_cv(), which pyexuber does not implement yet (see
# __init__.py's module docstring / docs/parity.md) -- both boundaries
# ported here are closed-form/table-lookup, no bootstrap needed.
# ---------------------------------------------------------------------------

# Kurozumi (2020) Table 1, transcribed from exuber/R/monitor.R's
# kurozumi_table1 (itself transcribed from a rendered PDF page -- see
# docs/monitoring.md, "Kurozumi (2020, 2021)"). Only q0_df (SADF, s0=0)
# and q04_df/q08_df (GSADF_{s0}, s0=0.4/0.8) are used here.
_KUROZUMI_SBAR = np.array([1, 1, 1, 3, 3, 3, 5, 5, 5])
_KUROZUMI_BETA = np.array([0.10, 0.05, 0.01] * 3)
_KUROZUMI_Q0_DF = np.array([0.6946, 1.0381, 1.6474, 1.0299, 1.3330, 1.8978, 1.1308, 1.4255, 1.9735])
_KUROZUMI_Q04_DF = np.array(
    [1.3969, 1.8081, 2.5927, 1.7088, 2.0737, 2.7677, 1.7988, 2.1480, 2.8276]
)
_KUROZUMI_Q08_DF = np.array(
    [1.9369, 2.3330, 3.0941, 2.1315, 2.4944, 3.2136, 2.1794, 2.5369, 3.2616]
)

# GSADF_{s0}(k)'s boundary function g_{s0}^df(k/m) = q * (a + b*log(c + k/m)).
_KUROZUMI_GSADF_ABC = {0.4: (0.76, 0.02, 0.34), 0.8: (0.73, 0.03, 0.90)}


def _sig_lvl_to_beta(sig_lvl: float) -> float:
    choices = (90, 95, 99)
    betas = (0.10, 0.05, 0.01)
    for c, b in zip(choices, betas, strict=True):
        if abs(sig_lvl - c) < 1e-8:
            return b
    raise ValueError(
        f"sig_lvl must be one of {choices} (Kurozumi (2020)'s Table 1 / Homm & Breitung "
        "(2012)'s Tables 7/8 only tabulate these significance levels)."
    )


def _kurozumi_sadf_q(sig_lvl: float, s_bar: float) -> float:
    """SADF boundary constant q_0^df, sig_lvl on the 0-100 scale, s_bar
    snapped to the nearest of Kurozumi's tabulated {1, 3, 5}."""
    beta = _sig_lvl_to_beta(sig_lvl)
    sbar_snap = min((1, 3, 5), key=lambda s: abs(s_bar - s))
    mask = (_KUROZUMI_SBAR == sbar_snap) & (np.abs(_KUROZUMI_BETA - beta) < 1e-8)
    return float(_KUROZUMI_Q0_DF[mask][0])


def _kurozumi_gsadf_q(sig_lvl: float, s_bar: float, s0: float) -> float:
    """GSADF_{s0} boundary constant, s0 snapped to the nearest of {0.4, 0.8}."""
    beta = _sig_lvl_to_beta(sig_lvl)
    sbar_snap = min((1, 3, 5), key=lambda s: abs(s_bar - s))
    s0_snap = min((0.4, 0.8), key=lambda s: abs(s0 - s))
    mask = (_KUROZUMI_SBAR == sbar_snap) & (np.abs(_KUROZUMI_BETA - beta) < 1e-8)
    return float((_KUROZUMI_Q04_DF if s0_snap == 0.4 else _KUROZUMI_Q08_DF)[mask][0])


def _kurozumi_gsadf_stat(y: np.ndarray, t_star: int, s0: float) -> np.ndarray:
    """Closed-form GSADF_{s0}(k) statistic path: max over window starts
    k1 in [0, floor(t_star*s0)) (0-indexed) of the with-intercept ADF
    t-statistic on window [k1, t], for every monitoring-period t =
    t_star, ..., n-1 (0-indexed). Same cumulative-sum-difference
    construction as exuber/R/monitor.R's kurozumi_gsadf_stat(), ported
    directly to 0-indexed numpy (prefix-sum arrays here have an extra
    leading 0, same trick as the R version)."""
    n = len(y)
    dy = np.diff(y)
    ylag = y[: n - 1]

    cs_x = np.concatenate(([0.0], np.cumsum(ylag)))
    cs_x2 = np.concatenate(([0.0], np.cumsum(ylag**2)))
    cs_y = np.concatenate(([0.0], np.cumsum(dy)))
    cs_y2 = np.concatenate(([0.0], np.cumsum(dy**2)))
    cs_xy = np.concatenate(([0.0], np.cumsum(ylag * dy)))

    k1_max = max(int(np.floor(t_star * s0)), 1)
    a_idx = np.arange(k1_max)  # window starts, 0-indexed: 0, ..., k1_max-1
    b_idx = np.arange(t_star - 1, n - 1)  # window ends (of `ylag`/`dy`), 0-indexed

    sx = cs_x[b_idx + 1, None] - cs_x[None, a_idx]
    sx2 = cs_x2[b_idx + 1, None] - cs_x2[None, a_idx]
    sy = cs_y[b_idx + 1, None] - cs_y[None, a_idx]
    sy2 = cs_y2[b_idx + 1, None] - cs_y2[None, a_idx]
    sxy = cs_xy[b_idx + 1, None] - cs_xy[None, a_idx]
    length = (b_idx[:, None] - a_idx[None, :]) + 1

    sxx_c = sx2 - sx**2 / length
    sxy_c = sxy - sx * sy / length
    syy_c = sy2 - sy**2 / length
    with np.errstate(invalid="ignore", divide="ignore"):
        beta = sxy_c / sxx_c
        ssr = syy_c - beta * sxy_c
        sigma2 = ssr / (length - 2)
        tstat = beta / np.sqrt(sigma2 / sxx_c)

    return np.nanmax(tstat, axis=1)


# Homm & Breitung (2012) Table 7(i): FLUC boundary constant b_{k,alpha},
# transcribed from exuber/R/monitor.R's hb_fluc_table (itself transcribed
# from a rendered PDF page). Tabulated at training length n in
# {20, 50, 100}, significance level alpha in {0.10, 0.05, 0.01}, and
# horizon ratio k in {2, 3, 4, 5, 6, 8, 10}.
_HB_K_GRID = (2, 3, 4, 5, 6, 8, 10)
_HB_N_GRID = (100, 50, 20)
_HB_FLUC_TABLE = {
    (100, 0.10): (3.05, 3.60, 3.93, 4.15, 4.31, 4.48, 4.57),
    (100, 0.05): (4.50, 5.14, 5.55, 5.69, 5.89, 6.05, 6.26),
    (100, 0.01): (7.76, 8.59, 9.06, 9.48, 9.62, 9.79, 9.99),
    (50, 0.10): (2.80, 3.33, 3.62, 3.80, 3.96, 4.14, 4.27),
    (50, 0.05): (4.19, 4.80, 5.11, 5.34, 5.50, 5.72, 5.81),
    (50, 0.01): (7.30, 8.11, 8.43, 8.82, 8.86, 9.25, 9.49),
    (20, 0.10): (2.49, 3.12, 3.44, 3.65, 3.78, 3.99, 4.12),
    (20, 0.05): (3.88, 4.56, 4.86, 5.06, 5.19, 5.38, 5.52),
    (20, 0.01): (7.00, 7.84, 8.26, 8.49, 8.66, 9.12, 9.19),
}


def _hb_fluc_q(sig_lvl: float, n_train: int, k: float) -> float:
    """b_{k,alpha}, n_train snapped to the nearest of {20, 50, 100} and k
    to the nearest of {2, ..., 10}."""
    beta = _sig_lvl_to_beta(sig_lvl)
    n_snap = min(_HB_N_GRID, key=lambda x: abs(n_train - x))
    k_snap = min(_HB_K_GRID, key=lambda x: abs(k - x))
    return _HB_FLUC_TABLE[(n_snap, beta)][_HB_K_GRID.index(k_snap)]


@dataclass
class MonitorResult:
    stat: np.ndarray  # (n_mon, nc): badf ("kurozumi" s0=0/"fluc") or the
    # GSADF_{s0} statistic ("kurozumi" s0>0, shared across series)
    boundary: np.ndarray  # (nc,) flat, or (n_mon,) k-varying when s0 > 0
    t_star: int
    alarm: np.ndarray  # (nc,), NaN where no breach occurred
    minw: int
    lag: int
    n: int
    sig_lvl: float
    boundary_type: str  # "kurozumi" or "fluc"
    s0: float
    series_names: list[str] | None = None


def monitor(
    data,
    r_star: float = 0.5,
    minw: int | None = None,
    sig_lvl: float = 95,
    lag: int = 0,
    boundary: str = "kurozumi",
    s0: float = 0,
) -> MonitorResult:
    """Real-time monitoring: fix a training window [0, T*) assumed free of
    exuberance, calibrate a boundary on it, then compare the running
    recursive statistic at each subsequent point against that fixed
    boundary, flagging the first breach.

    `boundary = "kurozumi"` (default) implements Kurozumi (2020)'s
    closed-form boundary: a published constant (his Table 1) compared
    against radf()'s `badf` sequence (his SADF(k) detector -- the s0=0,
    fixed-window-start case, the default). `s0 = 0.4` or `0.8` switches to
    his GSADF_{s0}(k) generalization: the window start ranges over
    [0, floor(T*s0)) instead of being fixed at 0, compared against his
    k-varying boundary function.

    `boundary = "fluc"` implements Homm & Breitung (2012)'s FLUC detector:
    their DF_{t/n} is likewise exactly radf()'s `badf` sequence, compared
    against a published constant from their Table 7.

    `sig_lvl` must be one of 90, 95, 99 (the levels both tables tabulate).
    """
    if boundary not in ("kurozumi", "fluc"):
        raise ValueError(
            "boundary must be 'kurozumi' or 'fluc' (boundary='bootstrap', Phillips & Shi "
            "(2020)'s own wild-bootstrap boundary, needs radf_wb_ps_cv(), not yet ported -- "
            "see docs/parity.md)"
        )
    x, columns = _to_2d_array(data)
    n, nc = x.shape
    minw = minw if minw is not None else psy_minw(n)
    if minw <= 2:
        raise ValueError("minw must be a positive integer greater than 2")
    t_star = _training_window(r_star, n, min_len=minw + lag + 1)

    if boundary == "kurozumi" and s0 > 0:
        s_bar = (n - t_star) / t_star
        q = _kurozumi_gsadf_q(sig_lvl, s_bar, s0)
        a, b, c = _KUROZUMI_GSADF_ABC[min(_KUROZUMI_GSADF_ABC, key=lambda s: abs(s - s0))]
        k_seq = np.arange(1, n - t_star + 1)
        boundary_path = q * (a + b * np.log(c + k_seq / t_star))

        stat_path = np.column_stack([_kurozumi_gsadf_stat(x[:, j], t_star, s0) for j in range(nc)])
        alarm = np.full(nc, np.nan)
        for j in range(nc):
            breach = np.where(stat_path[:, j] > boundary_path)[0]
            if breach.size:
                alarm[j] = t_star + breach[0]

        return MonitorResult(
            stat=stat_path,
            boundary=boundary_path,
            t_star=t_star,
            alarm=alarm,
            minw=minw,
            lag=lag,
            n=n,
            sig_lvl=sig_lvl,
            boundary_type="kurozumi",
            s0=s0,
            series_names=columns,
        )

    full: RadfResult = radf(x, minw=minw, lag=lag)
    zadj = minw + lag
    mon_from = max(t_star - zadj, 0)  # 0-indexed row into badf
    pointer = full.badf.shape[0]
    mon_rows = np.arange(mon_from, pointer)

    stat_path = full.badf
    if boundary == "kurozumi":
        s_bar = (n - t_star) / t_star
        q = _kurozumi_sadf_q(sig_lvl, s_bar)
    else:
        k = n / t_star
        q = _hb_fluc_q(sig_lvl, t_star, k)
    boundary_vec = np.full(nc, q)

    alarm = np.full(nc, np.nan)
    for j in range(nc):
        breach = np.where(stat_path[mon_rows, j] > boundary_vec[j])[0]
        if breach.size:
            alarm[j] = mon_rows[breach[0]] + zadj

    return MonitorResult(
        stat=stat_path,
        boundary=boundary_vec,
        t_star=t_star,
        alarm=alarm,
        minw=minw,
        lag=lag,
        n=n,
        sig_lvl=sig_lvl,
        boundary_type=boundary,
        s0=0,
        series_names=columns,
    )


# ---------------------------------------------------------------------------
# monitor_cusum() -- Homm & Breitung (2012)'s CUSUM real-time monitoring
# (Family B: a structurally different statistic from monitor()'s recursive
# ADF-family training-max -- a standardized running sum of first
# differences compared against a closed-form asymptotic boundary, no
# bootstrap, no C++). Astill, Harvey, Leybourne, Taylor & Zu (2023)'s
# volatility-robust one-sided-kernel "CUSUMV" variant (type="kernel") is
# included; their Corollary 1 establishes the same boundary function
# controls the false-alarm rate for both.
# ---------------------------------------------------------------------------

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
    """Same lookup-and-snap convention as _hb_fluc_q()."""
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


# ---------------------------------------------------------------------------
# lbi_test() + monitor_lbi() -- Breitung & Diegel (2025)'s locally best
# invariant (LBI) test: the static, full-sample statistic and its own
# sequential (constant-boundary CUSUM) monitoring extension. Both are the
# cheapest detectors in this module -- no bootstrap, no C++, and (for
# lbi_test) not even a table lookup: the null distribution is exactly
# standard normal (via stdlib's statistics.NormalDist, no scipy needed).
# ---------------------------------------------------------------------------


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
    from statistics import NormalDist

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


# ---------------------------------------------------------------------------
# quantile_test() -- Wu, Shi & Wu (2025)'s "global test": a quantile-
# regression (QR) analogue of the DF t-ratio, testing for a bubble via
# the tau-th conditional quantile of y_t on y_{t-1} instead of the
# conditional mean. See docs/alternative-paradigms.md, "Quantile-based
# detection". A single static test, not a recursive scan -- compare
# radf()'s single-shot `adf` statistic, not its recursive `bsadf`.
#
# QR solver: no simplex/interior-point LP package (statsmodels, scipy) is
# a pyexuber dependency, and adding one for a single-predictor fit is more
# than this module needs (ponytail: rung 5, prefer an already-installed
# dependency; none fits, and the fit itself is genuinely small). Instead,
# iteratively reweighted least squares (IRLS, Schlossmacher 1973's
# approach to the asymmetric L1/check-function loss) via plain numpy:
# converges to the exact QR solution for continuous data with no ties, to
# well within the tolerance this module's own tests use.
# ---------------------------------------------------------------------------


def _bw_nrd0(x: np.ndarray) -> float:
    """Port of R's stats::bw.nrd0() (the default density() bandwidth):
    0.9 * min(sd, IQR/1.34) * n^(-1/5)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    sd = np.std(x, ddof=1)
    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25
    hi = min(sd, iqr / 1.34) if iqr > 0 else sd
    return 0.9 * hi * n ** (-0.2) if hi > 0 else 1.0


def _dnorm(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)


def _quantile_check_density(u: np.ndarray, tau: float) -> tuple[float, float]:
    """b_tau (the tau-th sample quantile of u) and f_hat (a kernel density
    estimate of u's density at b_tau, via a Gaussian kernel with
    bw.nrd0's bandwidth) -- needed to studentize the QR coefficient the
    way OLS's residual-variance estimate studentizes the DF t-ratio."""
    b_tau = _quantile_narm(u, tau)
    h = _bw_nrd0(u)
    f_hat = np.mean(_dnorm((b_tau - u) / h)) / h
    return b_tau, f_hat


def _quantile_regression_fit(
    y: np.ndarray, x: np.ndarray, tau: float, max_iter: int = 200, tol: float = 1e-10
) -> tuple[float, float]:
    """Fit y = a + b*x at quantile `tau` via IRLS: repeatedly weighted
    least squares with weights 1/max(eps, |residual|) (asymmetric by
    tau/1-tau on each side), which minimizes the check-function loss at
    convergence. Returns (a, b)."""
    n = len(y)
    design = np.column_stack([np.ones(n), x])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]  # OLS start
    for _ in range(max_iter):
        resid = y - design @ beta
        w = np.where(resid >= 0, tau, 1 - tau) / np.maximum(np.abs(resid), 1e-8)
        wd = design * w[:, None]
        beta_new = np.linalg.solve(design.T @ wd, design.T @ (w * y))
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    return float(beta[0]), float(beta[1])


def _quantile_adf_tstat(y: np.ndarray) -> float:
    """Plain OLS ADF (intercept + one lag, no augmentation) t-statistic --
    used only to simulate quantile_test()'s Q component (the standard,
    demeaned Dickey-Fuller t-statistic distribution, verified in
    docs/alternative-paradigms.md to be bit-for-bit identical to radf()'s
    own single-shot `adf` field)."""
    n = len(y)
    dy = y[1:] - y[:-1]
    ylag = y[:-1]
    design = np.column_stack([np.ones(n - 1), ylag])
    beta, _, _, _ = np.linalg.lstsq(design, dy, rcond=None)
    resid = dy - design @ beta
    dof = (n - 1) - 2
    sigma2 = np.sum(resid**2) / dof
    se = np.sqrt(sigma2 * np.linalg.inv(design.T @ design)[1, 1])
    return float(beta[1] / se)


def _quantile_test_q_distribution(n: int, nrep: int, rng: np.random.Generator) -> np.ndarray:
    return np.array([_quantile_adf_tstat(np.cumsum(rng.normal(size=n))) for _ in range(nrep)])


@dataclass
class QuantileTestResult:
    tstat: np.ndarray  # (nc,)
    tau: np.ndarray  # (nc,), the tau actually used per series
    delta: np.ndarray  # (nc,)
    crit: np.ndarray  # (nc,)
    detected: np.ndarray  # (nc,) bool
    n: int
    sig_lvl: float
    iter: int
    series_names: list[str] | None = None


def quantile_test(
    data,
    tau: float | str = "optimal",
    tau_grid: np.ndarray | None = None,
    nrep: int = 1000,
    sig_lvl: float = 95,
    seed: int | None = None,
) -> QuantileTestResult:
    """Wu, Shi & Wu (2025)'s "global test": a quantile-regression (QR)
    analogue of the Dickey-Fuller t-ratio, testing for explosive behavior
    at a chosen conditional quantile `tau` of y_t on y_{t-1} rather than
    at the conditional mean. A single static test (compare radf()'s
    single-shot `adf`, not its recursive `bsadf`).

    `tau = "optimal"` (default) selects the quantile minimizing the
    asymptotic variance of the QR estimator (their eq. 33) by grid search
    over `tau_grid` (default 0.2 to 0.8 in steps of 0.05, the paper's own
    recommended practical range).

    The critical value is simulated per call: the statistic's limiting
    null distribution is sqrt(1 - delta^2) * z + delta * Q, z ~ N(0, 1),
    delta a data-estimated correlation coefficient, and Q the standard
    demeaned Dickey-Fuller t-statistic distribution (simulated the same
    random-walk-plus-OLS-t-stat way radf_mc_cv() simulates its own `adf`
    critical value).

    RNG note: uses numpy's Generator, not R's -- a given `seed` will not
    reproduce the same critical-value draws as R's quantile_test() (the
    point statistics tstat/tau/delta themselves match bit-for-bit).
    """
    _assert_sig_lvl(sig_lvl)
    if tau_grid is None:
        tau_grid = np.round(np.arange(0.20, 0.80 + 1e-9, 0.05), 10)
    x, columns = _to_2d_array(data)
    n, nc = x.shape
    rng = np.random.default_rng(seed)

    q_dist = _quantile_test_q_distribution(n, nrep, rng)

    tstat = np.empty(nc)
    crit = np.empty(nc)
    delta = np.empty(nc)
    tau_used = np.empty(nc)
    detected = np.empty(nc, dtype=bool)

    for j in range(nc):
        y = x[:, j]
        dy = y[1:] - y[:-1]
        ylag = y[:-1]

        tau_j: float
        if tau == "optimal":
            fhats = np.array([_quantile_check_density(dy, tt)[1] for tt in tau_grid])
            obj = (tau_grid * (1 - tau_grid)) / fhats**2
            tau_j = float(tau_grid[np.argmin(obj)])
        else:
            tau_j = float(tau)
        if not (0 < tau_j < 1):
            raise ValueError("tau must be in (0, 1) or 'optimal'.")

        _, alpha_hat = _quantile_regression_fit(y[1:], ylag, tau_j)

        _, f_hat = _quantile_check_density(dy, tau_j)
        y_pzy = np.sum((ylag - np.mean(ylag)) ** 2)
        tstat_j = (f_hat / np.sqrt(tau_j * (1 - tau_j))) * np.sqrt(y_pzy) * (alpha_hat - 1)

        psi = tau_j - (dy < _quantile_narm(dy, tau_j)).astype(float)
        delta_j = float(np.clip(np.corrcoef(dy, psi)[0, 1], -1, 1))

        z = rng.normal(size=nrep)
        u = np.sqrt(1 - delta_j**2) * z + delta_j * q_dist
        crit_j = _quantile_narm(u, sig_lvl / 100)

        tstat[j] = tstat_j
        crit[j] = crit_j
        delta[j] = delta_j
        tau_used[j] = tau_j
        detected[j] = tstat_j > crit_j

    return QuantileTestResult(
        tstat=tstat,
        tau=tau_used,
        delta=delta,
        crit=crit,
        detected=detected,
        n=n,
        sig_lvl=sig_lvl,
        iter=nrep,
        series_names=columns,
    )
