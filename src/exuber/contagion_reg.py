"""Contagion regression (Greenaway-McGrevy & Phillips 2016). Ported from
exuber's R/contagion_reg.R -- see docs/multivariate.md for the full
evaluation this implements.

Section 2.6's functional (time-varying) coefficient regression: a
fixed-width rolling-window AR(1) coefficient sequence for a "core"
series and a "satellite" series y (eq. 1), related by a Nadaraya-Watson
kernel regression at a chosen delay d (eq. 6), with the bandwidth
selected by leave-one-out cross-validation (eq. 7). Minimum-viable
subset: eq. 8's automatic delay search is not implemented -- call
contagion_reg() once per candidate d and compare fit if that's needed.
The source paper performs no formal inference on the coefficient itself
(point estimation/plotting only), so there is no critical value here.
"""

from dataclasses import dataclass

import numpy as np


def _contagion_prefix_sums(
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    n1 = len(y) - 1
    x = y[:n1]
    z = y[1 : n1 + 1]
    cx = np.concatenate(([0.0], np.cumsum(x)))
    cx2 = np.concatenate(([0.0], np.cumsum(x**2)))
    cz = np.concatenate(([0.0], np.cumsum(z)))
    cxz = np.concatenate(([0.0], np.cumsum(x * z)))
    return cx, cx2, cz, cxz, n1


def _contagion_fixed_window_beta(y: np.ndarray, S: int) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-window OLS AR(1) slope sequence (eq. 1): window {t = s-S+1,
    ..., s} is S LEVEL observations, i.e. S-1 (response, lag) pairs.
    Returns (t, beta) parallel arrays: t is the window's own end date
    (1-indexed, matching R's convention), for every window-end
    s = S, ..., len(y)."""
    cx, cx2, cz, cxz, n1 = _contagion_prefix_sums(y)
    hi = np.arange(S, n1 + 1)
    lo = hi - (S - 1)
    Sx = cx[hi] - cx[lo]
    Sxx = cx2[hi] - cx2[lo]
    Sz = cz[hi] - cz[lo]
    Sxz = cxz[hi] - cxz[lo]
    n_seg = S - 1
    beta = (n_seg * Sxz - Sx * Sz) / (n_seg * Sxx - Sx**2)
    return hi + 1, beta


def _contagion_kernel_weights(s: np.ndarray, T_len: int, r: np.ndarray, h: float) -> np.ndarray:
    """eq. 6's Gaussian kernel weight K_hs(r) = (1/h)*K((s/T-r)/h), for
    every (position, evaluation point) pair -- rows are positions `s`,
    columns are evaluation points `r`."""
    z = (s[:, None] / T_len - r[None, :]) / h
    return np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi) / h


def _align_shifted(
    t_core: np.ndarray, bcore_c: np.ndarray, t_j: np.ndarray, bj_c: np.ndarray, d: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aligns beta_j at date s with beta_core at date s-d (both centered),
    dropping s where s-d falls outside beta_core's own date range --
    R's named-vector indexing, done here via an index lookup."""
    pos = {int(t): i for i, t in enumerate(t_core)}
    idx = np.array([pos.get(int(t) - d, -1) for t in t_j])
    valid = idx >= 0
    return t_j[valid], bj_c[valid], bcore_c[idx[valid]]


def _contagion_nw_delta2(
    t_core: np.ndarray,
    beta_core: np.ndarray,
    t_j: np.ndarray,
    beta_j: np.ndarray,
    T_len: int,
    r_grid: np.ndarray,
    h: float,
    d: int,
) -> np.ndarray:
    """eq. 6: delta_2j(r; h, d), the Nadaraya-Watson local-constant kernel
    regression coefficient -- a closed-form ratio of two kernel-weighted
    sums (a no-intercept WLS solution). Centering (beta-tilde) uses each
    series' own full range, before any delay shift."""
    bj_c = beta_j - beta_j.mean()
    bcore_c = beta_core - beta_core.mean()
    s, bj_c, core_shift = _align_shifted(t_core, bcore_c, t_j, bj_c, d)

    K = _contagion_kernel_weights(s, T_len, r_grid, h)
    num = K.T @ (bj_c * core_shift)
    den = K.T @ (core_shift**2)
    return num / den


def _contagion_loocv_sse(
    h: float,
    t_core: np.ndarray,
    beta_core: np.ndarray,
    t_j: np.ndarray,
    beta_j: np.ndarray,
    T_len: int,
    d: int,
) -> float:
    """eq. 7's leave-one-out SSE at bandwidth h, delay d: evaluates the
    LOO version of eq. 6 at r = s/m for each available s (excluding the
    point itself from the kernel sum), sums squared prediction errors."""
    bj_c = beta_j - beta_j.mean()
    bcore_c = beta_core - beta_core.mean()
    s, bj_c, core_shift = _align_shifted(t_core, bcore_c, t_j, bj_c, d)

    m = len(s)
    r = s / m
    K = _contagion_kernel_weights(s, T_len, r, h)
    np.fill_diagonal(K, 0.0)
    num = K.T @ (bj_c * core_shift)
    den = K.T @ (core_shift**2)
    pred = (num / den) * core_shift
    return float(np.sum((bj_c - pred) ** 2))


def _golden_section_min(f, lo: float, hi: float, tol: float = 1e-5, max_iter: int = 100) -> float:
    """Bounded 1-D minimization -- exuber's R source uses stats::optimize()
    (golden section + successive parabolic interpolation); numpy has no
    bounded scalar minimizer built in, so this is the plain golden-section
    half of that, sufficient for eq. 7's unimodal-in-practice LOOCV SSE."""
    phi = (np.sqrt(5) - 1) / 2
    a, b = lo, hi
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = f(d)
    return (a + b) / 2


def _contagion_bandwidth_cv(
    t_core: np.ndarray, beta_core: np.ndarray, t_j: np.ndarray, beta_j: np.ndarray,
    T_len: int, d: int,
) -> float:
    """eq. 7: bandwidth via 1-D bounded LOOCV, H_T = [m^-1/2, m^-1/10]."""
    m = len(beta_j)
    H_T = (m ** (-1 / 2), m ** (-1 / 10))
    return _golden_section_min(
        lambda hh: _contagion_loocv_sse(hh, t_core, beta_core, t_j, beta_j, T_len, d),
        H_T[0], H_T[1],
    )


@dataclass
class ContagionRegResult:
    beta_core: np.ndarray
    beta_j: np.ndarray
    h: float
    d: int
    r_grid: np.ndarray
    delta2: np.ndarray
    n: int
    S: int


def contagion_reg(
    y,
    core,
    S: int | None = None,
    d: int = 0,
    h: float | None = None,
    r_grid=None,
) -> ContagionRegResult:
    """Bubble contagion regression (Greenaway-McGrevy & Phillips 2016).

    Estimates the time-varying contagion coefficient: a fixed-window
    rolling AR(1) coefficient sequence for a "core" series and a
    "satellite" series y, related by a Nadaraya-Watson kernel regression
    at a chosen delay d -- how strongly, and how (time-varying), the
    core's local persistence transmits to y, d periods later.

    Minimum-viable subset: the fixed-window AR(1) sequence (eq. 1), the
    Nadaraya-Watson regression at a single supplied d (eq. 6), and
    leave-one-out cross-validated bandwidth selection (eq. 7). eq. 8's
    automatic delay search is not implemented -- call this once per
    candidate d and compare fit. Not a hypothesis test: the source paper
    performs no formal inference on the coefficient (no confidence bands,
    no significance test) -- neither does this, by design.

    y: satellite (dependent) series. core: reference series, same length.
    S: fixed rolling-window width (default floor(0.33*len(y))).
    d: non-negative integer delay (default 0).
    h: bandwidth; None selects it via leave-one-out CV (eq. 7).
    r_grid: evaluation points as fractions of the sample (default 100
    points over [0, 1]).
    """
    y = np.asarray(y, dtype=float)
    core = np.asarray(core, dtype=float)
    if len(y) != len(core):
        raise ValueError("'y' and 'core' must be the same length.")
    if d < 0:
        raise ValueError("d must be non-negative")
    T_len = len(y)
    S = S if S is not None else max(int(0.33 * T_len), 5)
    if S <= 2:
        raise ValueError("S must be greater than 2")
    r_grid = np.linspace(0, 1, 100) if r_grid is None else np.asarray(r_grid, dtype=float)

    t_j, beta_j = _contagion_fixed_window_beta(y, S)
    t_core, beta_core = _contagion_fixed_window_beta(core, S)

    if h is None:
        h = _contagion_bandwidth_cv(t_core, beta_core, t_j, beta_j, T_len, d)
    delta2 = _contagion_nw_delta2(t_core, beta_core, t_j, beta_j, T_len, r_grid, h, d)

    return ContagionRegResult(
        beta_core=beta_core, beta_j=beta_j, h=float(h), d=int(d), r_grid=r_grid,
        delta2=delta2, n=T_len, S=S,
    )
