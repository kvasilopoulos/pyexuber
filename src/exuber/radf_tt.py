"""STADF/GSTADF -- time-transformed test for explosive bubbles under
non-stationary volatility (Kurozumi, Skrobotov & Tsarev 2024). Ported
from exuber's R/radf_tt.R -- see docs/volatility-robustness.md (root
repo) for the formulas, papers and independent validation this ports.

RNG note: uses numpy's Generator, not R's RNG -- see sim.py's module
docstring; a given `seed` will not reproduce the same draws as R's
radf_tt_cv().
"""

import numpy as np

from exuber._gls_dfstat import _grid_to_radf_result, _mc_cv_grid
from exuber.cv import RadfCv
from exuber.radf import RadfResult, _to_2d_array, psy_minw


def variance_profile(
    y: np.ndarray, kernel: str = "uniform", h: float | None = None
) -> tuple[np.ndarray, float]:
    """Nonparametric variance profile estimate, eq. (18)-(19) of Kurozumi,
    Skrobotov & Tsarev (2024): local (Nadaraya-Watson-type) no-intercept
    kernel regression of the time-varying AR(1) coefficient, truncated
    residuals, cumulative-sum-of-squares profile. Returns (eta_grid,
    omega2); eta_grid has length Tn + 1 (Tn = len(y) - 1)."""
    if kernel not in ("uniform", "gaussian"):
        raise ValueError("kernel must be 'uniform' or 'gaussian'")
    y = np.asarray(y, dtype=float)
    yc = y - y[0]
    tn = len(yc) - 1
    dy = np.diff(yc)
    ylag = yc[:tn]

    h = h if h is not None else tn ** (-2 / 5)
    idx = np.arange(1, tn + 1)

    def kern(u: np.ndarray) -> np.ndarray:
        if kernel == "uniform":
            return (np.abs(u) <= 1).astype(float)
        return np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi)

    delta = np.empty(tn)
    eps = np.empty(tn)
    for t in range(1, tn + 1):
        w = kern((idx - t) / (tn * h))
        sw_xx = np.sum(w * ylag**2)
        sw_xy = np.sum(w * ylag * dy)
        delta[t - 1] = sw_xy / sw_xx if sw_xx > 0 else 0.0
        eps[t - 1] = dy[t - 1] - delta[t - 1] * ylag[t - 1]

    # Truncation threshold psi_T (footnote 6): max local-window residual sd.
    win = max(2, round(0.1 * tn))
    n_starts = max(1, round(0.9 * tn))
    cbar = max(
        np.std(eps[a : min(a + win + 1, tn)], ddof=1)
        for a in range(n_starts)
        if min(a + win + 1, tn) - a > 1
    )
    psi_t = cbar * tn ** (1 / 7)

    eps_star = np.where(np.abs(eps) >= psi_t, 0.0, eps)
    cs = np.cumsum(eps_star**2)
    total = cs[-1]
    omega2 = total / tn
    eta_grid = np.concatenate(([0.0], cs)) / total
    return eta_grid, omega2


def _time_transform(y: np.ndarray, kernel: str, h: float | None) -> np.ndarray:
    eta_grid, _ = variance_profile(y, kernel=kernel, h=h)
    tn = len(y) - 1
    tgrid = np.linspace(0, 1, tn + 1)
    # Generalized inverse of the monotone, piecewise-linear eta_hat via
    # linear interpolation (exact since eta_hat is piecewise-linear on the
    # observation grid), matching R's approx(eta_grid, tgrid, xout = tgrid).
    s = np.clip(tgrid, 0.0, 1.0)
    g_of_s = np.interp(s, eta_grid, tgrid)
    resampled_idx = np.clip(np.round(g_of_s * tn).astype(int), 0, len(y) - 1)
    return y[resampled_idx]


def radf_tt(
    data, minw: int | None = None, kernel: str = "uniform", h: float | None = None
) -> RadfResult:
    """Time-transformed test for explosive bubbles under non-stationary
    volatility (STADF/GSTADF), Kurozumi, Skrobotov & Tsarev (2024): the
    series is time-deformed using a nonparametric variance-profile
    estimate, after which the usual (homoskedastic) recursive sup-ADF
    critical values apply -- no bootstrap needed. Pair with
    `radf_tt_cv()`.
    """
    x, columns = _to_2d_array(data)
    nr, nc = x.shape
    minw = minw if minw is not None else psy_minw(nr)

    transformed = np.column_stack(
        [_time_transform(x[:, j], kernel, h) for j in range(nc)]
    )
    return _grid_to_radf_result(transformed, minw, columns, lambda col: col)


def radf_tt_cv(
    n: int, minw: int | None = None, nrep: int = 2000, seed: int | None = None
) -> RadfCv:
    """Monte Carlo critical values for `radf_tt()`. Per Theorem 1 of
    Kurozumi, Skrobotov & Tsarev (2024), the null limiting distribution is
    pivotal (free of the volatility process), so this is simulated once
    under a plain random walk rather than per dataset. `sadf_cv` (STADF)
    can be checked against Whitehouse (2019)'s published asymptotic
    triple, quoted in the paper's footnote 4: for minw/n = 0.1, (10%, 5%,
    1%) = (2.319, 2.626, 3.223).
    """
    return _mc_cv_grid(n, minw, nrep, seed, lambda y: y, method="Time-Transformed MC")
