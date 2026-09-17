"""Kernel spot-volatility estimator, shared by radf_kp() (kernel-purge)
and radf_sbz()/radf_sbz_union() (SBZ) -- both need the same Nadaraya-
Watson spot-variance estimate of the first differences, so it lives here
rather than in either module.
"""

import numpy as np


def nw_spot_vol(
    e: np.ndarray, kernel: str = "gaussian", h: float | None = None
) -> tuple[np.ndarray, float]:
    """Nadaraya-Watson kernel smoother of a squared innovation series `e`,
    with leave-one-out cross-validated bandwidth over [1/(2T), 1/6]
    (Harvey, Leybourne & Zu 2019, footnote 2) when `h` is not given.
    Returns (sigma2, h), sigma2 one value per t = 1..T (t/T grid)."""
    if kernel not in ("gaussian", "uniform"):
        raise ValueError("kernel must be 'gaussian' or 'uniform'")
    e = np.asarray(e, dtype=float)
    tn = len(e)
    s = np.arange(2, tn + 1) / tn  # i/T for i = 2..T
    e2 = e[1:] ** 2  # e_i^2 for i = 2..T, aligned with s
    t_grid = np.arange(1, tn + 1) / tn

    def kern(u: np.ndarray) -> np.ndarray:
        if kernel == "gaussian":
            return np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi)
        return (np.abs(u) <= 1).astype(float) / 2

    def spot_vol_at(hh: float, drop_self: bool = False) -> np.ndarray:
        out = np.empty(len(t_grid))
        for j, tg in enumerate(t_grid):
            w = kern((s - tg) / hh)
            if drop_self:
                self_idx = np.abs(s - tg) < np.sqrt(np.finfo(float).eps)
                w = np.where(self_idx, 0.0, w)
            wsum = w.sum()
            out[j] = np.mean(e2) if wsum <= 0 else np.sum(w * e2) / wsum
        return out

    if h is None:
        hl = 1 / (2 * tn)
        hu = 1 / 6
        grid = np.exp(np.linspace(np.log(hl), np.log(hu), 10))
        s_to_tgrid_idx = np.searchsorted(t_grid, s)  # s values sit exactly on t_grid
        cv = np.empty(len(grid))
        for k, hh in enumerate(grid):
            s2_loo = spot_vol_at(hh, drop_self=True)
            cv[k] = np.mean((e2 - s2_loo[s_to_tgrid_idx]) ** 2)
        h = float(grid[np.argmin(cv)])

    sigma2 = spot_vol_at(h, drop_self=False)
    return sigma2, h


def kernel_spot_vol(
    y: np.ndarray, kernel: str = "gaussian", h: float | None = None
) -> tuple[np.ndarray, float]:
    """Nonparametric spot-volatility estimator, eq. (6) of Harvey,
    Leybourne & Zu (2019): `nw_spot_vol()` applied to the raw series'
    first differences."""
    return nw_spot_vol(np.diff(np.asarray(y, dtype=float)), kernel=kernel, h=h)
