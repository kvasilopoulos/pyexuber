"""Kernel-purge test (Harvey, Leybourne, Taylor & Zu 2024). Ported from
exuber's R/radf_kp.R -- see docs/volatility-robustness.md (root repo)
for the formulas, papers and independent validation this ports.
"""

import numpy as np

from exuber._kernel_vol import kernel_spot_vol
from exuber.radf import RadfResult, _to_2d_array


def kernel_purge(y: np.ndarray, kernel: str = "gaussian", h: float | None = None) -> np.ndarray:
    """Kernel-purged transform (eq. 4-5): x_t = cumsum(Delta y_t /
    sigma_hat_t). Feed the result to `radf()` -- the purged statistic's
    null distribution is proven identical to the standard homoskedastic
    GSADF null (Theorem 1/Remark 3.2 of Harvey, Leybourne, Taylor & Zu
    2024), so no new critical-value machinery is needed."""
    y = np.asarray(y, dtype=float)
    tn = len(y) - 1
    h = h if h is not None else 0.1 * tn ** (-0.25)
    sigma2, _ = kernel_spot_vol(y, kernel=kernel, h=h)
    return np.cumsum(np.diff(y) / np.sqrt(sigma2))


def radf_kp(
    data, minw: int | None = None, kernel: str = "gaussian", h: float | None = None
) -> RadfResult:
    """Kernel-purged heteroskedasticity-robust PSY test, Harvey, Leybourne,
    Taylor & Zu (2024): "purges" unconditional heteroskedasticity by
    cumulating the series' first differences after dividing each by a
    kernel spot-volatility estimate, then runs the ordinary
    (with-intercept) `radf()` on the purged series. `radf_mc_cv()` applies
    directly to the result (see module docstring / Details above)."""
    from exuber.radf import radf

    x, columns = _to_2d_array(data)
    nc = x.shape[1]
    purged = np.column_stack([kernel_purge(x[:, j], kernel=kernel, h=h) for j in range(nc)])
    if columns is not None:
        result = radf(purged, minw=minw)
        result.series_names = columns
        return result
    return radf(purged, minw=minw)
