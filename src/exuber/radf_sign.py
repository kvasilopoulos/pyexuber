"""Sign-based sGSADF/s-bar-GSADF bubble tests (Harvey, Leybourne & Zu
2020). Ported from exuber's R/radf_sign.R (R keeps sign_dm in the same
file -- matches). See docs/volatility-robustness.md (root repo) for the
formulas, papers and independent validation this ports.

RNG note: uses numpy's Generator, not R's RNG -- see sim.py's module
docstring; a given `seed` will not reproduce the same draws as R's
radf_sign_cv()/radf_sign_dm_cv().
"""

import numpy as np

from exuber._gls_dfstat import _grid_to_radf_result, _mc_cv_grid
from exuber.cv import RadfCv
from exuber.radf import RadfResult, _to_2d_array, psy_minw


def sign_transform(y: np.ndarray) -> np.ndarray:
    """C_t := sum_{i<=t} sign(Delta y_i) -- exactly invariant to the
    (even time-varying) volatility of Delta y_t."""
    y = np.asarray(y, dtype=float)
    return np.concatenate(([0.0], np.cumsum(np.sign(np.diff(y)))))


def sign_demean_transform(y: np.ndarray) -> np.ndarray:
    """HLZ (2020)'s second sign-based analogue (HLTZ 2025's s-bar):
    Ctilde_t := sum_{i=2}^t {sign(dy_i) - (i-1)^-1 * sum_{j=2}^i sign(dy_j)}
    -- an expanding-window (not per-window) demeaning, so the demeaning
    term at each i is simply C_i / (i - 1), an O(T) one-time transform."""
    y = np.asarray(y, dtype=float)
    s = np.sign(np.diff(y))
    n = len(s)
    cs = np.cumsum(s)
    return np.concatenate(([0.0], np.cumsum(s - cs / np.arange(1, n + 1))))


def radf_sign(data, minw: int | None = None) -> RadfResult:
    """Sign-based bubble test (sPWY/sPSY), Harvey, Leybourne & Zu (2020):
    the recursive sup-ADF statistic applied to the cumulated sign of the
    first differences, exactly invariant to (even time-varying)
    heteroskedasticity -- no bootstrap needed. Pair with `radf_sign_cv()`.
    """
    x, columns = _to_2d_array(data)
    minw = minw if minw is not None else psy_minw(x.shape[0])
    return _grid_to_radf_result(x, minw, columns, sign_transform)


def radf_sign_cv(
    n: int, minw: int | None = None, nrep: int = 2000, seed: int | None = None
) -> RadfCv:
    """Monte Carlo critical values for `radf_sign()`. Per Theorem 2 of
    Harvey, Leybourne & Zu (2020), pivotal -- simulated once under a plain
    random walk. `sadf_cv`/`gsadf_cv` (minw/n = 0.1) can be checked
    against the paper's Table 1 asymptotic sPWY/sPSY rows: (2.410, 2.734,
    3.248) and (2.933, 3.180, 3.655).
    """
    return _mc_cv_grid(n, minw, nrep, seed, sign_transform, method="Sign-Based MC")


def radf_sign_dm(data, minw: int | None = None) -> RadfResult:
    """Recursively demeaned sign-based bubble test (s-bar-PWY/s-bar-PSY),
    Harvey, Leybourne & Zu (2020)'s second sign-based analogue. Shares
    `radf_sign()`'s exact heteroskedasticity invariance and (per Harvey,
    Leybourne, Tatlow & Zu 2025) its level-shift robustness, under a
    strictly weaker assumption (no zero-median requirement). Pair with
    `radf_sign_dm_cv()`.
    """
    x, columns = _to_2d_array(data)
    minw = minw if minw is not None else psy_minw(x.shape[0])
    return _grid_to_radf_result(x, minw, columns, sign_demean_transform)


def radf_sign_dm_cv(
    n: int, minw: int | None = None, nrep: int = 2000, seed: int | None = None
) -> RadfCv:
    """Monte Carlo critical values for `radf_sign_dm()`. Pivotal like
    `radf_sign_cv()`, simulated once under a plain random walk."""
    return _mc_cv_grid(
        n, minw, nrep, seed, sign_demean_transform, method="Sign-Based MC (demeaned)"
    )
