"""Shared monitoring helpers: ports of exuber's R/utils-defensive.R
(training_window(), assert_sig_lvl(), quantile_narm()), plus the Homm &
Breitung (2012) sampling grid ({20, 50, 100} training lengths x
{2, ..., 10} horizon ratios) shared by monitor.py's FLUC table and
monitor_cusum.py's finite-boundary table.
"""

import numpy as np


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


# Homm & Breitung (2012) Tables 7/8: tabulated at training length n in
# {20, 50, 100}, significance level alpha in {0.10, 0.05, 0.01}, and
# horizon ratio k in {2, 3, 4, 5, 6, 8, 10}. Shared grid for monitor.py's
# FLUC table and monitor_cusum.py's finite-boundary table.
_HB_K_GRID = (2, 3, 4, 5, 6, 8, 10)
_HB_N_GRID = (100, 50, 20)
