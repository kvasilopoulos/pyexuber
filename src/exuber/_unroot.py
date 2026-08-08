import numpy as np


def _embed(x: np.ndarray, k: int) -> np.ndarray:
    """Port of R's embed(x, k): row i holds x[i+k-1], x[i+k-2], ..., x[i]."""
    n = len(x)
    return np.column_stack([x[k - 1 - i : n - i] for i in range(k)])


def unroot(x: np.ndarray, lag: int = 0) -> np.ndarray:
    """Port of exuber's R/unroot.R:unroot() -- builds the regression matrix
    exubercore::radf() expects: column 0 is the dependent variable, the rest
    are regressors (constant, lag, lagged differences for lag > 0)."""
    x = np.asarray(x, dtype=float)
    if lag == 0:
        e = _embed(x, 2)
        return np.column_stack([e[:, 0], e[:, 1]])

    x_embed = _embed(x, lag + 2)
    dx_embed = _embed(np.diff(x), lag + 1)[:, 1:]
    x_lev = x_embed[:, 0]
    x_lag = x_embed[:, 1]
    ct = np.ones_like(x_lev)
    return np.column_stack([x_lev, ct, x_lag, dx_embed])
