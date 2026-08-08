from dataclasses import dataclass

import numpy as np

from exuber._unroot import unroot


def psy_minw(n: int) -> int:
    """Default minimum window: (0.01 + 1.8/sqrt(n)) * n, floored."""
    return int(np.floor((0.01 + 1.8 / np.sqrt(n)) * n))


def psy_ds(n: int, rule: int = 1, delta: float = 1.0) -> int:
    """Minimum duration for a datestamp() episode. rule=1: delta*log(n);
    rule=2: delta*log(n)/n."""
    if rule not in (1, 2):
        raise ValueError("rule must be 1 or 2")
    if delta <= 0:
        raise ValueError("delta must be positive")
    value = delta * np.log(n) if rule == 1 else delta * np.log(n) / n
    return int(np.round(value))


def _to_2d_array(data) -> tuple[np.ndarray, list[str] | None]:
    """Accepts numpy ndarray, a 1-D sequence, or anything exposing
    to_numpy()/columns (pandas.DataFrame, polars.DataFrame) without
    requiring either package to be installed."""
    columns = None
    if hasattr(data, "to_numpy"):
        columns = list(getattr(data, "columns", [])) or None
        data = data.to_numpy()
    x = np.asarray(data, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    return x, columns


@dataclass
class RadfResult:
    adf: np.ndarray
    badf: np.ndarray
    sadf: np.ndarray
    bsadf: np.ndarray
    gsadf: np.ndarray
    bsadf_panel: np.ndarray
    gsadf_panel: float
    minw: int
    lag: int
    n: int
    series_names: list[str] | None = None


def radf(data, minw: int | None = None, lag: int = 0) -> RadfResult:
    """Recursive univariate and panel Augmented Dickey-Fuller test.

    Mirrors exuber's R radf(data, minw, lag): same statistics (adf, badf,
    sadf, bsadf, gsadf, bsadf_panel, gsadf_panel), computed via the same
    exubercore::radf() routine exuber uses.
    """
    from . import _core  # lazy: keeps the dataclasses/psy_minw importable
    # without the compiled extension (e.g. for pure-Python unit tests).

    x, columns = _to_2d_array(data)
    n, nc = x.shape
    minw = minw if minw is not None else psy_minw(n)
    pointer = n - minw - lag

    adf = np.zeros(nc)
    sadf = np.zeros(nc)
    gsadf = np.zeros(nc)
    badf = np.zeros((pointer, nc))
    bsadf = np.zeros((pointer, nc))

    for i in range(nc):
        yxmat = unroot(x[:, i], lag=lag)
        result = _core.radf_stat(yxmat, minw, lag)
        badf[:, i] = result[:pointer]
        adf[i] = result[pointer]
        sadf[i] = result[pointer + 1]
        gsadf[i] = result[pointer + 2]
        bsadf[:, i] = result[pointer + 3 :]

    bsadf_panel = bsadf.mean(axis=1)
    gsadf_panel = float(bsadf_panel.max())

    return RadfResult(
        adf=adf,
        badf=badf,
        sadf=sadf,
        bsadf=bsadf,
        gsadf=gsadf,
        bsadf_panel=bsadf_panel,
        gsadf_panel=gsadf_panel,
        minw=minw,
        lag=lag,
        n=n,
        series_names=columns,
    )
