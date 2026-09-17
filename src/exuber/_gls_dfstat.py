"""Shared no-intercept recursive-DF machinery (GLS-demeaned STADF family),
used by radf_tt() and radf_sign()/radf_sign_dm() -- both apply the same
gls_dfstat_grid() to a differently-transformed series, so the shared
grid/critical-value drivers live here rather than in either module.
"""

import numpy as np

from exuber.cv import PCNT, RadfCv
from exuber.radf import RadfResult, psy_minw


def gls_dfstat_grid(y: np.ndarray, minw: int) -> dict:
    """GLS-demeaned (no-intercept) recursive sup-ADF family, eq. (9) of
    Kurozumi, Skrobotov & Tsarev (2024). `y` is a levels series; internally
    demeaned by its first observation (not OLS-demeaned the way radf()'s
    own unroot()/exubercore regression is). Fully vectorized over the
    (r1, r2) grid via cumulative sums, same construction as exuber's R
    gls_dfstat_grid() -- see that function's own comments for the formula
    derivation. Returns the same badf/bsadf/adf/sadf/gsadf shape radf()
    does, one row per candidate window end (length n - minw)."""
    y = np.asarray(y, dtype=float)
    yc = y - y[0]
    n1 = len(yc) - 1
    dy = np.diff(yc)
    ylag = yc[:n1]

    if minw >= n1:
        raise ValueError("minw must be smaller than the number of observations minus one")

    csxx = np.concatenate(([0.0], np.cumsum(ylag**2)))
    csxy = np.concatenate(([0.0], np.cumsum(ylag * dy)))
    csyy = np.concatenate(([0.0], np.cumsum(dy**2)))

    b_idx = np.arange(minw, n1 + 1)  # window ends
    a_idx = np.arange(1, n1 + 1)  # window starts (exclusive lower bound)

    sxx = csxx[b_idx][:, None] - csxx[a_idx - 1][None, :]
    sxy = csxy[b_idx][:, None] - csxy[a_idx - 1][None, :]
    syy = csyy[b_idx][:, None] - csyy[a_idx - 1][None, :]
    length = (b_idx[:, None] - a_idx[None, :] + 1).astype(float)

    valid = length >= minw
    sxx = np.where(valid, sxx, np.nan)

    with np.errstate(invalid="ignore", divide="ignore"):
        beta = sxy / sxx
        ssr = syy - beta * sxy
        sigma2 = ssr / (length - 1)
        tstat = sxy / np.sqrt(sigma2 * sxx)

    valid = valid & np.isfinite(tstat)
    tstat = np.where(valid, tstat, np.nan)

    badf = tstat[:, 0].copy()
    adf = badf[-1]
    sadf = float(np.nanmax(badf))

    tstat_filled = np.where(valid, tstat, -np.inf)
    best_col = np.argmax(tstat_filled, axis=1)
    bsadf = tstat_filled[np.arange(tstat_filled.shape[0]), best_col]
    gsadf = float(np.max(bsadf))

    return {"badf": badf, "bsadf": bsadf, "adf": float(adf), "sadf": sadf, "gsadf": gsadf}


def _panel(bsadf: np.ndarray) -> tuple[np.ndarray, float]:
    bsadf_panel = bsadf.mean(axis=1)
    return bsadf_panel, float(bsadf_panel.max())


def _grid_to_radf_result(
    x: np.ndarray, minw: int, columns: list[str] | None, transform
) -> RadfResult:
    """Shared driver for radf_tt/radf_sign/radf_sign_dm: apply `transform`
    to each series, run gls_dfstat_grid(), assemble a RadfResult."""
    nr, nc = x.shape
    n_minw = nr - minw
    adf = np.empty(nc)
    sadf = np.empty(nc)
    gsadf = np.empty(nc)
    badf = np.empty((n_minw, nc))
    bsadf = np.empty((n_minw, nc))

    for j in range(nc):
        res = gls_dfstat_grid(transform(x[:, j]), minw)
        badf[:, j] = res["badf"]
        bsadf[:, j] = res["bsadf"]
        adf[j] = res["adf"]
        sadf[j] = res["sadf"]
        gsadf[j] = res["gsadf"]

    bsadf_panel, gsadf_panel = _panel(bsadf)
    return RadfResult(
        adf=adf, badf=badf, sadf=sadf, bsadf=bsadf, gsadf=gsadf,
        bsadf_panel=bsadf_panel, gsadf_panel=gsadf_panel,
        minw=minw, lag=0, n=nr, series_names=columns,
    )


def _mc_cv_grid(
    n: int, minw: int | None, nrep: int, seed: int | None, transform, method: str
) -> RadfCv:
    """Shared driver for radf_tt_cv/radf_sign_cv/radf_sign_dm_cv: simulate
    the pivotal null distribution of gls_dfstat_grid() applied to
    `transform(cumsum(normal))`."""
    minw = minw if minw is not None else psy_minw(n)
    rng = np.random.default_rng(seed)

    n_minw = n - minw
    adf = np.empty(nrep)
    sadf = np.empty(nrep)
    gsadf = np.empty(nrep)
    badf = np.empty((n_minw, nrep))
    bsadf = np.empty((n_minw, nrep))

    for i in range(nrep):
        y = np.cumsum(rng.normal(size=n))
        res = gls_dfstat_grid(transform(y), minw)
        badf[:, i] = res["badf"]
        bsadf[:, i] = res["bsadf"]
        adf[i] = res["adf"]
        sadf[i] = res["sadf"]
        gsadf[i] = res["gsadf"]

    return RadfCv(
        adf_cv=np.quantile(adf, PCNT),
        sadf_cv=np.quantile(sadf, PCNT),
        gsadf_cv=np.quantile(gsadf, PCNT),
        badf_cv=np.quantile(badf, PCNT, axis=1).T,
        bsadf_cv=np.quantile(bsadf, PCNT, axis=1).T,
        method=method, minw=minw, n=n, iter=nrep,
    )
