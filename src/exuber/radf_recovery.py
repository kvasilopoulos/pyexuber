"""Reverse-regression crisis-origination/market-recovery dating
(Phillips & Shi 2014). Ported from exuber's R/radf_recovery.R (see
docs/dating-and-root-inference.md in the umbrella repo for the full
methodology and validation record). Shipped with the same caveats as R:
f_r (recovery) validates well, f_c (crisis origination) and the false-
detection rate under H0 are noisier and not fully resolved -- see
radf_recovery()'s docstring and docs/dating-and-root-inference.md's
"Reverse-regression recovery dating" section.

Indexing note: unlike dating_pdc()'s/dating_hls()'s/dating_hlw()'s/
dating_knp()'s 1-indexed outputs, radf_recovery()'s f_c/f_r ARE 0-indexed
positions (the same convention as datestamp()'s Episode), since they're
derived directly from radf()'s own bsadf array the same way datestamp()
is.
"""

import warnings
from dataclasses import dataclass

import numpy as np

from exuber._unroot import unroot
from exuber.cv import PCNT
from exuber.datestamp import SIG_IDX
from exuber.radf import _to_2d_array, psy_minw, radf

# -- Reverse-regression recovery dating (Phillips & Shi 2014) ---------------

_RECOVERY_CAVEAT = (
    "Experimental. f_c and the overall false-detection rate are exploratory "
    "pending further validation; see radf_recovery()'s docstring."
)


@dataclass
class RadfRecoveryCv:
    bsadf_cv: np.ndarray  # (n_minw, 3), columns 90%/95%/99%
    minw: int
    n: int
    iter: int
    lag: int = 0


def _radf_recovery_mc(
    n: int, minw: int | None = None, nrep: int = 1000, seed: int | None = None, lag: int = 0
) -> tuple[np.ndarray, int]:
    from . import _core  # lazy: see radf.py's radf() for why

    minw = minw if minw is not None else psy_minw(n)
    rng = np.random.default_rng(seed)
    n_minw = n - minw - lag

    badf = np.empty((n_minw, nrep))
    for i in range(nrep):
        y = np.cumsum(rng.normal(size=n))[::-1]  # reversed null path -- see module docstring
        yxmat = unroot(y, lag=lag)
        result = _core.radf_stat(yxmat, minw, lag)
        badf[:, i] = result[:n_minw]
    return badf, minw


def radf_recovery_cv(
    n: int, minw: int | None = None, nrep: int = 1000, seed: int | None = None, lag: int = 0
) -> RadfRecoveryCv:
    """Monte Carlo critical values for radf_recovery()'s reverse-time bsadf
    statistic, calibrated to Phillips & Shi (2014)'s own (different from
    the forward test's) null limiting distribution: reversal induces an
    endogeneity between the reverse-time regressor and its own innovation
    with no forward-regression analogue (see this module's docstring), so
    reusing radf_mc_cv()'s forward critical values would be a mismatched
    boundary, not an approximation of known quality.
    """
    badf, minw = _radf_recovery_mc(n, minw, nrep, seed, lag)
    bsadf_cv = np.quantile(np.maximum.accumulate(badf, axis=0), PCNT, axis=1).T
    return RadfRecoveryCv(bsadf_cv=bsadf_cv, minw=minw, n=n, iter=nrep, lag=lag)


@dataclass
class RadfRecoveryResult:
    f_c: np.ndarray  # crisis-origination date, NaN if not identified
    f_r: np.ndarray  # market-recovery date, NaN if not identified
    detected: np.ndarray
    censored: np.ndarray
    series_names: list[str] | None
    sig_lvl: int
    minw: int
    lag: int
    n: int


def _recovery_dates_from_bsadf(
    bsadf_rev: np.ndarray,
    bsadf_cv: np.ndarray,
    minw: int,
    lag: int,
    n: int,
    sig_lvl: int = 95,
    series_names: list[str] | None = None,
) -> RadfRecoveryResult:
    """Phillips & Shi (2014) eq. 8-9 crossing rule, operating on an
    already-computed reverse-time bsadf array (kept separate from
    radf_recovery() so the crossing logic is testable without radf()/the
    C++ extension -- same split as datestamp() operating on a
    pre-computed RadfResult/RadfCv).

    Locates the first up-crossing of the reversal-calibrated boundary
    (market recovery, f_r) then the next down-crossing, searched only
    after the up-crossing (crisis origination in the original series,
    f_c) -- so f_c <= f_r always, by construction, whenever both are
    identified and uncensored. Positions are 0-indexed into the original
    (non-reversed) series, matching datestamp()'s Episode convention.
    """
    if sig_lvl not in SIG_IDX:
        raise ValueError("sig_lvl must be one of 90, 95, 99")
    sidx = SIG_IDX[sig_lvl]
    zadj = minw + lag
    n_minw, nc = bsadf_rev.shape
    names = series_names or [f"series{i + 1}" for i in range(nc)]

    f_c = np.full(nc, np.nan)
    f_r = np.full(nc, np.nan)
    detected = np.zeros(nc, dtype=bool)
    censored = np.zeros(nc, dtype=bool)

    for j in range(nc):
        exceed = bsadf_rev[:, j] > bsadf_cv[:, sidx]
        exceed_idx = np.where(exceed)[0]
        if len(exceed_idx) == 0:
            continue
        detected[j] = True
        g_e = exceed_idx[0]
        rev_pos_e = g_e + zadj
        f_r[j] = n - 1 - rev_pos_e

        not_exceed_after = np.where(~exceed[g_e:])[0]
        if len(not_exceed_after) == 0:
            censored[j] = True
            continue
        g_c = g_e + not_exceed_after[0]
        rev_pos_c = min(g_c + zadj, n - 1)
        f_c[j] = n - 1 - rev_pos_c

    return RadfRecoveryResult(
        f_c=f_c, f_r=f_r, detected=detected, censored=censored,
        series_names=names, sig_lvl=sig_lvl, minw=minw, lag=lag, n=n,
    )


def radf_recovery(
    data,
    minw: int | None = None,
    lag: int = 0,
    nrep: int = 1000,
    sig_lvl: int = 95,
    seed: int | None = None,
) -> RadfRecoveryResult:
    """Reverse-regression dating of crisis origination and market recovery
    (Phillips & Shi 2014). Reverses the series, runs radf()'s existing
    bsadf recursion on it, and locates the first up-crossing of a
    reversal-calibrated critical value boundary (market recovery, f_r)
    followed by the next down-crossing (crisis/collapse origination in
    the original series, f_c), then maps both back to the original time
    index. f_c <= f_r always, by construction, when both are identified.

    Caveats (validation status, see docs/dating-and-root-inference.md's
    "Reverse-regression recovery dating" for the full numbers): f_r
    validates well against synthetic collapse-then-recovery data. f_c
    shows a materially larger residual bias, and the empirical
    false-detection rate under a pure random-walk null is around 29% at
    n=100/minw=20/95% -- higher than comparable forward-test numbers
    elsewhere in this package. Treat f_c and the overall detection rate
    as exploratory pending further validation.
    """
    if sig_lvl not in SIG_IDX:
        raise ValueError("sig_lvl must be one of 90, 95, 99")
    warnings.warn(_RECOVERY_CAVEAT, stacklevel=2)

    x, columns = _to_2d_array(data)
    n = x.shape[0]
    minw = minw if minw is not None else psy_minw(n)

    x_rev = x[::-1, :]
    rev_fit = radf(x_rev, minw=minw, lag=lag)
    cv = radf_recovery_cv(n=n, minw=minw, nrep=nrep, seed=seed, lag=lag)

    return _recovery_dates_from_bsadf(
        rev_fit.bsadf, cv.bsadf_cv, minw=minw, lag=lag, n=n, sig_lvl=sig_lvl, series_names=columns
    )
