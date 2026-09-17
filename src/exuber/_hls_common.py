"""Shared SSR/BIC segment-fitting machinery (Harvey, Leybourne & Sollis
2017), used by dating_hls(), dating_hlw() and dating_knp() -- all three
build on the same closed-form prefix-sum segment fits, so the helpers
live here rather than in any one of the three.

Indexing: breakpoints are internally 0-indexed "boundary counts" b in
0..n1 (n1 = len(y)-1) -- y[b] is the observation at that boundary (same
convention dating_pdc()'s _pdc_find_break() uses).
"""

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class _HlsPrefixSums:
    cx: np.ndarray
    cx2: np.ndarray
    cz: np.ndarray
    cz2: np.ndarray
    cxz: np.ndarray
    n1: int


def _hls_prefix_sums(y: np.ndarray) -> _HlsPrefixSums:
    n1 = len(y) - 1
    x = y[:n1]
    z = np.diff(y)
    return _HlsPrefixSums(
        cx=np.concatenate(([0.0], np.cumsum(x))),
        cx2=np.concatenate(([0.0], np.cumsum(x**2))),
        cz=np.concatenate(([0.0], np.cumsum(z))),
        cz2=np.concatenate(([0.0], np.cumsum(z**2))),
        cxz=np.concatenate(([0.0], np.cumsum(x * z))),
        n1=n1,
    )


def _hls_segment_ssr(ps: _HlsPrefixSums, lo, hi, fit: bool):
    """SSR of the segment(s) with i-index in (lo, hi] (lo/hi may be
    scalars or equal-length/broadcastable arrays). fit=False: no active
    dummy, SSR = sum(z^2). fit=True: intercept + slope OLS of z on x."""
    sx = ps.cx[hi] - ps.cx[lo]
    sxx = ps.cx2[hi] - ps.cx2[lo]
    sz = ps.cz[hi] - ps.cz[lo]
    szz = ps.cz2[hi] - ps.cz2[lo]
    if not fit:
        return szz
    sxz = ps.cxz[hi] - ps.cxz[lo]
    n_seg = np.asarray(hi) - np.asarray(lo)
    b = (n_seg * sxz - sx * sz) / (n_seg * sxx - sx**2)
    a = (sz - b * sx) / n_seg
    return szz - a * sz - b * sxz


def _hls_segment_coef(ps: _HlsPrefixSums, lo, hi) -> tuple[float, float]:
    """Intercept + slope OLS coefficients of z on x over segment (lo, hi]
    (same closed form as _hls_segment_ssr(..., fit=True), returning the
    coefficients themselves rather than the SSR -- used by dating_knp())."""
    sx = ps.cx[hi] - ps.cx[lo]
    sxx = ps.cx2[hi] - ps.cx2[lo]
    sz = ps.cz[hi] - ps.cz[lo]
    sxz = ps.cxz[hi] - ps.cxz[lo]
    n_seg = np.asarray(hi) - np.asarray(lo)
    b = (n_seg * sxz - sx * sz) / (n_seg * sxx - sx**2)
    a = (sz - b * sx) / n_seg
    return float(a), float(b)


def _hls_model1(y: np.ndarray, ps: _HlsPrefixSums, trim: float) -> tuple[int | None, float]:
    """Model 1: unit root -> bubble to sample end. Sign constraint:
    y_T > y_tau1 (series ends above where the bubble started)."""
    n1 = ps.n1
    k_min = max(2, math.ceil(trim * n1))
    taus = np.arange(k_min, n1 - k_min + 1)
    taus = taus[y[n1] > y[taus]]
    if len(taus) == 0:
        return None, math.inf
    ssr = _hls_segment_ssr(ps, 0, taus, False) + _hls_segment_ssr(ps, taus, n1, True)
    best = int(np.argmin(ssr))
    return int(taus[best]), float(ssr[best])


def _hls_model23(
    y: np.ndarray, ps: _HlsPrefixSums, trim: float, right_fit: bool
) -> tuple[int | None, int | None, float]:
    """Model 2 (right_fit=False, post-bubble tail unfitted) / Model 3
    (right_fit=True, post-bubble tail fitted as its own collapse regime).
    Sign constraint: y_tau2 > y_tau1 always; right_fit also requires
    y_tau2 > y_T (the peak must exceed the fitted collapse's endpoint)."""
    n1 = ps.n1
    k_min = max(2, math.ceil(trim * n1))
    best_ssr, best_tau1, best_tau2 = math.inf, None, None
    tau1_max = n1 - 2 * k_min
    if tau1_max < k_min:
        return None, None, math.inf
    for tau1 in range(k_min, tau1_max + 1):
        tau2 = np.arange(tau1 + k_min, n1 - k_min + 1)
        valid = y[tau2] > y[tau1]
        if right_fit:
            valid = valid & (y[tau2] > y[n1])
        tau2 = tau2[valid]
        if len(tau2) == 0:
            continue
        ssr = (
            _hls_segment_ssr(ps, 0, tau1, False)
            + _hls_segment_ssr(ps, tau1, tau2, True)
            + _hls_segment_ssr(ps, tau2, n1, right_fit)
        )
        j = int(np.argmin(ssr))
        if ssr[j] < best_ssr:
            best_ssr, best_tau1, best_tau2 = float(ssr[j]), tau1, int(tau2[j])
    return best_tau1, best_tau2, best_ssr


def _hls_model4(
    y: np.ndarray, ps: _HlsPrefixSums, trim: float
) -> tuple[int | None, int | None, int | None, float]:
    """Model 4: unit root -> bubble -> collapse -> unit root recovery.
    Sign constraints: y_tau2 > y_tau1 and y_tau2 > y_tau3 (the peak must
    exceed both the bubble's start and the fitted collapse's endpoint)."""
    n1 = ps.n1
    k_min = max(2, math.ceil(trim * n1))
    best_ssr: float = math.inf
    best_tau1: int | None = None
    best_tau2: int | None = None
    best_tau3: int | None = None
    tau1_max = n1 - 3 * k_min
    if tau1_max < k_min:
        return None, None, None, math.inf
    for tau1 in range(k_min, tau1_max + 1):
        tau2_max = n1 - 2 * k_min
        for tau2 in range(tau1 + k_min, tau2_max + 1):
            if y[tau2] <= y[tau1]:
                continue
            tau3 = np.arange(tau2 + k_min, n1 - k_min + 1)
            tau3 = tau3[y[tau2] > y[tau3]]
            if len(tau3) == 0:
                continue
            ssr = (
                _hls_segment_ssr(ps, 0, tau1, False)
                + _hls_segment_ssr(ps, tau1, tau2, True)
                + _hls_segment_ssr(ps, tau2, tau3, True)
                + _hls_segment_ssr(ps, tau3, n1, False)
            )
            j = int(np.argmin(ssr))
            if ssr[j] < best_ssr:
                best_ssr = float(ssr[j])
                best_tau1, best_tau2, best_tau3 = tau1, tau2, int(tau3[j])
    return best_tau1, best_tau2, best_tau3, best_ssr


def _hls_bic(ssr: float, n: int, df: int) -> float:
    return n * math.log(ssr / n) + df * math.log(n)


_HLS_DFS = (3, 4, 6, 7)  # models 1-4


@dataclass
class HlsFit:
    model: int  # 1-4
    breaks: dict[str, int | None]  # "tau1"/"tau2"/"tau3" -> 0-indexed boundary count
    bic: np.ndarray  # length 4, inf for unrequested models


def _hls_fit_series(y: np.ndarray, trim: float, models: tuple[int, ...] = (1, 2, 3, 4)) -> HlsFit:
    """Fits the requested subset of HLS's four models to a single series
    `y` and BIC-selects among them. Shared by dating_hls() and (per-window,
    restricted to models (2, 4)) dating_hlw()."""
    n = len(y)
    ps = _hls_prefix_sums(y)
    bic = np.full(4, math.inf)
    fits: dict[int, tuple] = {}

    if 1 in models:
        tau1, ssr = _hls_model1(y, ps, trim)
        fits[1] = (tau1,)
        bic[0] = _hls_bic(ssr, n, _HLS_DFS[0])
    if 2 in models:
        tau1, tau2, ssr = _hls_model23(y, ps, trim, right_fit=False)
        fits[2] = (tau1, tau2)
        bic[1] = _hls_bic(ssr, n, _HLS_DFS[1])
    if 3 in models:
        tau1, tau2, ssr = _hls_model23(y, ps, trim, right_fit=True)
        fits[3] = (tau1, tau2)
        bic[2] = _hls_bic(ssr, n, _HLS_DFS[2])
    if 4 in models:
        tau1, tau2, tau3, ssr = _hls_model4(y, ps, trim)
        fits[4] = (tau1, tau2, tau3)
        bic[3] = _hls_bic(ssr, n, _HLS_DFS[3])

    jopt = int(np.argmin(bic)) + 1
    fit = fits[jopt]
    if jopt == 1:
        breaks = {"tau1": fit[0]}
    elif jopt in (2, 3):
        breaks = {"tau1": fit[0], "tau2": fit[1]}
    else:
        breaks = {"tau1": fit[0], "tau2": fit[1], "tau3": fit[2]}
    return HlsFit(model=jopt, breaks=breaks, bic=bic)
