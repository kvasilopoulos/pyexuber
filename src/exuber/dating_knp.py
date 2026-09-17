"""Bias-corrected single-bubble dating (Kejriwal, Nguyen & Perron 2025).
Ported from exuber's R/dating_knp.R (see docs/dating-and-root-
inference.md in the umbrella repo for the full methodology and
validation record).

KNP's own model (unit root -> intercept+slope-fitted explosive regime
-> unit root resuming from a shifted level after an instantaneous
collapse) is structurally identical to HLS's own Model 2, so this
reuses exuber._hls_common's _hls_prefix_sums()/_hls_segment_ssr()/
_hls_segment_coef() directly. Plain OLS over this model is provably
inconsistent (their Theorem 1): the origination-date estimate converges
to the true COLLAPSE date, not the origination date. Their Theorem 2 fix
omits the single squared residual at the candidate collapse-date
observation from the objective before minimising -- no new regression,
just subtracting one already-available squared term. Unlike HLS's Model
2/3 fit, KNP's candidate set imposes no sign constraint on the fitted
"peak". The multi-bubble Bai-Perron/Perron-Qu dynamic-programming
extension (KNP's Section 3) is not implemented -- same scoping decision
as dating_hlw()'s own unimplemented run-joining heuristic (genuinely new
algorithmic machinery, not a small addition to already-shipped code).

Indexing note: like dating_pdc()/dating_hls()/dating_hlw(), dating_knp()'s
origination/collapse are 1-indexed row positions into `data` -- NOT the
0-indexed convention datestamp()'s Episode uses (kept 1-indexed
deliberately so a number reported here matches the equivalent R run
bit-for-bit).
"""

import math
from dataclasses import dataclass

import numpy as np

from exuber._hls_common import _hls_prefix_sums, _hls_segment_coef, _hls_segment_ssr
from exuber.radf import _to_2d_array

# -- Bias-corrected single-bubble dating (Kejriwal, Nguyen & Perron 2025) ---


def _knp_find_break(
    y: np.ndarray, trim: float = 0.05, omit: bool = True
) -> tuple[int | None, int | None, float]:
    n1 = len(y) - 1
    ps = _hls_prefix_sums(y)
    k_min = max(2, math.ceil(trim * n1))
    tau1_max = n1 - 2 * k_min
    if tau1_max < k_min:
        return None, None, math.inf

    best_ssr, best_tau1, best_tau2 = math.inf, None, None
    for tau1 in range(k_min, tau1_max + 1):
        tau2 = np.arange(tau1 + k_min, n1 - k_min + 1)
        ssr = (
            _hls_segment_ssr(ps, 0, tau1, False)
            + _hls_segment_ssr(ps, tau1, tau2, True)
            + _hls_segment_ssr(ps, tau2, n1, False)
        )
        if omit:
            z2_single = ps.cz2[tau2 + 1] - ps.cz2[tau2]  # (Delta y_{tau2+1})^2
            ssr = ssr - z2_single
        j = int(np.argmin(ssr))
        if ssr[j] < best_ssr:
            best_ssr, best_tau1, best_tau2 = float(ssr[j]), tau1, int(tau2[j])
    return best_tau1, best_tau2, best_ssr


@dataclass
class DatingKnpResult:
    origination: np.ndarray  # R-bit-for-bit 1-indexed position, NaN if not found
    collapse: np.ndarray
    delta: np.ndarray  # fitted explosive AR coefficient (1 + slope)
    series_names: list[str] | None
    trim: float
    omit: bool
    n: int


def dating_knp(data, trim: float = 0.05, omit: bool = True) -> DatingKnpResult:
    """Bias-corrected single-bubble dating (Kejriwal, Nguyen & Perron 2025).
    Dates a single bubble episode (origination, collapse) by minimising a
    residual-omission-corrected sum of squared residuals over a
    three-regime model (unit root, explosive, unit root resuming from a
    shifted level after an instantaneous collapse). Plain OLS over this
    model is provably inconsistent -- the origination-date estimate
    converges to the true collapse date, not the origination date --
    which omit=True (the default) fixes by dropping the single squared
    residual at the candidate collapse date from the objective before
    minimising.

    omit=False gives the plain, provably inconsistent OLS estimator
    (Theorem 1) -- kept mainly to demonstrate the correction's effect,
    not for practical dating. Needs no critical values -- this is
    residual-sum-of-squares model selection, not a hypothesis test.
    """
    x, columns = _to_2d_array(data)
    n, nc = x.shape
    names = columns or [f"series{i + 1}" for i in range(nc)]

    origination = np.full(nc, np.nan)
    collapse = np.full(nc, np.nan)
    delta = np.full(nc, np.nan)

    for j in range(nc):
        y = x[:, j]
        ps = _hls_prefix_sums(y)
        tau1, tau2, _ssr = _knp_find_break(y, trim, omit)
        if tau1 is not None:
            origination[j] = tau1 + 1
        if tau2 is not None:
            collapse[j] = tau2 + 1
        if tau1 is not None and tau2 is not None:
            _a, b = _hls_segment_coef(ps, tau1, tau2)
            delta[j] = b + 1

    return DatingKnpResult(
        origination=origination, collapse=collapse, delta=delta,
        series_names=names, trim=trim, omit=omit, n=n,
    )
