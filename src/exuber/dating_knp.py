"""Bias-corrected bubble dating (Kejriwal, Nguyen & Perron 2025). Ported from
exuber's R/dating_knp.R. The methodology and validation record are in
docs/dating-and-root-inference.md in the umbrella repo.

The KNP model has a unit root, then an explosive regime fitted with an
intercept and slope, then a unit root that resumes from a shifted level after
an instantaneous collapse. It is structurally identical to Model 2 of HLS, so
this module reuses _hls_prefix_sums(), _hls_segment_ssr() and
_hls_segment_coef() from exuber._hls_common directly.

Plain OLS over this model is provably inconsistent (Theorem 1 of the
paper): the estimate of the origination date converges to the true collapse
date and not to the origination date. The fix in Theorem 2 omits the single
squared residual at the candidate collapse-date observation from the
objective before minimising. This needs no new regression, only the
subtraction of one squared term that is already available. Unlike the Model
2/3 fit of HLS, the KNP candidate set imposes no sign constraint on the
fitted "peak".

For breaks > 2 the module uses the dynamic programme of Section 3 of the
paper (_knp_dp). Regimes alternate between unit root and explosive, every
unit-root regime after a collapse omits its first residual, and every segment
SSR costs O(1) from the HLS prefix sums. The programme therefore returns the
exact global minimiser in O(m T^2). As in the paper, the number of breaks is
taken as given.

Indexing note: as in dating_pdc(), dating_hls() and dating_hlw(), the
origination and collapse that dating_knp() returns are 1-indexed row
positions into `data`. They do not follow the 0-indexed convention of the
Episode in datestamp(). We keep them 1-indexed on purpose, so that a number
reported here matches the equivalent R run bit for bit.
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


def _knp_dp(
    y: np.ndarray, breaks: int, trim: float = 0.05, omit: bool = True
) -> tuple[list[int] | None, float]:
    """KNP's m-break dynamic programme (their Section 3.2). Pairs are
    i-indexed 1..n1; regime j covers (tau_{j-1}, tau_j]; odd j is a unit-root
    regime (cost sum z^2, minus its first term after a collapse when
    `omit`), even j an explosive one (OLS SSR with intercept). Every
    regime has at least k_min pairs. Returns (tau_1..tau_m, SSR)."""
    ps = _hls_prefix_sums(y)
    n1 = ps.n1
    k_min = max(2, math.ceil(trim * n1))
    nreg = breaks + 1
    if nreg * k_min > n1:
        return None, math.inf

    def cost(j: int, lo, hi):
        if j % 2 == 0:
            return _hls_segment_ssr(ps, lo, hi, True)
        ssr = _hls_segment_ssr(ps, lo, hi, False)
        if omit and j > 1:
            ssr = ssr - (ps.cz2[np.asarray(lo) + 1] - ps.cz2[lo])
        return ssr

    v = np.full((nreg + 1, n1 + 1), np.inf)
    arg = np.zeros((nreg + 1, n1 + 1), dtype=int)
    hi1 = np.arange(k_min, n1 + 1)
    v[1, hi1] = cost(1, 0, hi1)
    for j in range(2, nreg + 1):
        his = [n1] if j == nreg else range(j * k_min, n1 - (nreg - j) * k_min + 1)
        for hi in his:
            lo = np.arange((j - 1) * k_min, hi - k_min + 1)
            cand = v[j - 1, lo] + cost(j, lo, hi)
            k = int(np.argmin(cand))
            v[j, hi] = cand[k]
            arg[j, hi] = lo[k]
    tau = [0] * breaks
    hi = n1
    for j in range(nreg, 1, -1):
        hi = int(arg[j, hi])
        tau[j - 2] = hi
    return tau, float(v[nreg, n1])


@dataclass
class DatingKnpResult:
    # R-bit-for-bit 1-indexed positions, NaN if not found: (nc,) for one
    # bubble, (n_bubbles, nc) for more (R returns vectors vs. matrices alike)
    origination: np.ndarray
    collapse: np.ndarray  # NaN for a bubble still running at the sample end
    delta: np.ndarray  # fitted explosive AR coefficient (1 + slope)
    series_names: list[str] | None
    trim: float
    omit: bool
    n: int
    breaks: int = 2


def dating_knp(data, trim: float = 0.05, omit: bool = True, breaks: int = 2) -> DatingKnpResult:
    """Bias-corrected bubble dating (Kejriwal, Nguyen & Perron 2025).
    Dates a single bubble episode (origination, collapse) by minimising a
    residual-omission-corrected sum of squared residuals over a
    three-regime model (unit root, explosive, unit root resuming from a
    shifted level after an instantaneous collapse). Plain OLS over this
    model is provably inconsistent: the origination-date estimate
    converges to the true collapse date and not the origination date.
    The default omit=True fixes this by dropping the single squared
    residual at the candidate collapse date from the objective before
    minimising.

    omit=False gives the plain, provably inconsistent OLS estimator
    (Theorem 1). We keep it mainly to show the effect of the correction and
    not for practical dating. It needs no critical values, because it
    selects a model by residual sum of squares and does not test a
    hypothesis.

    breaks is the number of break dates (the paper's m): 2 per bubble, an
    odd number letting the last bubble run to the sample end. breaks > 2
    uses KNP's dynamic programme (exact global minimiser, O(breaks * n^2)).
    """
    if breaks < 1:
        raise ValueError("breaks must be a positive integer")
    x, columns = _to_2d_array(data)
    n, nc = x.shape
    names = columns or [f"series{i + 1}" for i in range(nc)]

    nb = math.ceil(breaks / 2)
    origination = np.full((nb, nc), np.nan)
    collapse = np.full((nb, nc), np.nan)
    delta = np.full((nb, nc), np.nan)

    for j in range(nc):
        y = x[:, j]
        ps = _hls_prefix_sums(y)
        if breaks == 2:
            tau1, tau2, _ssr = _knp_find_break(y, trim, omit)
            tau = None if tau1 is None or tau2 is None else [tau1, tau2]
        else:
            tau, _ssr = _knp_dp(y, breaks, trim, omit)
        if tau is None:
            continue
        ends = [*tau, ps.n1]
        for b in range(nb):
            t1, t2 = ends[2 * b], ends[2 * b + 1]
            origination[b, j] = t1 + 1
            if 2 * b + 2 <= breaks:
                collapse[b, j] = t2 + 1
            _a, slope = _hls_segment_coef(ps, t1, t2)
            delta[b, j] = slope + 1

    if nb == 1:
        origination, collapse, delta = origination[0], collapse[0], delta[0]
    return DatingKnpResult(
        origination=origination, collapse=collapse, delta=delta,
        series_names=names, trim=trim, omit=omit, n=n, breaks=breaks,
    )
