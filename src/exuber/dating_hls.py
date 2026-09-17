"""SSR/BIC bubble dating (Harvey, Leybourne & Sollis 2017). Ported from
exuber's R/dating_hls.R (see docs/dating-and-root-inference.md in the
umbrella repo for the full methodology and validation record).

Replaces PSY's threshold-crossing rule with a model-based SSR-minimisation
+ BIC rule: four candidate regime-dummy regressions of Delta y_t on
y_{t-1} (unit-root-to-end / unit-root-bubble-unit-root / unit-root-
bubble-collapse / unit-root-bubble-collapse-unit-root), each fit by
residual-sum-of-squares minimisation over candidate break fractions
(jointly, per model), with BIC selecting among the four. Because the
four models' dummy windows never overlap, each candidate partition's
SSR is exactly the sum of independent per-segment closed-form OLS fits
(a no-dummy segment has SSR = sum(Delta y_t^2); a dummy segment is a
plain intercept+slope fit) -- a closed-form ratio of cumulative sums,
the same style of O(1)-per-candidate lookup dating_pdc()'s
_pdc_find_break() uses, so the joint grid search needs no repeated
regression fit, only prefix-sum differences (matches exuber's own
R/dating_hls.R exactly). The shared prefix-sum/model-fit machinery lives
in exuber._hls_common (also used by dating_hlw() and dating_knp()).

Indexing: breakpoints are internally 0-indexed "boundary counts" b in
0..n1 (n1 = len(y)-1) -- y[b] is the observation at that boundary (same
convention _pdc_find_break() uses). dating_hls()'s *output*
origination/collapse/recovery add 1 to match R's own dating_hls()
output number (R: idx[b + 1L], i.e. R's 1-indexed position b+1) --
matching dating_pdc()'s existing "R-bit-for-bit" convention, not
datestamp()'s 0-indexed Episode convention.
"""

from dataclasses import dataclass

import numpy as np

from exuber._hls_common import _hls_fit_series
from exuber.radf import _to_2d_array


@dataclass
class DatingHlsResult:
    model: np.ndarray  # int per series, 1-4
    origination: np.ndarray  # NaN if not applicable
    collapse: np.ndarray
    recovery: np.ndarray
    bic: np.ndarray  # (nc, 4)
    series_names: list[str] | None
    trim: float
    n: int


def dating_hls(data, trim: float = 0.05) -> DatingHlsResult:
    """SSR/BIC bubble dating (Harvey, Leybourne & Sollis 2017). Fits four
    candidate regime-dummy regressions of Delta y_t on y_{t-1} (unit-
    root-to-end / unit-root-bubble-unit-root / unit-root-bubble-collapse
    / unit-root-bubble-collapse-unit-root), each by joint residual-sum-
    -of-squares minimisation over its candidate break fraction(s), and
    selects among them by BIC.

    Unlike datestamp() (threshold-crossing on the recursive BSADF
    statistic) or dating_pdc() (a fixed 3/4-regime structure with
    sequentially, not jointly, estimated breaks), this jointly searches
    breakpoints within each of four candidate regime structures and lets
    BIC pick the structure itself. Needs no critical values -- this is
    model selection, not a hypothesis test.
    """
    x, columns = _to_2d_array(data)
    n, nc = x.shape
    names = columns or [f"series{i + 1}" for i in range(nc)]

    model = np.empty(nc, dtype=int)
    origination = np.full(nc, np.nan)
    collapse = np.full(nc, np.nan)
    recovery = np.full(nc, np.nan)
    bic_mat = np.full((nc, 4), np.nan)

    for j in range(nc):
        fit = _hls_fit_series(x[:, j], trim, models=(1, 2, 3, 4))
        model[j] = fit.model
        bic_mat[j, :] = fit.bic
        b = fit.breaks
        tau1, tau2, tau3 = b.get("tau1"), b.get("tau2"), b.get("tau3")
        if tau1 is not None:
            origination[j] = tau1 + 1
        if tau2 is not None:
            collapse[j] = tau2 + 1
        if tau3 is not None:
            recovery[j] = tau3 + 1

    return DatingHlsResult(
        model=model, origination=origination, collapse=collapse, recovery=recovery,
        bic=bic_mat, series_names=names, trim=trim, n=n,
    )
