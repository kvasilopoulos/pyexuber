"""SSR/BIC bubble dating (Harvey, Leybourne & Sollis 2017). Ported from
exuber's R/dating_hls.R. The methodology and validation record are in
docs/dating-and-root-inference.md in the umbrella repo.

This method replaces the threshold-crossing rule of PSY with a model-based
rule that minimises the sum of squared residuals (SSR) and selects the model
by BIC. There are four candidate regime-dummy regressions of Delta y_t on
y_{t-1}: unit root to the end, unit root then bubble then unit root, unit
root then bubble then collapse, and unit root then bubble then collapse then
unit root. Each is fitted by minimising the SSR over candidate break
fractions, jointly within the model, and BIC then selects among the four.

The dummy windows of the four models never overlap. The SSR of a candidate
partition is therefore exactly the sum of independent closed-form OLS fits
for each segment. A segment without a dummy has SSR = sum(Delta y_t^2), and
a segment with a dummy is a plain intercept-and-slope fit. Each fit is a
closed-form ratio of cumulative sums, and it can be looked up in O(1) per
candidate, as _pdc_find_break() in dating_pdc() does. The joint grid search
needs no repeated regression fits, only differences of prefix sums, and it
matches exuber's own R/dating_hls.R exactly. The shared prefix-sum and
model-fit code lives in exuber._hls_common, which dating_hlw() and
dating_knp() also use.

Indexing: breakpoints are held internally as 0-indexed "boundary counts" b in
0..n1 (n1 = len(y)-1), and y[b] is the observation at that boundary. This is
the convention of _pdc_find_break(). The origination, collapse and recovery
that dating_hls() returns add 1, so that they match the numbers in the output
of R's own dating_hls() (R uses idx[b + 1L], the 1-indexed position b+1). This
follows the "R bit for bit" convention that dating_pdc() already uses, and
not the 0-indexed Episode convention of datestamp().
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
    BIC pick the structure itself. It needs no critical values, because it
    selects a model and does not test a hypothesis.
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
