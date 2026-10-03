"""Per-series test verdicts. Ports diagnostics.radf_obj() and
summary.radf_obj() from exuber's R/radf-methods.R.

Divergence from R: R labels a non-rejection with the string "Reject" in
`sig`, and tidy.dg_radf() maps it to NA. Here it is None. R's summary()
returns a list of tibbles, and here it returns a dict of pandas DataFrames
with the same columns (stat, tstat, 90, 95, 99). pandas is an optional extra,
as for tidy().
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from exuber.cv import RadfCv, RadfSbCv
from exuber.datestamp import SIG_IDX, _cv_overall
from exuber.radf import RadfResult
from exuber.tidy import _require_pandas, _series_names

if TYPE_CHECKING:
    import pandas as pd

_SIG_LABELS = ("10%", "5%", "1%")


@dataclass
class Diagnostics:
    positive: list[str]  # series that reject H0 at sig_lvl
    negative: list[str]
    # strongest level rejected per series: "10%"/"5%"/"1%"; None = no rejection
    sig: list[str | None]
    dummy: np.ndarray  # 1 if the series rejects at sig_lvl, else 0
    option: str
    sig_lvl: int
    method: str
    panel: bool


def _check(result: RadfResult, cv: RadfCv | RadfSbCv) -> None:
    if not isinstance(cv, (RadfCv, RadfSbCv)):
        raise TypeError("cv must be a RadfCv or RadfSbCv")
    if cv.minw != result.minw or cv.n != result.n:
        raise ValueError("cv and result disagree on n/minw; simulate cv for this sample")


def _level(tstat: float, crit: np.ndarray) -> str | None:
    """Strongest significance level at which tstat >= crit (crit ordered 90/95/99)."""
    passed = int(np.sum(tstat >= crit))
    return _SIG_LABELS[passed - 1] if passed else None


def diagnostics(
    result: RadfResult, cv: RadfCv | RadfSbCv, option: str = "gsadf", sig_lvl: int = 95
) -> Diagnostics:
    """Port of R's diagnostics.radf_obj(): which series reject the null of
    no explosive behaviour, using the overall gsadf (or sadf) statistic.
    With a sieve-bootstrap cv the verdict is for the panel (gsadf_panel)."""
    if option not in ("gsadf", "sadf"):
        raise ValueError('option must be "gsadf" or "sadf"')
    if sig_lvl not in SIG_IDX:
        raise ValueError("sig_lvl must be one of 90, 95, 99")
    _check(result, cv)

    if isinstance(cv, RadfSbCv):
        if option == "sadf":
            raise ValueError("option cannot be 'sadf' when cv is a sieve-bootstrap RadfSbCv")
        names = ["panel"]
        tstats = np.array([result.gsadf_panel])
        crits = [np.asarray(cv.gsadf_panel_cv)]
        option = "gsadf_panel"
    else:
        names = _series_names(result)
        tstats = np.asarray(getattr(result, option))
        cv_arr = np.asarray(getattr(cv, f"{option}_cv"))
        crits = [_cv_overall(cv_arr, j) for j in range(len(names))]

    dummy = np.array([int(t >= c[SIG_IDX[sig_lvl]]) for t, c in zip(tstats, crits, strict=True)])
    return Diagnostics(
        positive=[n for n, d in zip(names, dummy, strict=True) if d],
        negative=[n for n, d in zip(names, dummy, strict=True) if not d],
        sig=[_level(t, c) for t, c in zip(tstats, crits, strict=True)],
        dummy=dummy,
        option=option,
        sig_lvl=sig_lvl,
        method=cv.method,
        panel=isinstance(cv, RadfSbCv),
    )


def summary(result: RadfResult, cv: RadfCv | RadfSbCv) -> "dict[str, pd.DataFrame]":
    """Port of R's summary.radf_obj(): per series, the adf/sadf/gsadf
    statistics next to their 90/95/99% critical values. With a
    sieve-bootstrap cv, a single "panel" entry for gsadf_panel."""
    _check(result, cv)
    pd = _require_pandas()
    cols = ["stat", "tstat", "90", "95", "99"]

    if isinstance(cv, RadfSbCv):
        row = ["gsadf_panel", result.gsadf_panel, *np.asarray(cv.gsadf_panel_cv)]
        return {"panel": pd.DataFrame([row], columns=cols)}

    out = {}
    for j, name in enumerate(_series_names(result)):
        rows = [
            [stat, float(getattr(result, stat)[j]), *_cv_overall(getattr(cv, f"{stat}_cv"), j)]
            for stat in ("adf", "sadf", "gsadf")
        ]
        out[name] = pd.DataFrame(rows, columns=cols)
    return out
