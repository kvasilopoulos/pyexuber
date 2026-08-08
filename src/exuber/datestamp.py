"""Date-stamping of explosive episodes. Port of exuber's R/radf-methods.R
datestamp.radf_obj(), with one deliberate simplification: R selects which
series to date-stamp via diagnostics_internal()/augment_join() (a tibble
pipeline built around R's dplyr internals with no direct Python analogue).
Here a series is date-stamped if its overall statistic (gsadf or sadf,
matching `option`) exceeds the corresponding overall critical value at
`sig_lvl` -- the same substantive test, expressed directly instead of
through that pipeline. `nonrejected` and the peak "Signal" (positive/
negative) field from the R version are not ported (deferred, not silently
dropped -- the raw price/level series isn't retained on RadfResult yet).
"""

from dataclasses import dataclass

import numpy as np

from exuber.cv import RadfCv
from exuber.radf import RadfResult

SIG_IDX = {90: 0, 95: 1, 99: 2}


@dataclass
class Episode:
    start: int
    peak: int
    end: int | None  # None means the episode is still ongoing at the end of the sample
    duration: int
    ongoing: bool


def _stamp(indices: np.ndarray) -> list[tuple[int, int]]:
    """Group positions where the exuberance condition holds into contiguous
    (start, end) runs, end exclusive. Port of R's stamp(); verified against
    R's actual output for the 1-indexed -> 0-indexed translation."""
    if len(indices) == 0:
        return []
    is_start = np.concatenate(([True], np.diff(indices) != 1))
    is_end = np.concatenate((np.diff(indices) != 1, [True]))
    starts = indices[is_start]
    ends = indices[is_end] + 1
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def _cv_curve(cv_arr: np.ndarray, j: int) -> np.ndarray:
    """cv_arr is (T, 3) if shared across series (Monte Carlo) or (T, 3, nc)
    if per-series (wild/sieve bootstrap)."""
    return cv_arr if cv_arr.ndim == 2 else cv_arr[:, :, j]


def _cv_overall(cv_arr: np.ndarray, j: int) -> np.ndarray:
    """cv_arr is (3,) if shared across series or (nc, 3) if per-series."""
    return cv_arr if cv_arr.ndim == 1 else cv_arr[j]


def datestamp(
    result: RadfResult, cv: RadfCv, min_duration: int = 0, sig_lvl: int = 95, option: str = "gsadf"
) -> dict[str, list[Episode]]:
    """Date-stamp periods of explosive behaviour.

    For each series whose overall statistic rejects the null at `sig_lvl`,
    finds contiguous runs where the BSADF (option="gsadf") or BADF
    (option="sadf") sequence exceeds the matching critical value curve,
    filtered to episodes of at least `min_duration`. Start/peak/end are
    positions into the original series (0-indexed, offset by minw + lag to
    account for the recursive window's warm-up).
    """
    if sig_lvl not in SIG_IDX:
        raise ValueError("sig_lvl must be one of 90, 95, 99")
    if option not in ("gsadf", "sadf"):
        raise ValueError("option must be 'gsadf' or 'sadf'")
    if min_duration < 0:
        raise ValueError("min_duration must be non-negative")
    sidx = SIG_IDX[sig_lvl]

    if option == "gsadf":
        tstat_seq, cv_seq = result.bsadf, cv.bsadf_cv
        tstat_overall, cv_overall = result.gsadf, cv.gsadf_cv
    else:
        tstat_seq, cv_seq = result.badf, cv.badf_cv
        tstat_overall, cv_overall = result.sadf, cv.sadf_cv

    nc = tstat_seq.shape[1]
    names = result.series_names or [f"series{i + 1}" for i in range(nc)]
    zadj = result.minw + result.lag

    out: dict[str, list[Episode]] = {}
    for j in range(nc):
        if tstat_overall[j] <= _cv_overall(cv_overall, j)[sidx]:
            continue  # doesn't reject the null overall -- not date-stamped

        series_tstat = tstat_seq[:, j]
        series_cv = _cv_curve(cv_seq, j)[:, sidx]
        exceed = np.where(series_tstat > series_cv)[0]

        episodes = []
        for start, end in _stamp(exceed):
            duration = end - start
            if duration < min_duration:
                continue
            peak = start + int(np.argmax(series_tstat[start:end]))
            ongoing = end >= len(series_tstat)
            episodes.append(
                Episode(
                    start=start + zadj,
                    peak=peak + zadj,
                    end=None if ongoing else end + zadj,
                    duration=duration,
                    ongoing=ongoing,
                )
            )
        if episodes:
            out[names[j]] = episodes

    return out
