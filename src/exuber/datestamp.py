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

`option="svadf"` (Sarkar & Wells 2026, arXiv:2604.12062, a non-peer-
reviewed preprint) is a structurally different dating rule, folded in the
same way R's `datestamp.radf_obj()` folds it (see R/radf-methods.R,
R/svadf.R): reuses `radf()`'s own `badf` sequence directly (no `cv`
needed), comparing it against two different closed-form, sample-size-only
thresholds -- `log(t)/10` for origination, `log(t)/2` for collapse (the
paper's own Section 5.1 calibration) -- and detects at most one
origination/collapse pair per series, unlike the `cv`-based options which
can find multiple episodes.
"""

import warnings
from dataclasses import dataclass

import numpy as np

from exuber.cv import RadfCv
from exuber.radf import RadfResult

SIG_IDX = {90: 0, 95: 1, 99: 2}

SVADF_CAVEAT = (
    "datestamp(option='svadf'): Sarkar & Wells (2026) is a non-peer-reviewed "
    "preprint, a different bar than every other method in this package."
)


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


def svadf_threshold(t: np.ndarray, kind: str) -> np.ndarray:
    """Sarkar & Wells (2026)'s closed-form, sample-size-only thresholds
    (Section 5.1): `log(t)/10` for origination, `log(t)/2` for collapse."""
    if kind not in ("origination", "collapse"):
        raise ValueError("kind must be 'origination' or 'collapse'")
    return np.log(t) / 10 if kind == "origination" else np.log(t) / 2


def _datestamp_svadf(result: RadfResult, min_duration: int) -> dict[str, list[Episode]]:
    """SV-ADF asymmetric-threshold dating (Sarkar & Wells 2026): unlike the
    cv-based options, origination and collapse compare `badf` against two
    DIFFERENT thresholds, so it doesn't reduce to a shared `tstat > crit`
    boolean -- detects at most one origination/collapse pair per series
    (the paper's own procedure)."""
    warnings.warn(SVADF_CAVEAT, stacklevel=3)
    badf = result.badf
    pointer, nc = badf.shape
    names = result.series_names or [f"series{i + 1}" for i in range(nc)]
    zadj = result.minw + result.lag
    t_idx = zadj + np.arange(1, pointer + 1)
    orig_thresh = svadf_threshold(t_idx, "origination")
    coll_thresh = svadf_threshold(t_idx, "collapse")

    out: dict[str, list[Episode]] = {}
    for j in range(nc):
        series = badf[:, j]
        above = np.where(series > orig_thresh)[0]
        if len(above) == 0:
            continue
        orig_runs = [r for r in _stamp(above) if (r[1] - r[0]) >= min_duration]
        if not orig_runs:
            continue
        start = orig_runs[0][0]

        below = np.where(series < coll_thresh)[0]
        below = below[below > start]
        end = pointer  # sentinel: one past the last row -- no collapse found, ongoing
        if len(below) > 0:
            coll_runs = [r for r in _stamp(below) if (r[1] - r[0]) >= min_duration]
            if coll_runs:
                end = coll_runs[0][0]

        duration = end - start
        peak = start + int(np.argmax(series[start:end]))
        ongoing = end >= pointer
        out[names[j]] = [
            Episode(
                start=start + zadj,
                peak=peak + zadj,
                end=None if ongoing else end + zadj,
                duration=duration,
                ongoing=ongoing,
            )
        ]
    return out


def datestamp(
    result: RadfResult,
    cv: RadfCv | None = None,
    min_duration: int = 0,
    sig_lvl: int = 95,
    option: str = "gsadf",
) -> dict[str, list[Episode]]:
    """Date-stamp periods of explosive behaviour.

    For each series whose overall statistic rejects the null at `sig_lvl`,
    finds contiguous runs where the BSADF (option="gsadf") or BADF
    (option="sadf") sequence exceeds the matching critical value curve,
    filtered to episodes of at least `min_duration`. Start/peak/end are
    positions into the original series (0-indexed, offset by minw + lag to
    account for the recursive window's warm-up).

    `option="svadf"` uses a structurally different rule (see module
    docstring) and does not need `cv` at all.
    """
    if option not in ("gsadf", "sadf", "svadf"):
        raise ValueError("option must be 'gsadf', 'sadf' or 'svadf'")
    if min_duration < 0:
        raise ValueError("min_duration must be non-negative")

    if option == "svadf":
        return _datestamp_svadf(result, min_duration)

    if cv is None:
        raise ValueError("cv is required unless option='svadf'")
    if sig_lvl not in SIG_IDX:
        raise ValueError("sig_lvl must be one of 90, 95, 99")
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
