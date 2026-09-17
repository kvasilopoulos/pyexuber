"""Multi-bubble SSR/BIC dating (Harvey, Leybourne & Whitehouse 2020).
Ported from exuber's R/dating_hlw.R (see docs/dating-and-root-
inference.md in the umbrella repo for the full methodology and
validation record).

A two-step wrapper around dating_hls(): Step 1 runs PSY's existing
detection+dating (radf()/datestamp()) to get preliminary start/end
positions for each detected explosive episode, and carves the sample
into disjoint "date windows" (splitting at the midpoint between one
episode's end and the next one's start). Step 2 applies
exuber._hls_common's _hls_fit_series() independently within each window
-- restricted to models {2, 4} for every window but the last (a window
boundary is by construction a unit-root point, not a genuine sample end)
-- with a sequential adjustment rule so window j+1 always starts at the
first observation of window j's just-fitted post-explosive regime.

Deviation from R (an improvement, not a workaround): R's dating_hlw()
has to wrap its datestamp() call in tryCatch(error=, warning=) because
R's datestamp() raises a hard error when no series has any detected
episode. Python's datestamp() never raises for that case -- it simply
returns an empty dict per series (see datestamp.py) -- so no
try/except is needed here at all.
"""

import math
from dataclasses import dataclass

import numpy as np

from exuber._hls_common import _hls_fit_series
from exuber.cv import radf_wb_cv
from exuber.datestamp import Episode, datestamp
from exuber.radf import _to_2d_array, psy_ds, psy_minw, radf


def _hlw_local_to_global(local_tau: int, s0: int) -> int:
    """Global, R-bit-for-bit 1-indexed output position for a window-local
    raw boundary count (from _hls_fit_series(), before dating_hls()'s own
    "+1" conversion), given the window's 0-indexed global start `s0`."""
    return s0 + local_tau + 1


@dataclass
class HlwEpisode:
    model: int  # 2 or 4 for every non-final window, 1-4 for the final one
    origination: float
    collapse: float
    recovery: float  # NaN unless model == 4


def _dating_hlw_from_episodes(
    y: np.ndarray, episodes: list[Episode], n: int, trim: float
) -> list[HlwEpisode]:
    """Step 2 (window construction + per-window HLS fitting + sequential
    start adjustment), given a single series' already step-1-detected
    episodes -- factored out so it's testable with hand-built Episodes,
    no radf()/datestamp()/the C++ extension needed (same split as
    radf_recovery's _recovery_dates_from_bsadf()).
    """
    nhat = len(episodes)
    if nhat == 0:
        return []

    starts = [ep.start for ep in episodes]
    ends = [ep.end if ep.end is not None else n for ep in episodes]

    e = [0] * nhat
    for jj in range(nhat):
        if jj < nhat - 1:
            e[jj] = ends[jj] + (starts[jj + 1] - ends[jj]) // 2
        else:
            e[jj] = n

    out: list[HlwEpisode] = []
    s0 = 0
    for jj in range(nhat):
        if s0 >= e[jj]:
            continue
        y_win = y[s0 : e[jj]]
        models_allowed = (2, 4) if jj < nhat - 1 else (1, 2, 3, 4)
        fit = _hls_fit_series(y_win, trim, models=models_allowed)

        tau1, tau2, tau3 = fit.breaks.get("tau1"), fit.breaks.get("tau2"), fit.breaks.get("tau3")
        origination = _hlw_local_to_global(tau1, s0) if tau1 is not None else math.nan
        collapse = _hlw_local_to_global(tau2, s0) if tau2 is not None else math.nan
        recovery = _hlw_local_to_global(tau3, s0) if tau3 is not None else math.nan
        out.append(
            HlwEpisode(
                model=fit.model, origination=origination, collapse=collapse, recovery=recovery
            )
        )

        if jj < nhat - 1:
            local_break = tau2 if fit.model == 2 else tau3
            assert local_break is not None
            s0 = s0 + local_break

    return out


@dataclass
class DatingHlwResult:
    episodes: dict[str, list[HlwEpisode]]  # one entry per series with >=1 window
    series_names: list[str] | None
    trim: float
    minw: int
    n: int


def dating_hlw(
    data,
    cv=None,
    minw: int | None = None,
    trim: float = 0.1,
    min_duration: int | None = None,
    nboot: int = 199,
    seed: int | None = None,
) -> DatingHlwResult:
    """Multi-bubble SSR/BIC dating (Harvey, Leybourne & Whitehouse 2020).
    Extends dating_hls() to series with more than one explosive episode:
    runs PSY's existing detection/dating (radf()/datestamp()) to locate
    preliminary episode windows, then re-dates each window with
    dating_hls()-style SSR/BIC fitting (restricted to models {2, 4} for
    every window but the last).

    When exactly one episode is detected, this reduces exactly to
    dating_hls() applied to the whole series -- the paper's own stated
    property, verified by test (the single window then runs [0, n) and
    fits all four models).

    The step-2 SSR/BIC dating needs no critical values, same as
    dating_hls(). Step 1's PSY detection does use a wild bootstrap
    critical value (cv/nboot/seed below, defaulting to radf_wb_cv())
    only to locate the preliminary episode windows.
    """
    x, columns = _to_2d_array(data)
    n, nc = x.shape
    names = columns or [f"series{i + 1}" for i in range(nc)]
    minw = minw if minw is not None else psy_minw(n)
    min_duration = min_duration if min_duration is not None else psy_ds(n)

    full = radf(x, minw=minw)
    if cv is None:
        cv = radf_wb_cv(x, minw=minw, nboot=nboot, seed=seed)
    ds = datestamp(full, cv, min_duration=min_duration)

    results: dict[str, list[HlwEpisode]] = {}
    for j, name in enumerate(names):
        results[name] = _dating_hlw_from_episodes(x[:, j], ds.get(name, []), n, trim)

    return DatingHlwResult(episodes=results, series_names=names, trim=trim, minw=minw, n=n)
