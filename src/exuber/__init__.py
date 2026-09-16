"""pyexuber: Python bindings for exubercore (recursive right-tailed unit
root tests for explosive time series).

Scope so far:
  - radf(): the core recursive ADF/SADF/GSADF/BSADF statistic (C++, via
    exubercore).
  - radf_mc_cv/distr, radf_wb_cv/distr (HLST wild bootstrap),
    radf_wb_ps_cv/distr (Phillips-Shi wild bootstrap variant),
    radf_sb_cv/distr (sieve bootstrap): critical values / distributions,
    pure Python + numpy (RNG-driven, mirrors exuber's R orchestration
    around the same core statistic). lag_select()/adf_res()
    (exuber._lagselect, internal) are the deterministic exception,
    verified bit-for-bit against R.
  - sim_*: bubble DGP simulators, pure Python + numpy.
  - datestamp(): episode date-stamping (Start/Peak/End/Duration/Ongoing),
    with one simplification -- see datestamp.py's module docstring.
  - radf_crit(): precomputed Monte Carlo critical values from the shared
    store exuber's R package also reads (crit.py), so a typical analysis
    needn't simulate its own.
  - tidy()/augment(): DataFrame-producing accessors for a RadfResult
    (radf_obj's methods only, not radf_cv/radf_distr's, and not
    tidy_join/augment_join/summary/diagnostics -- see tidy.py's module
    docstring). Needs pandas (`pip install pyexuber[pandas]`), lazily
    imported.

Not yet ported (deferred, not silently dropped):
  - the 2026-08 sim_*() DGP extensions (sim_coexplosive, sim_common,
    sim_falsebubble, sim_fi, sim_mar, sim_msbubble, sim_tree, sim_vol_*,
    sim_dgp1/2, sim_innov, and optional axes on sim_psy1/sim_blan).
  - tidy()/augment() for radf_cv/radf_distr, tidy_join()/augment_join(),
    summary(), diagnostics().

Note on radf_sb_cv/distr: exuber's own R/radf_sb.R has an off-by-one bug
in its bootstrap DGP for lag > 0 (`initmat[j, lag:1]` is one element short
of the `lag + 1` the recursive AR filter needs, confirmed by direct R
inspection -- `type = "fixed"`'s default lag = 0 is unaffected, which is
why the existing `radf_sb_cv_aic_bic_validation.R` script didn't catch
it). This port does not reproduce that bug -- see cv.py's `_radf_sb`.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from exuber.crit import radf_crit
from exuber.cv import (
    RadfCv,
    RadfDistr,
    RadfSbCv,
    RadfSbDistr,
    radf_mc_cv,
    radf_mc_distr,
    radf_sb_cv,
    radf_sb_distr,
    radf_wb_cv,
    radf_wb_distr,
    radf_wb_ps_cv,
    radf_wb_ps_distr,
)
from exuber.datestamp import Episode, datestamp
from exuber.radf import RadfResult, psy_ds, psy_minw, radf
from exuber.sim import sim_blan, sim_div, sim_evans, sim_ps1, sim_ps2, sim_psy1, sim_psy2
from exuber.tidy import augment, tidy

try:
    __version__ = _version("pyexuber")
except PackageNotFoundError:  # source tree on sys.path without an install
    __version__ = "0+unknown"

__all__ = [
    "radf",
    "RadfResult",
    "psy_minw",
    "psy_ds",
    "radf_crit",
    "radf_mc_cv",
    "radf_mc_distr",
    "radf_wb_cv",
    "radf_wb_distr",
    "radf_wb_ps_cv",
    "radf_wb_ps_distr",
    "radf_sb_cv",
    "radf_sb_distr",
    "RadfCv",
    "RadfDistr",
    "RadfSbCv",
    "RadfSbDistr",
    "datestamp",
    "Episode",
    "sim_psy1",
    "sim_psy2",
    "sim_ps1",
    "sim_ps2",
    "sim_blan",
    "sim_evans",
    "sim_div",
    "tidy",
    "augment",
]
