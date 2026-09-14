"""pyexuber: Python bindings for exubercore (recursive right-tailed unit
root tests for explosive time series).

Scope so far:
  - radf(): the core recursive ADF/SADF/GSADF/BSADF statistic (C++, via
    exubercore).
  - radf_mc_cv/distr, radf_wb_cv/distr (HLST wild bootstrap): critical
    values / distributions, pure Python + numpy (RNG-driven, mirrors
    exuber's R orchestration around the same core statistic).
  - sim_*: bubble DGP simulators, pure Python + numpy.
  - datestamp(): episode date-stamping (Start/Peak/End/Duration/Ongoing),
    with one simplification -- see datestamp.py's module docstring.
  - radf_crit(): precomputed Monte Carlo critical values from the shared
    store exuber's R package also reads (crit.py), so a typical analysis
    needn't simulate its own.

Not yet ported (deferred, not silently dropped):
  - radf_wb_cv2/distr2 (Phillips & Shi PS wild bootstrap variant): needs
    an OLS-based lag-selection/AR-fit subsystem (adf_res/lag_select in
    exuber's R/radf_wb.R) not built here yet.
  - radf_sb_cv/distr (sieve bootstrap): exuber's R implementation appears
    to overwrite rather than accumulate across panel series inside its
    bootstrap loop (R/radf_sb.R) -- porting that faithfully needs
    verification against R directly before shipping it, not a guess.
  - .summary()/.tidy()/.diagnostics() DataFrame-producing methods.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from exuber.crit import radf_crit
from exuber.cv import RadfCv, RadfDistr, radf_mc_cv, radf_mc_distr, radf_wb_cv, radf_wb_distr
from exuber.datestamp import Episode, datestamp
from exuber.radf import RadfResult, psy_ds, psy_minw, radf
from exuber.sim import sim_blan, sim_div, sim_evans, sim_ps1, sim_ps2, sim_psy1, sim_psy2

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
    "RadfCv",
    "RadfDistr",
    "datestamp",
    "Episode",
    "sim_psy1",
    "sim_psy2",
    "sim_ps1",
    "sim_ps2",
    "sim_blan",
    "sim_evans",
    "sim_div",
]
