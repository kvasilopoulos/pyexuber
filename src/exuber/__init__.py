"""pyexuber: Python bindings for exubercore (recursive right-tailed unit
root tests for explosive time series).

Scope so far:
  - radf(): the core recursive ADF/SADF/GSADF/BSADF statistic (C++, via
    exubercore).
  - radf_mc_cv/distr, radf_wb_cv/distr (HLST wild bootstrap),
    radf_wb_ps_cv/distr (Phillips-Shi wild bootstrap variant): critical
    values / distributions, pure Python + numpy (RNG-driven, mirrors
    exuber's R orchestration around the same core statistic). lag_select()/
    adf_res() (exuber._lagselect, internal) are the deterministic
    exception, verified bit-for-bit against R.
  - sim_*: bubble DGP simulators, pure Python + numpy.
  - datestamp(): episode date-stamping (Start/Peak/End/Duration/Ongoing),
    with one simplification -- see datestamp.py's module docstring.
  - radf_crit(): precomputed Monte Carlo critical values from the shared
    store exuber's R package also reads (crit.py), so a typical analysis
    needn't simulate its own.

Not yet ported (deferred, not silently dropped):
  - radf_sb_cv/distr (sieve bootstrap, R/radf_sb.R): needs the same
    lag-selection subsystem plus its own bootstrap DGP loop, which needs
    verification against R directly before shipping it, not a guess.
  - the 2026-08 sim_*() DGP extensions (sim_coexplosive, sim_common,
    sim_falsebubble, sim_fi, sim_mar, sim_msbubble, sim_tree, sim_vol_*,
    sim_dgp1/2, sim_innov, and optional axes on sim_psy1/sim_blan).
  - .summary()/.tidy()/.diagnostics() DataFrame-producing methods.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from exuber.crit import radf_crit
from exuber.cv import (
    RadfCv,
    RadfDistr,
    radf_mc_cv,
    radf_mc_distr,
    radf_wb_cv,
    radf_wb_distr,
    radf_wb_ps_cv,
    radf_wb_ps_distr,
)
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
    "radf_wb_ps_cv",
    "radf_wb_ps_distr",
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
