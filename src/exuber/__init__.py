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
  - sim_*: bubble DGP simulators, pure Python + numpy, in sim.py. Not
    re-exported here (see "Not re-exported" below) -- import directly,
    e.g. `from exuber.sim import sim_psy1`.
  - datestamp(): episode date-stamping (Start/Peak/End/Duration/Ongoing),
    with one simplification -- see datestamp.py's module docstring.
  - radf_crit(): precomputed Monte Carlo critical values from the shared
    store exuber's R package also reads (crit.py), so a typical analysis
    needn't simulate its own.
  - radf_common()/radf_common_cv() (Chen, Phillips & Shi 2023 common-
    bubble detection via PCA + PSY): radf_common.py.
  - cobubble_test() (Evripidou, Harvey, Leybourne & Sollis 2022
    co-explosive behaviour test): cobubble_test.py.
  - contagion_reg() (Greenaway-McGrevy & Phillips 2016 bubble contagion
    regression, minimum-viable subset): contagion_reg.py.
  - monitor() (Phillips & Shi 2020 training/monitoring split, Kurozumi
    2020 / Homm & Breitung 2012 boundaries): monitor.py.
  - monitor_cusum() (Homm & Breitung 2012 CUSUM real-time monitoring):
    monitor_cusum.py.
  - lbi_test()/monitor_lbi() (Breitung & Diegel 2025 locally best
    invariant test): lbi_test.py.
  - quantile_test() (Wu, Shi & Wu 2025 quantile-regression global test):
    quantile_test.py.
  - monitor_quantile() (Wu, Shi & Wu 2025 QPWY recursive quantile
    monitoring): monitor_quantile.py. See these modules' own docstrings
    for exactly which boundary variants of each are ported vs. deferred.
  - rootstamp()/rootstamp_episodes() (Guo, Sun & Wang 2019 / Phillips-
    Magdalinos 2007 root inference): rootstamp.py.
  - dating_pdc() (Pang, Du & Chong 2021 / Kurozumi & Skrobotov 2023
    sequential sample-splitting dating): dating_pdc.py.
  - radf_recovery()/radf_recovery_cv() (Phillips & Shi 2014 reverse-
    regression recovery dating): radf_recovery.py.
  - dating_hls() (Harvey, Leybourne & Sollis 2017 SSR+BIC dating):
    dating_hls.py.
  - dating_hlw() (Harvey, Leybourne & Whitehouse 2020 multi-bubble
    wrapper around dating_hls()): dating_hlw.py.
  - dating_knp() (Kejriwal, Nguyen & Perron 2025 bias-corrected single-
    bubble dating): dating_knp.py.
  - tidy()/augment(): DataFrame-producing accessors for a RadfResult
    (radf_obj's methods only, not radf_cv/radf_distr's, and not
    tidy_join/augment_join/summary/diagnostics -- see tidy.py's module
    docstring). Needs pandas (`pip install pyexuber[pandas]`), lazily
    imported.

Only the callable functions above are exported here. Each one's return
type (RadfCv, DatingHlsResult, MonitorResult, ...) is a plain dataclass
defined next to it and importable from its own submodule when you need
it for a type hint or isinstance check, e.g. `from exuber.cv import
RadfCv` or `from exuber.dating_hls import DatingHlsResult` -- not
re-exported here, since most usage never needs to name the type.

Not re-exported here: exuber.sim's sim_*() DGP simulators. They remain
fully usable via `from exuber.sim import sim_psy1` etc., just not part
of the top-level `exuber` namespace.

Not yet ported (deferred, not silently dropped):
  - the remaining 2026-08 sim_*() DGP extensions: sim_coexplosive,
    sim_common, sim_falsebubble, sim_mar, sim_msbubble, sim_tree,
    sim_dgp1/2 -- larger, more involved DGPs than the innovation
    generators above, left for later.
  - tidy()/augment() for radf_cv/radf_distr, tidy_join()/augment_join(),
    summary(), diagnostics().
  - QPSY (monitor_quantile()'s double-recursion sibling): O(T^2) QR fits,
    a materially larger cost class than QPWY's O(T), not attempted.

Note on radf_sb_cv/distr: this port's bootstrap DGP prepends the *full*
initmat[j, :] (reversed) rather than R's original initmat[j, lag:1],
which was one element short of the lag + 1 the recursive AR filter needs
for lag > 0 (R's index-0 drop rule made `lag:1` accidentally correct only
at lag = 0). That was a real bug in exuber's own R/radf_sb.R, since
fixed upstream (initmat[j, (lag + 1):1]) -- see cv.py's `_radf_sb`.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from exuber.cobubble_test import cobubble_test
from exuber.contagion_reg import contagion_reg
from exuber.crit import radf_crit
from exuber.cv import (
    radf_mc_cv,
    radf_mc_distr,
    radf_sb_cv,
    radf_sb_distr,
    radf_wb_cv,
    radf_wb_distr,
    radf_wb_ps_cv,
    radf_wb_ps_distr,
)
from exuber.datestamp import datestamp
from exuber.dating_hls import dating_hls
from exuber.dating_hlw import dating_hlw
from exuber.dating_knp import dating_knp
from exuber.dating_pdc import dating_pdc
from exuber.lbi_test import lbi_test, monitor_lbi
from exuber.monitor import monitor
from exuber.monitor_cusum import monitor_cusum
from exuber.monitor_quantile import monitor_quantile
from exuber.quantile_test import quantile_test
from exuber.radf import psy_ds, psy_minw, radf
from exuber.radf_common import radf_common, radf_common_cv
from exuber.radf_recovery import radf_recovery, radf_recovery_cv
from exuber.rootstamp import rootstamp, rootstamp_episodes
from exuber.tidy import augment, tidy

try:
    __version__ = _version("pyexuber")
except PackageNotFoundError:  # source tree on sys.path without an install
    __version__ = "0+unknown"

__all__ = [
    "radf",
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
    "datestamp",
    "radf_common",
    "radf_common_cv",
    "cobubble_test",
    "contagion_reg",
    "monitor",
    "monitor_cusum",
    "lbi_test",
    "monitor_lbi",
    "quantile_test",
    "monitor_quantile",
    "rootstamp",
    "rootstamp_episodes",
    "dating_pdc",
    "radf_recovery",
    "radf_recovery_cv",
    "dating_hls",
    "dating_hlw",
    "dating_knp",
    "tidy",
    "augment",
]
