"""pyexuber: Python bindings for exubercore, with recursive right-tailed unit
root tests for explosive time series.

The package contains the following parts.

  - radf() computes the recursive ADF, SADF, GSADF and BSADF statistics. It
    is written in C++ and reaches Python through exubercore.
  - radf_mc_cv/distr, radf_wb_cv/distr (HLST wild bootstrap),
    radf_wb_ps_cv/distr (Phillips-Shi wild bootstrap variant) and
    radf_sb_cv/distr (sieve bootstrap) give critical values and
    distributions. They are written in Python and numpy, because they are
    driven by a random number generator, and they mirror exuber's R code
    around the same core statistic. lag_select() and adf_res() (in
    exuber._lagselect, internal) are deterministic, and we verified them
    bit for bit against R.
  - The sim_*() functions in sim.py simulate bubble DGPs. They are not
    re-exported (see "Not re-exported" below). Import them directly, for
    example `from exuber.sim import sim_psy1`.
  - datestamp() dates the episodes (Start, Peak, End, Duration, Ongoing). It
    makes one simplification, described in the module docstring of
    datestamp.py.
  - radf_crit() returns precomputed Monte Carlo critical values from the
    shared store that exuber's R package also reads (crit.py), so a typical
    analysis does not need to simulate its own.
  - radf_common() and radf_common_cv() (Chen, Phillips & Shi 2023) detect a
    common bubble by principal components and PSY (radf_common.py).
  - cobubble_test() is the co-explosive behaviour test of Evripidou, Harvey,
    Leybourne & Sollis 2022 (cobubble_test.py).
  - contagion_reg() is the bubble contagion regression of Greenaway-McGrevy
    & Phillips 2016, in a minimal subset (contagion_reg.py).
  - monitor() implements the training and monitoring split of Phillips & Shi
    2020, with the boundaries of Kurozumi 2020 and Homm & Breitung 2012
    (monitor.py).
  - monitor_cusum() is the CUSUM real-time monitor of Homm & Breitung 2012
    (monitor_cusum.py).
  - lbi_test() and monitor_lbi() implement the locally best invariant test of
    Breitung & Diegel 2025 (lbi_test.py).
  - quantile_test() is the quantile-regression global test of Wu, Shi & Wu
    2025 (quantile_test.py).
  - monitor_quantile() is the recursive quantile monitor (QPWY and QPSY) of
    Wu, Shi & Wu 2025 (monitor_quantile.py). The module docstrings of these
    monitors say which boundary variants are ported and which are deferred.
  - rootstamp() and rootstamp_episodes() estimate the root of each episode
    (Guo, Sun & Wang 2019; Phillips & Magdalinos 2007) (rootstamp.py).
  - dating_pdc() dates bubbles by sequential sample splitting (Pang, Du &
    Chong 2021; Kurozumi & Skrobotov 2023) (dating_pdc.py).
  - radf_recovery() and radf_recovery_cv() date a bubble by the reverse
    regression of Phillips & Shi 2014 (radf_recovery.py).
  - dating_hls() dates bubbles by SSR and BIC (Harvey, Leybourne & Sollis
    2017) (dating_hls.py).
  - dating_hlw() wraps dating_hls() for several bubbles (Harvey, Leybourne &
    Whitehouse 2020) (dating_hlw.py).
  - dating_knp() is the bias-corrected dating of Kejriwal, Nguyen & Perron
    2025, for one or several bubbles (dating_knp.py).
  - tidy(), augment(), tidy_join() and augment_join() return DataFrames for
    results, critical values and distributions (see the module docstring of
    tidy.py). They need pandas (`pip install pyexuber[pandas]`), which is
    imported lazily.
  - diagnostics() and summary() give a reject or not-reject verdict for each
    series and a table of statistics against critical values
    (diagnostics.py). summary() needs pandas, and diagnostics() does not.

Only the callable functions above are exported here. The return type of each
one (RadfCv, DatingHlsResult, MonitorResult and so on) is a plain dataclass
defined next to the function. It can be imported from its own submodule when
you need it for a type hint or an isinstance check, for example
`from exuber.cv import RadfCv` or
`from exuber.dating_hls import DatingHlsResult`. We do not re-export these
types, because most code never needs to name them.

Not re-exported here: the sim_*() DGP simulators of exuber.sim. They remain
fully usable through `from exuber.sim import sim_psy1` and similar imports,
but they are not part of the top-level `exuber` namespace.

Not yet ported. These parts are deferred on purpose and have not been
dropped by accident.
  - The bootstrap critical values of monitor_quantile() (Algorithm 1 of the
    paper). Each replicate costs the full O(T^2) QR sweep of QPSY. The
    asymptotic boundary is ported, with its small-sample caveat.

Note on radf_sb_cv/distr. The bootstrap DGP in this port prepends the full
initmat[j, :] in reverse. R originally used initmat[j, lag:1], which is one
element short of the lag + 1 values that the recursive AR filter needs when
lag > 0. R's rule of dropping index 0 made `lag:1` correct only at lag = 0.
This was a real bug in exuber's own R/radf_sb.R, and it has since been fixed
upstream (initmat[j, (lag + 1):1]). See `_radf_sb` in cv.py.
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
from exuber.diagnostics import diagnostics, summary
from exuber.lbi_test import lbi_test, monitor_lbi
from exuber.monitor import monitor
from exuber.monitor_cusum import monitor_cusum
from exuber.monitor_quantile import monitor_quantile
from exuber.quantile_test import quantile_test
from exuber.radf import psy_ds, psy_minw, radf
from exuber.radf_common import radf_common, radf_common_cv
from exuber.radf_recovery import radf_recovery, radf_recovery_cv
from exuber.rootstamp import rootstamp, rootstamp_episodes
from exuber.tidy import augment, augment_join, tidy, tidy_join

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
    "tidy_join",
    "augment_join",
    "diagnostics",
    "summary",
]
