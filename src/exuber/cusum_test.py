"""cusum_test(): Kurozumi & Nishi (2025)'s retrospective CUSUM (CS/GCS) and
CUSUM-of-squares (CSSQ/GCSSQ) bubble tests. Ported from exuber's
R/cusum_test.R -- see docs/volatility-robustness.md (root repo).

Every statistic is a sup/inf of a partial-sum process of Delta y_t (or
its square), so the "generalized" (every window start) versions reduce to
a running max/min: O(T), no double loop. CS/GCS reject on the right tail;
CSSQ/GCSSQ are two-sided, each tail at alpha/2. Critical values are Table
I's published asymptotic ones (ssu_test._KN_TABLE).
"""

from dataclasses import dataclass

import numpy as np

from exuber.radf import _to_2d_array
from exuber.ssu_test import ssu_q

_TYPES = ("cs", "gcs", "cssq", "gcssq")


def _cusum_test_path(y: np.ndarray, type: str) -> tuple[np.ndarray, np.ndarray | None]:
    """(sup path, inf path or None), one value per end point k = 1..T-1."""
    d = np.diff(y)
    nd = len(d)
    if type in ("cs", "gcs"):
        p = np.concatenate([[0.0], np.cumsum(d)]) / np.sqrt(np.mean(d**2) * nd)
    else:
        d2 = d**2
        dev = np.cumsum(d2) - np.arange(1, nd + 1) / nd * d2.sum()
        p = np.concatenate([[0.0], dev]) / np.sqrt((np.mean(d2**2) - np.mean(d2) ** 2) * nd)
    cur = p[1:]
    if type == "cs":
        return cur, None
    if type == "cssq":
        return cur, cur
    drawup = cur - np.minimum.accumulate(p)[:-1]
    if type == "gcs":
        return drawup, None
    return drawup, cur - np.maximum.accumulate(p)[:-1]


@dataclass
class CusumTestResult:
    """Output of cusum_test(): sup path (and inf path for CSSQ/GCSSQ), the
    statistic(s), Table I critical value(s) and the decision per series."""

    stat: np.ndarray  # (n - 1, nc)
    sup: np.ndarray  # (nc,)
    crit: tuple[float, ...]  # (crit,) or (sup, inf)
    detected: np.ndarray  # bool, (nc,)
    type: str
    n: int
    sig_lvl: float
    stat_inf: np.ndarray | None = None
    inf: np.ndarray | None = None
    series_names: list[str] | None = None


def cusum_test(data, sig_lvl: float = 95, type: str = "cs") -> CusumTestResult:
    """Kurozumi & Nishi (2025)'s CUSUM (type="cs"), generalized CUSUM
    ("gcs"), CUSUM-of-squares ("cssq") and generalized CUSUM-of-squares
    ("gcssq") bubble tests on the first differences. CS/GCS reject when
    the cumulated increments get too large; CSSQ/GCSSQ are two-sided,
    rejecting when the cumulated squared increments drift too far above or
    below their full-sample average. The paper finds the CUSUM type loses
    almost all power once the explosive coefficient is genuinely
    stochastic; see ssu_test() for its more powerful statistics."""
    if type not in _TYPES:
        raise ValueError(f"type must be one of {_TYPES}")
    x, columns = _to_2d_array(data)
    n, nc = x.shape
    two_sided = type in ("cssq", "gcssq")
    crit_sup = ssu_q(sig_lvl, f"{type}_sup" if two_sided else type)
    crit_inf = ssu_q(sig_lvl, f"{type}_inf") if two_sided else None
    crit = (crit_sup,) if crit_inf is None else (crit_sup, crit_inf)
    stat = np.empty((n - 1, nc))
    stat_inf = np.empty((n - 1, nc)) if two_sided else None
    for j in range(nc):
        sup_path, inf_path = _cusum_test_path(x[:, j], type)
        stat[:, j] = sup_path
        if stat_inf is not None:
            stat_inf[:, j] = inf_path
    sup = stat.max(axis=0)
    if stat_inf is not None and crit_inf is not None:
        inf = stat_inf.min(axis=0)
        detected = (sup >= crit_sup) | (inf <= crit_inf)
    else:
        inf = None
        detected = sup > crit_sup
    return CusumTestResult(
        stat=stat, sup=sup, crit=crit, detected=detected, type=type, n=n,
        sig_lvl=sig_lvl, stat_inf=stat_inf, inf=inf, series_names=columns,
    )
