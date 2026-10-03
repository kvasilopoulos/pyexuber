"""SSU and GSSU: tests for a stochastic explosive coefficient (Kurozumi & Nishi
2025), with their UR/GUR procedure that takes the union of rejections. Ported
from exuber's R/ssu_test.R. See docs/volatility-robustness.md (root repo) for
the formulas, the papers and the independent validation of this port. Every
critical value is the published asymptotic constant of Table I (_KN_TABLE,
which cusum_test.py also uses).
"""

from dataclasses import dataclass

import numpy as np

from exuber.radf import _to_2d_array, psy_minw

_SSU_LEVELS = (90, 95, 99)
# Kurozumi & Nishi (2025) Table I (their 10,000-rep Monte Carlo): SSU at
# r0 = psy_minw()'s formula, GSSU at r0 = -0.004 + 2.24/sqrt(T), the union
# scaling constants ur/gur, and the CUSUM-family columns (CSSQ/GCSSQ are
# two-sided, each tail at alpha/2, so the level-alpha row is the test at alpha).
_KN_TABLE = {
    "ssu": (2.90, 3.30, 4.20),
    "gssu": (4.83, 5.37, 6.81),
    "ur": (1.16, 1.13, 1.09),
    "gur": (1.11, 1.10, 1.08),
    "cs": (1.62, 1.93, 2.57),
    "gcs": (1.90, 2.20, 2.78),
    "cssq_sup": (1.19, 1.32, 1.59),
    "cssq_inf": (-1.21, -1.34, -1.60),
    "gcssq_sup": (1.60, 1.72, 1.98),
    "gcssq_inf": (-1.62, -1.72, -1.97),
}


def ssu_q(sig_lvl: float, stat: str = "ssu") -> float:
    """Kurozumi & Nishi (2025) Table I's published asymptotic critical
    value for `stat` (from their own Monte Carlo with 10,000 replications).
    No simulation is needed."""
    for lvl, crit in zip(_SSU_LEVELS, _KN_TABLE[stat], strict=True):
        if abs(sig_lvl - lvl) < 1e-8:
            return crit
    raise ValueError(
        f"sig_lvl must be one of {_SSU_LEVELS} (Kurozumi & Nishi (2025)'s "
        "Table I only tabulates these significance levels)"
    )


def ssu_prefix_sums(y: np.ndarray) -> dict:
    """All cumulative sums SSU's closed form needs, built from x1 =
    y_{t-1} (level lag) and d1 = Delta y_t (difference); every term in the
    ADF regression (eq. 6), the SSU regression (eq. 7), and the
    bias-correction cross-moment reduces to a window sum of one of these
    twelve products."""
    y = np.asarray(y, dtype=float)
    n1 = len(y) - 1
    x1 = y[:n1]
    d1 = y[1 : n1 + 1] - x1

    def mk(v: np.ndarray) -> np.ndarray:
        return np.concatenate(([0.0], np.cumsum(v)))

    return {
        "n1": n1,
        "x1": mk(x1), "x1_2": mk(x1**2), "x1_3": mk(x1**3), "x1_4": mk(x1**4),
        "d1": mk(d1), "d1_2": mk(d1**2), "d1_3": mk(d1**3), "d1_4": mk(d1**4),
        "d1x1": mk(d1 * x1),
        "d1_2x1_2": mk(d1**2 * x1**2),
        "d1x1_2": mk(d1 * x1**2),
        "x1d1_2": mk(x1 * d1**2),
    }


def ssu_stat_path(ps: dict, hi_idx: np.ndarray, lo=0) -> np.ndarray:
    """t^{omega,c}_{r1,r2} for every window (lo, hi] of regression pairs,
    `lo`/`hi_idx` broadcast against each other: lo = 0 gives the
    single-recursion path of SSU, and a grid of (lo, hi) pairs gives that of
    GSSU. exuber's R/ssu_test.R derives the bilinear cross-moment expansion
    that this function implements."""
    hi_idx = np.asarray(hi_idx)
    lo = np.asarray(lo)

    def s(name: str) -> np.ndarray:
        return ps[name][hi_idx] - ps[name][lo]

    length = (hi_idx - lo).astype(float)

    sx1 = s("x1")
    sx1x1 = s("x1_2")
    sd1 = s("d1")
    sd1x1 = s("d1x1")
    sd1d1 = s("d1_2")
    delta_hat = (length * sd1x1 - sx1 * sd1) / (length * sx1x1 - sx1**2)
    mu1_hat = (sd1 - delta_hat * sx1) / length
    ssr6 = sd1d1 - mu1_hat * sd1 - delta_hat * sd1x1
    sigma2_eps = ssr6 / (length - 2)

    sx2 = sx1x1
    sx2x2 = s("x1_4")
    sd2 = sd1d1
    sd2x2 = s("d1_2x1_2")
    sd2d2 = s("d1_4")
    omega_hat = (length * sd2x2 - sx2 * sd2) / (length * sx2x2 - sx2**2)
    mu2_hat = (sd2 - omega_hat * sx2) / length
    ssr7 = sd2d2 - mu2_hat * sd2 - omega_hat * sd2x2
    sigma2_eta = ssr7 / (length - 2)
    sx2x2_c = sx2x2 - sx2**2 / length
    t_omega = omega_hat / np.sqrt(sigma2_eta / sx2x2_c)

    sd1d2 = s("d1_3")
    sd1x2 = s("d1x1_2")
    sx1x2 = s("x1_3")
    sx1d2 = s("x1d1_2")
    sum_eh = (
        sd1d2 - mu2_hat * sd1 - omega_hat * sd1x2 - mu1_hat * sd2
        + length * mu1_hat * mu2_hat + mu1_hat * omega_hat * sx2
        - delta_hat * sx1d2 + delta_hat * mu2_hat * sx1 + delta_hat * omega_hat * sx1x2
    )
    # same 1/(floor(T r2) - floor(T r1) - 1) as both variances (page 6)
    sigma2_epseta = sum_eh / (length - 2)

    sigma_eps = np.sqrt(sigma2_eps)
    sigma_eta = np.sqrt(sigma2_eta)
    psi_hat = sigma2_epseta / (sigma_eps * sigma_eta)

    ybar2 = sx2 / length
    num_corr = sd1x2 - ybar2 * sd1
    den_corr = np.sqrt(sx2x2_c)

    correction = (psi_hat / sigma_eps) * num_corr / den_corr
    return (t_omega - correction) / np.sqrt(1 - psi_hat**2)


def gssu_stat_path(ps: dict, hi_idx: np.ndarray, minw: int) -> np.ndarray:
    """GSSU's recursive path: for each end point hi, the sup over window
    starts lo = 0, ..., hi - minw (bsadf's shape); its max is GSSU."""
    return np.array([ssu_stat_path(ps, hi, np.arange(hi - minw + 1)).max() for hi in hi_idx])


def gssu_minw(n: int) -> int:
    """Kurozumi & Nishi's GSSU minimum window, r0 = -0.004 + 2.24/sqrt(T)
    (psy_minw()'s formula oversizes GSSU)."""
    return int(np.floor(n * (-0.004 + 2.24 / np.sqrt(n))))


@dataclass
class SsuTestResult:
    """Output of `ssu_test()`: the recursive SSU statistic path, its
    published critical value, and the sup-statistic/detection outcome per
    series."""

    stat: np.ndarray  # (n - minw, nc)
    sadf: np.ndarray  # (nc,)
    crit: float
    detected: np.ndarray  # bool, (nc,)
    minw: int
    n: int
    sig_lvl: float
    series_names: list[str] | None = None
    type: str = "ssu"
    adf_stat: np.ndarray | None = None  # SADF/GSADF, union only
    union_stat: np.ndarray | None = None
    union_crit: float | None = None
    union_detected: np.ndarray | None = None


def ssu_test(
    data,
    minw: int | None = None,
    sig_lvl: float = 95,
    type: str = "ssu",
    union: bool = False,
    cv=None,
) -> SsuTestResult:
    """Stochastic unit root bubble tests, Kurozumi & Nishi (2025): test for
    a *stochastic* (rather than deterministic) unit root in the squared
    first differences, bias-corrected against its dependence on the
    correlation with the plain ADF regression's innovations.

    type="ssu" is the single recursion (SADF's shape, minw = psy_minw(n));
    type="gssu" also takes the supremum over window starts, which gives the
    shape of GSADF, with minw = floor(n * (-0.004 + 2.24/sqrt(n))) as in the
    paper. union=True adds the union of rejections of the paper,
    UR = max(SADF/cv_sadf, SSU/cv_ssu) (GUR with GSADF and GSSU), which is
    compared with the published ur/gur constant. The SADF/GSADF side is
    radf(data, lag=0) against `cv` (a RadfCv, by default radf_crit(n)), and it
    needs the compiled `_core` extension. All critical values are the
    published asymptotic constants of Table I.
    """
    if type not in ("ssu", "gssu"):
        raise ValueError("type must be 'ssu' or 'gssu'")
    x, columns = _to_2d_array(data)
    n, nc = x.shape
    if minw is None:
        minw = psy_minw(n) if type == "ssu" else gssu_minw(n)
    crit = ssu_q(sig_lvl, type)

    hi_idx = np.arange(minw, n)  # R: minw:(n - 1L), 1-indexed -> same values here
    stat = np.empty((len(hi_idx), nc))
    for j in range(nc):
        ps = ssu_prefix_sums(x[:, j])
        if type == "ssu":
            stat[:, j] = ssu_stat_path(ps, hi_idx)
        else:
            stat[:, j] = gssu_stat_path(ps, hi_idx, minw)

    sadf = stat.max(axis=0)
    detected = sadf > crit
    res = SsuTestResult(
        stat=stat, sadf=sadf, crit=crit, detected=detected,
        minw=minw, n=n, sig_lvl=sig_lvl, series_names=columns, type=type,
    )

    if union:
        from exuber.crit import radf_crit
        from exuber.radf import radf

        r = radf(x, lag=0)
        if cv is None:
            cv = radf_crit(n)
        if cv is None:
            raise ValueError(f"no precomputed critical values for n = {n}; pass `cv`")
        k = _SSU_LEVELS.index(int(round(sig_lvl)))
        adf_stat = np.asarray(r.sadf if type == "ssu" else r.gsadf, dtype=float).reshape(nc)
        cv_adf = np.ravel(cv.sadf_cv if type == "ssu" else cv.gsadf_cv)[k]
        res.adf_stat = adf_stat
        res.union_stat = np.maximum(adf_stat / cv_adf, sadf / crit)
        res.union_crit = ssu_q(sig_lvl, "ur" if type == "ssu" else "gur")
        res.union_detected = res.union_stat > res.union_crit
    return res
