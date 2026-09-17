"""SSU: stochastic explosive-coefficient test (Kurozumi & Nishi 2025).
Ported from exuber's R/ssu_test.R -- see docs/volatility-robustness.md
(root repo) for the formulas, papers and independent validation this
ports. Standalone: no shared helper module needed (unlike radf_tt/
radf_sign or radf_kp/radf_sbz).
"""

from dataclasses import dataclass

import numpy as np

from exuber.radf import _to_2d_array, psy_minw

_SSU_LEVELS = (90, 95, 99)
_SSU_CRIT = (2.90, 3.30, 4.20)


def ssu_q(sig_lvl: float) -> float:
    """Kurozumi & Nishi (2025) Table I's published SSU asymptotic critical
    value (their own 10,000-rep Monte Carlo) -- no simulation needed."""
    for lvl, crit in zip(_SSU_LEVELS, _SSU_CRIT, strict=True):
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


def ssu_stat_path(ps: dict, hi_idx: np.ndarray) -> np.ndarray:
    """t^{omega,c}_{0,r2} (SSU's own r1 = 0, fixed) for every candidate r2
    in `hi_idx` -- see exuber's R/ssu_test.R for the derivation of the
    bilinear cross-moment expansion this implements."""
    hi_idx = np.asarray(hi_idx)

    def s(name: str) -> np.ndarray:
        return ps[name][hi_idx]

    length = hi_idx.astype(float)

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
    sigma2_epseta = sum_eh / (length - 1)

    sigma_eps = np.sqrt(sigma2_eps)
    sigma_eta = np.sqrt(sigma2_eta)
    psi_hat = sigma2_epseta / (sigma_eps * sigma_eta)

    ybar2 = sx2 / length
    num_corr = sd1x2 - ybar2 * sd1
    den_corr = np.sqrt(sx2x2_c)

    correction = (psi_hat / sigma_eps) * num_corr / den_corr
    return (t_omega - correction) / np.sqrt(1 - psi_hat**2)


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


def ssu_test(data, minw: int | None = None, sig_lvl: float = 95) -> SsuTestResult:
    """Stochastic unit root bubble test (SSU), Kurozumi & Nishi (2025):
    tests for a *stochastic* (rather than deterministic) unit root in the
    squared first differences, bias-corrected against its dependence on
    the correlation with the plain ADF regression's innovations. Only the
    single-recursion SSU statistic is implemented (not GSSU, CUSUM/
    CUSUM-SQ, or the union-of-rejections procedure -- see
    docs/volatility-robustness.md, root repo, for the minimum-viable-
    subset scoping). The critical value is Table I's published constant,
    no simulation needed."""
    x, columns = _to_2d_array(data)
    n, nc = x.shape
    minw = minw if minw is not None else psy_minw(n)
    crit = ssu_q(sig_lvl)

    hi_idx = np.arange(minw, n)  # R: minw:(n - 1L), 1-indexed -> same values here
    stat = np.empty((len(hi_idx), nc))
    for j in range(nc):
        ps = ssu_prefix_sums(x[:, j])
        stat[:, j] = ssu_stat_path(ps, hi_idx)

    sadf = stat.max(axis=0)
    detected = sadf > crit

    return SsuTestResult(
        stat=stat, sadf=sadf, crit=crit, detected=detected,
        minw=minw, n=n, sig_lvl=sig_lvl, series_names=columns,
    )
