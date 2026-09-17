"""quantile_test() -- Wu, Shi & Wu (2025)'s "global test": a quantile-
regression (QR) analogue of the DF t-ratio, testing for a bubble via
the tau-th conditional quantile of y_t on y_{t-1} instead of the
conditional mean. Ported from exuber's R/quantile_test.R. See
docs/alternative-paradigms.md, "Quantile-based detection". A single
static test, not a recursive scan -- compare radf()'s single-shot `adf`
statistic, not its recursive `bsadf`.

QR solver: no simplex/interior-point LP package (statsmodels, scipy) is
a pyexuber dependency, and adding one for a single-predictor fit is more
than this module needs (ponytail: rung 5, prefer an already-installed
dependency; none fits, and the fit itself is genuinely small). Instead,
iteratively reweighted least squares (IRLS, Schlossmacher 1973's
approach to the asymmetric L1/check-function loss) via plain numpy:
converges to the exact QR solution for continuous data with no ties, to
well within the tolerance this module's own tests use.

RNG note: uses numpy's Generator, not R's RNG -- a given `seed` will not
reproduce the same critical-value draws as the R function of the same
name (see sim.py's module docstring). The point statistics themselves
(tstat, delta) are fully deterministic and match R bit-for-bit.
"""

from dataclasses import dataclass

import numpy as np

from exuber._monitor_common import _assert_sig_lvl, _quantile_narm
from exuber.radf import _to_2d_array


def _bw_nrd0(x: np.ndarray) -> float:
    """Port of R's stats::bw.nrd0() (the default density() bandwidth):
    0.9 * min(sd, IQR/1.34) * n^(-1/5)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    sd = np.std(x, ddof=1)
    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25
    hi = min(sd, iqr / 1.34) if iqr > 0 else sd
    return 0.9 * hi * n ** (-0.2) if hi > 0 else 1.0


def _dnorm(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)


def _quantile_check_density(u: np.ndarray, tau: float) -> tuple[float, float]:
    """b_tau (the tau-th sample quantile of u) and f_hat (a kernel density
    estimate of u's density at b_tau, via a Gaussian kernel with
    bw.nrd0's bandwidth) -- needed to studentize the QR coefficient the
    way OLS's residual-variance estimate studentizes the DF t-ratio."""
    b_tau = _quantile_narm(u, tau)
    h = _bw_nrd0(u)
    f_hat = np.mean(_dnorm((b_tau - u) / h)) / h
    return b_tau, f_hat


def _quantile_regression_fit(
    y: np.ndarray, x: np.ndarray, tau: float, max_iter: int = 200, tol: float = 1e-10
) -> tuple[float, float]:
    """Fit y = a + b*x at quantile `tau` via IRLS: repeatedly weighted
    least squares with weights 1/max(eps, |residual|) (asymmetric by
    tau/1-tau on each side), which minimizes the check-function loss at
    convergence. Returns (a, b)."""
    n = len(y)
    design = np.column_stack([np.ones(n), x])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]  # OLS start
    for _ in range(max_iter):
        resid = y - design @ beta
        w = np.where(resid >= 0, tau, 1 - tau) / np.maximum(np.abs(resid), 1e-8)
        wd = design * w[:, None]
        beta_new = np.linalg.solve(design.T @ wd, design.T @ (w * y))
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    return float(beta[0]), float(beta[1])


def _quantile_adf_tstat(y: np.ndarray) -> float:
    """Plain OLS ADF (intercept + one lag, no augmentation) t-statistic --
    used only to simulate quantile_test()'s Q component (the standard,
    demeaned Dickey-Fuller t-statistic distribution, verified in
    docs/alternative-paradigms.md to be bit-for-bit identical to radf()'s
    own single-shot `adf` field)."""
    n = len(y)
    dy = y[1:] - y[:-1]
    ylag = y[:-1]
    design = np.column_stack([np.ones(n - 1), ylag])
    beta, _, _, _ = np.linalg.lstsq(design, dy, rcond=None)
    resid = dy - design @ beta
    dof = (n - 1) - 2
    sigma2 = np.sum(resid**2) / dof
    se = np.sqrt(sigma2 * np.linalg.inv(design.T @ design)[1, 1])
    return float(beta[1] / se)


def _quantile_test_q_distribution(n: int, nrep: int, rng: np.random.Generator) -> np.ndarray:
    return np.array([_quantile_adf_tstat(np.cumsum(rng.normal(size=n))) for _ in range(nrep)])


@dataclass
class QuantileTestResult:
    tstat: np.ndarray  # (nc,)
    tau: np.ndarray  # (nc,), the tau actually used per series
    delta: np.ndarray  # (nc,)
    crit: np.ndarray  # (nc,)
    detected: np.ndarray  # (nc,) bool
    n: int
    sig_lvl: float
    iter: int
    series_names: list[str] | None = None


def quantile_test(
    data,
    tau: float | str = "optimal",
    tau_grid: np.ndarray | None = None,
    nrep: int = 1000,
    sig_lvl: float = 95,
    seed: int | None = None,
) -> QuantileTestResult:
    """Wu, Shi & Wu (2025)'s "global test": a quantile-regression (QR)
    analogue of the Dickey-Fuller t-ratio, testing for explosive behavior
    at a chosen conditional quantile `tau` of y_t on y_{t-1} rather than
    at the conditional mean. A single static test (compare radf()'s
    single-shot `adf`, not its recursive `bsadf`).

    `tau = "optimal"` (default) selects the quantile minimizing the
    asymptotic variance of the QR estimator (their eq. 33) by grid search
    over `tau_grid` (default 0.2 to 0.8 in steps of 0.05, the paper's own
    recommended practical range).

    The critical value is simulated per call: the statistic's limiting
    null distribution is sqrt(1 - delta^2) * z + delta * Q, z ~ N(0, 1),
    delta a data-estimated correlation coefficient, and Q the standard
    demeaned Dickey-Fuller t-statistic distribution (simulated the same
    random-walk-plus-OLS-t-stat way radf_mc_cv() simulates its own `adf`
    critical value).

    RNG note: uses numpy's Generator, not R's -- a given `seed` will not
    reproduce the same critical-value draws as R's quantile_test() (the
    point statistics tstat/tau/delta themselves match bit-for-bit).
    """
    _assert_sig_lvl(sig_lvl)
    if tau_grid is None:
        tau_grid = np.round(np.arange(0.20, 0.80 + 1e-9, 0.05), 10)
    x, columns = _to_2d_array(data)
    n, nc = x.shape
    rng = np.random.default_rng(seed)

    q_dist = _quantile_test_q_distribution(n, nrep, rng)

    tstat = np.empty(nc)
    crit = np.empty(nc)
    delta = np.empty(nc)
    tau_used = np.empty(nc)
    detected = np.empty(nc, dtype=bool)

    for j in range(nc):
        y = x[:, j]
        dy = y[1:] - y[:-1]
        ylag = y[:-1]

        tau_j: float
        if tau == "optimal":
            fhats = np.array([_quantile_check_density(dy, tt)[1] for tt in tau_grid])
            obj = (tau_grid * (1 - tau_grid)) / fhats**2
            tau_j = float(tau_grid[np.argmin(obj)])
        else:
            tau_j = float(tau)
        if not (0 < tau_j < 1):
            raise ValueError("tau must be in (0, 1) or 'optimal'.")

        _, alpha_hat = _quantile_regression_fit(y[1:], ylag, tau_j)

        _, f_hat = _quantile_check_density(dy, tau_j)
        y_pzy = np.sum((ylag - np.mean(ylag)) ** 2)
        tstat_j = (f_hat / np.sqrt(tau_j * (1 - tau_j))) * np.sqrt(y_pzy) * (alpha_hat - 1)

        psi = tau_j - (dy < _quantile_narm(dy, tau_j)).astype(float)
        delta_j = float(np.clip(np.corrcoef(dy, psi)[0, 1], -1, 1))

        z = rng.normal(size=nrep)
        u = np.sqrt(1 - delta_j**2) * z + delta_j * q_dist
        crit_j = _quantile_narm(u, sig_lvl / 100)

        tstat[j] = tstat_j
        crit[j] = crit_j
        delta[j] = delta_j
        tau_used[j] = tau_j
        detected[j] = tstat_j > crit_j

    return QuantileTestResult(
        tstat=tstat,
        tau=tau_used,
        delta=delta,
        crit=crit,
        detected=detected,
        n=n,
        sig_lvl=sig_lvl,
        iter=nrep,
        series_names=columns,
    )
