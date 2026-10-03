"""Bubble and DGP simulators. Ports of exuber's R/sim.R.

Note on reproducibility. R's rnorm() and rbinom() use R's own random number
algorithm. These functions use numpy's Generator (PCG64), so a given `seed`
does not reproduce the draws of the R functions with the same names. Only the
algorithm is ported and not the bit stream. See the RNG design note of
exubercore.
"""

import math
from collections.abc import Sequence

import numpy as np


def sim_psy1(
    n: int,
    te: float | None = None,
    tf: float | None = None,
    c: float = 1.0,
    alpha: float = 0.6,
    sigma: float = 6.79,
    seed: int | None = None,
    e: np.ndarray | None = None,
    shifts: dict[str, Sequence[float]] | None = None,
    coef_noise: np.ndarray | None = None,
    coef_a: float = 1.0,
) -> np.ndarray:
    """Single-bubble process: martingale -> mildly explosive -> martingale.

    `e`: optional length-(n-1) innovations replacing i.i.d. N(0, sigma^2)
    (e.g. sim_vol_break(n - 1), sim_vol_garch(n - 1), sim_innov(n - 1)).
    `shifts`: optional {"date": [...], "size": [...]} one-period
    deterministic level shifts, dates in 2..n (1-indexed, like R).
    `coef_noise`/`coef_a`: optional length-(n-1) perturbation of the
    explosive-regime coefficient: delta + coef_a * coef_noise[t] / sqrt(n).
    """
    te = te if te is not None else 0.4 * n
    tf = tf if tf is not None else 0.15 * n + te
    rng = np.random.default_rng(seed)
    delta = 1 + c * n ** (-alpha)

    if e is not None and len(e) != n - 1:
        raise ValueError("e must have length n - 1")
    if coef_noise is not None and len(coef_noise) != n - 1:
        raise ValueError("coef_noise must have length n - 1")
    eps = np.asarray(e, dtype=float) if e is not None else rng.normal(scale=sigma, size=n - 1)

    shift_at = np.zeros(n)
    if shifts is not None:
        for date, size in zip(shifts["date"], shifts["size"], strict=True):
            shift_at[int(date) - 1] += size

    y = np.empty(n)
    y[0] = 100.0
    for t in range(2, n + 1):
        i = t - 1
        if coef_noise is not None and te <= t <= tf:
            delta_t = delta + coef_a * coef_noise[i - 1] / np.sqrt(n)
        else:
            delta_t = delta
        if t < te:
            y[i] = y[i - 1] + eps[i - 1] + shift_at[i]
        elif te <= t <= tf:
            y[i] = delta_t * y[i - 1] + eps[i - 1] + shift_at[i]
        elif t == tf + 1:
            y[i] = y[int(te) - 1] + eps[i - 1] + shift_at[i]
        else:
            y[i] = y[i - 1] + eps[i - 1] + shift_at[i]
    return y


def sim_psy2(
    n: int,
    te1: float | None = None,
    tf1: float | None = None,
    te2: float | None = None,
    tf2: float | None = None,
    c: float = 1.0,
    alpha: float = 0.6,
    sigma: float = 6.79,
    seed: int | None = None,
) -> np.ndarray:
    """Two-bubble process: two episodes of mildly explosive dynamics."""
    te1 = te1 if te1 is not None else 0.2 * n
    tf1 = tf1 if tf1 is not None else 0.2 * n + te1
    te2 = te2 if te2 is not None else 0.6 * n
    tf2 = tf2 if tf2 is not None else 0.1 * n + te2
    rng = np.random.default_rng(seed)
    delta = 1 + c * n ** (-alpha)

    y = np.empty(n)
    y[0] = 100.0
    for t in range(2, n + 1):
        i = t - 1
        if t < te1:
            y[i] = y[i - 1] + rng.normal(scale=sigma)
        elif te1 <= t <= tf1:
            y[i] = delta * y[i - 1] + rng.normal(scale=sigma)
        elif t == tf1 + 1:
            y[i] = y[int(te1) - 1] + rng.normal(scale=sigma)
        elif tf1 + 1 < t < te2:
            y[i] = y[i - 1] + rng.normal(scale=sigma)
        elif te2 <= t <= tf2:
            y[i] = delta * y[i - 1] + rng.normal(scale=sigma)
        elif t == tf2 + 1:
            y[i] = y[int(te2) - 1] + rng.normal(scale=sigma)
        else:
            y[i] = y[i - 1] + rng.normal(scale=sigma)
    return y


def sim_ps1(
    n: int,
    te: float | None = None,
    tf: float | None = None,
    tr: float | None = None,
    c: float = 1.0,
    c1: float = 1.0,
    c2: float = 1.0,
    eta: float = 0.6,
    alpha: float = 0.6,
    beta: float = 0.5,
    sigma: float = 6.79,
    seed: int | None = None,
) -> np.ndarray:
    """Single-bubble process with an explicit collapse regime (Phillips & Shi 2018)."""
    te = te if te is not None else 0.4 * n
    tf = tf if tf is not None else te + 0.2 * n
    tr = tr if tr is not None else tf + 0.1 * n
    rng = np.random.default_rng(seed)
    drift = c * n ** (-eta)
    delta = 1 + c1 * n ** (-alpha)
    gamma = 1 - c2 * n ** (-beta)

    y = np.empty(n)
    y[0] = 100.0
    for t in range(2, n + 1):
        i = t - 1
        if t < te:
            y[i] = drift + y[i - 1] + rng.normal(scale=sigma)
        elif te <= t <= tf:
            y[i] = delta * y[i - 1] + rng.normal(scale=sigma)
        elif tf < t <= tr:
            y[i] = gamma * y[i - 1] + rng.normal(scale=sigma)
        else:
            y[i] = drift + y[i - 1] + rng.normal(scale=sigma)
    return y


def sim_ps2(
    n: int,
    te1: float | None = None,
    tf1: float | None = None,
    tr1: float | None = None,
    te2: float | None = None,
    tf2: float | None = None,
    tr2: float | None = None,
    c: float = 1.0,
    c1: float = 1.0,
    c2: float = 1.0,
    eta: float = 0.6,
    alpha: float = 0.6,
    beta: float = 0.5,
    sigma: float = 6.79,
    seed: int | None = None,
) -> np.ndarray:
    """Two-bubble process with explicit collapse regimes (Phillips & Shi 2018)."""
    te1 = te1 if te1 is not None else 0.2 * n
    tf1 = tf1 if tf1 is not None else te1 + 0.2 * n
    tr1 = tr1 if tr1 is not None else tf1 + 0.1 * n
    te2 = te2 if te2 is not None else 0.6 * n
    tf2 = tf2 if tf2 is not None else te2 + 0.15 * n
    tr2 = tr2 if tr2 is not None else tf2 + 0.1 * n
    rng = np.random.default_rng(seed)
    drift = c * n ** (-eta)
    delta = 1 + c1 * n ** (-alpha)
    gamma = 1 - c2 * n ** (-beta)

    y = np.empty(n)
    y[0] = 100.0
    for t in range(2, n + 1):
        i = t - 1
        if t < te1:
            y[i] = drift + y[i - 1] + rng.normal(scale=sigma)
        elif te1 <= t <= tf1:
            y[i] = delta * y[i - 1] + rng.normal(scale=sigma)
        elif tf1 < t <= tr1:
            y[i] = gamma * y[i - 1] + rng.normal(scale=sigma)
        elif tr1 + 1 < t < te2:
            y[i] = drift + y[i - 1] + rng.normal(scale=sigma)
        elif te2 + 1 <= t <= tf2:
            y[i] = delta * y[i - 1] + rng.normal(scale=sigma)
        elif tf2 + 1 < t <= tr2:
            y[i] = gamma * y[i - 1] + rng.normal(scale=sigma)
        else:
            y[i] = drift + y[i - 1] + rng.normal(scale=sigma)
    return y


def sim_blan(
    n: int,
    pi: float = 0.7,
    sigma: float = 0.03,
    r: float = 0.05,
    b0: float = 0.1,
    type: str = "blanchard",
    delta: float = 0.984,
    rw_sigma: float = 0.05,
    seed: int | None = None,
) -> np.ndarray:
    """Blanchard (1979) rational bubble process. `type="rotermann_wilfling"`
    switches to Rotermann & Wilfling's multiplicative-lognormal variant
    (`delta`, `rw_sigma` are only used by that variant)."""
    if type not in ("blanchard", "rotermann_wilfling"):
        raise ValueError('type must be "blanchard" or "rotermann_wilfling"')
    rng = np.random.default_rng(seed)
    b = np.empty(n)
    b[0] = b0

    if type == "blanchard":
        theta = rng.binomial(1, pi, size=n)
        i = 0
        while i < n - 1:
            if b[i] > 0:
                if theta[i] == 1:
                    b[i + 1] = (1 + r) / pi * b[i] + rng.normal(scale=sigma)
                else:
                    b[i + 1] = rng.normal(scale=sigma)
                i += 1
            else:
                i -= 1
    else:
        if not (0 <= delta <= 1):
            raise ValueError("delta must be in [0, 1]")
        if not (rw_sigma > 0):
            raise ValueError("rw_sigma must be > 0")
        theta = rng.binomial(1, pi, size=n - 1)
        u = rng.lognormal(mean=-(rw_sigma**2) / 2, sigma=rw_sigma, size=n - 1)
        for i in range(n - 1):
            if theta[i] == 1:
                b[i + 1] = b[i] * u[i] / delta
            else:
                b[i + 1] = (1 - pi * delta) / (1 - pi) * b[i] * u[i]
    return b


def sim_evans(
    n: int,
    alpha: float = 1.0,
    delta: float = 0.5,
    tau: float = 0.05,
    pi: float = 0.7,
    r: float = 0.05,
    b1: float | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Evans (1991) periodically collapsing rational bubble process."""
    if not (0 < delta < (1 + r) * alpha):
        raise ValueError("alpha and delta should satisfy: 0 < delta < (1+r)*alpha")
    b1 = b1 if b1 is not None else delta

    rng = np.random.default_rng(seed)
    y = rng.normal(0, tau, size=n)
    u = np.exp(y - tau**2 / 2)
    theta = rng.binomial(1, pi, size=n)

    b = np.empty(n)
    b[0] = b1
    for i in range(n - 1):
        if b[i] <= alpha:
            b[i + 1] = (1 + r) * b[i] * u[i + 1]
        else:
            drift = pi ** (-1) * (1 + r) * theta[i + 1] * (b[i] - (1 + r) ** (-1) * delta)
            b[i + 1] = (delta + drift) * u[i + 1]
    return b


def sim_div(
    n: int,
    mu: float | None = None,
    sigma: float | None = None,
    r: float = 0.05,
    log: bool = False,
    output: str = "pf",
    seed: int | None = None,
) -> np.ndarray:
    """Simulate (log) dividends from a random walk with drift (West 1988)."""
    initval = 1.3
    if mu is None:
        mu = 0.013 if log else 0.0373
    if sigma is None:
        sigma = np.sqrt(0.16) if log else np.sqrt(0.1574)
    if output not in ("pf", "d"):
        raise ValueError("output must be 'pf' or 'd'")

    rng = np.random.default_rng(seed)
    # R: x <- mu + c(initval, rnorm(n-1, 0, sigma)); mu is added elementwise,
    # including to initval. filter(x, c(1), init=1.3, method="recursive"):
    # d[0] = x[0] + init, d[t] = x[t] + d[t-1].
    x = np.concatenate(([mu + initval], mu + rng.normal(0, sigma, size=n - 1)))
    init = 1.3
    d = np.empty(n)
    d[0] = x[0] + init
    for t in range(1, n):
        d[t] = x[t] + d[t - 1]

    if log:
        g = np.exp(mu + sigma**2 / 2) - 1
        pf = (1 + g) * d / (r - g)
    else:
        pf = mu * (1 + r) * r ** (-2) + d / r

    return pf if output == "pf" else d


# -- Innovation generators (for sim_psy1(..., e=...) etc.) ------------------


def _beta_fn(a: float, b: float) -> float:
    return math.gamma(a) * math.gamma(b) / math.gamma(a + b)


def sim_innov(
    n: int,
    dist: str = "normal",
    sigma: float = 6.79,
    df: float = 5,
    xi: float = 0.0,
    seed: int | None = None,
) -> np.ndarray:
    """Heavy-tailed/skewed innovations for sim_psy1(..., e=sim_innov(...)):
    `dist="t"` rescales Student-t(df) to unit variance; `dist="skew_t"`
    combines two independent standardized t draws Azzalini-style
    (`xi` skews right if positive, left if negative, 0 = symmetric `t`)."""
    if dist not in ("normal", "t", "skew_t"):
        raise ValueError('dist must be "normal", "t", or "skew_t"')
    if not (sigma >= 0):
        raise ValueError("sigma must be >= 0")
    if not (df > 2):
        raise ValueError("df must be > 2")
    rng = np.random.default_rng(seed)

    if dist == "normal":
        z = rng.standard_normal(n)
    else:
        t0 = rng.standard_t(df, size=n) / np.sqrt(df / (df - 2))
        if dist == "t":
            z = t0
        else:
            t1 = rng.standard_t(df, size=n) / np.sqrt(df / (df - 2))
            delta = xi / np.sqrt(1 + xi**2)
            raw = delta * np.abs(t0) + np.sqrt(1 - delta**2) * t1
            e_abs_t0 = (2 * np.sqrt(df) / ((df - 1) * _beta_fn(df / 2, 0.5))) / np.sqrt(
                df / (df - 2)
            )
            mean_raw = delta * e_abs_t0
            sd_raw = np.sqrt(max(1 - delta**2 * e_abs_t0**2, np.finfo(float).eps))
            z = (raw - mean_raw) / sd_raw

    return z * sigma


def sim_vol_garch(
    n: int,
    omega: float = 0.1,
    alpha: float = 0.1,
    beta: float = 0.8,
    gamma: float = 0.0,
    seed: int | None = None,
) -> np.ndarray:
    """GARCH(1,1)/TGARCH(1,1) innovations z_t = sqrt(h_t) * eps_t, h_t =
    omega + alpha*z[t-1]^2 + beta*h[t-1] + gamma*z[t-1]^2*1{z[t-1]<0}, for
    sim_psy1(..., e=sim_vol_garch(...)). `gamma=0` (default) is plain
    GARCH(1,1); `gamma>0` adds the TGARCH leverage term."""
    if not (omega > 0 and alpha >= 0 and beta >= 0 and gamma >= 0):
        raise ValueError("need omega > 0, alpha/beta/gamma >= 0")
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n)
    h = np.empty(n)
    z = np.empty(n)
    h_prev = 0.0
    z_prev = 0.0
    for t in range(n):
        h[t] = omega + alpha * z_prev**2 + beta * h_prev + gamma * z_prev**2 * (z_prev < 0)
        z[t] = np.sqrt(h[t]) * eps[t]
        h_prev, z_prev = h[t], z[t]
    return z


def sim_vol_break(
    n: int, tau: float = 0.5, ratio: float = 3.0, sigma: float = 6.79, seed: int | None = None
) -> np.ndarray:
    """i.i.d. Gaussian shocks whose std shifts permanently from `sigma` to
    `sigma * ratio` at observation floor(tau * n), for
    sim_psy1(..., e=sim_vol_break(...)). This is the DGP with non-stationary
    volatility that the volatility-robust tests (radf_wb_cv and others)
    target."""
    if not (0 <= tau <= 1):
        raise ValueError("tau must be in [0, 1]")
    if not (ratio > 0 and sigma >= 0):
        raise ValueError("need ratio > 0, sigma >= 0")
    rng = np.random.default_rng(seed)
    idx = np.arange(1, n + 1)
    sd_t = sigma * np.where(idx > np.floor(tau * n), ratio, 1.0)
    return rng.normal(scale=sd_t)


def sim_vol_cir(
    n: int,
    kappa: float = 0.03,
    theta: float = 0.25,
    xi: float = 0.1,
    sigma0_sq: float | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """CIR (square-root) stochastic-variance shocks, Euler-Maruyama
    discretized: d(sigma^2) = kappa*(theta - sigma^2)dt + xi*sigma*dB, for
    sim_psy1(..., e=sim_vol_cir(...))."""
    sigma0_sq = theta if sigma0_sq is None else sigma0_sq
    if not (kappa > 0 and theta > 0 and xi > 0 and sigma0_sq >= 0):
        raise ValueError("need kappa/theta/xi > 0, sigma0_sq >= 0")
    rng = np.random.default_rng(seed)
    dt = 1.0 / n
    sig2 = np.empty(n)
    sig2[0] = sigma0_sq
    if n > 1:
        db = rng.normal(scale=np.sqrt(dt), size=n - 1)
        for i in range(1, n):
            prev = max(sig2[i - 1], 0.0)
            sig2[i] = max(prev + kappa * (theta - prev) * dt + xi * np.sqrt(prev) * db[i - 1], 0.0)
    return np.sqrt(sig2) * rng.normal(size=n)


def sim_vol_sv(
    n: int, phi: float = 0.98, tau: float = 0.1, log_sigma0_sq: float = 0.0, seed: int | None = None
) -> np.ndarray:
    """AR(1) lognormal stochastic-volatility shocks z_t = sigma_t * eps_t,
    log(sigma_t^2) = phi*log(sigma[t-1]^2) + eta_t, eta_t ~ iid N(0, tau^2),
    for sim_psy1(..., e=sim_vol_sv(...))."""
    if not (0 <= phi <= 1):
        raise ValueError("phi must be in [0, 1]")
    if not (tau > 0):
        raise ValueError("tau must be > 0")
    rng = np.random.default_rng(seed)
    log_sig2 = np.empty(n)
    log_sig2[0] = log_sigma0_sq
    if n > 1:
        eta = rng.normal(scale=tau, size=n - 1)
        for i in range(1, n):
            log_sig2[i] = phi * log_sig2[i - 1] + eta[i - 1]
    return np.exp(log_sig2 / 2) * rng.normal(size=n)


def sim_fi(n: int, d: float = 0.2, sigma: float = 1.0, seed: int | None = None) -> np.ndarray:
    """Fractionally-integrated (long-memory) innovations u_t =
    Delta^(-d) * eps_t via a truncated MA(inf) expansion (truncated at
    max(500, 5n) lags with a matching burn-in), for
    sim_psy1(..., e=sim_fi(...))."""
    if not (0 < d < 0.5 and sigma > 0):
        raise ValueError("need 0 < d < 0.5, sigma > 0")
    rng = np.random.default_rng(seed)
    m = max(500, 5 * n)
    eps = rng.normal(scale=sigma, size=n + m)
    psi = np.empty(m + 1)
    psi[0] = 1.0
    for j in range(1, m + 1):
        psi[j] = psi[j - 1] * (j - 1 + d) / j
    u = np.empty(n)
    for k in range(n):
        window = eps[k : k + m + 1]
        u[k] = np.dot(psi, window[::-1])
    return u


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1 + np.vectorize(math.erf)(x / math.sqrt(2)))


def sim_tree(
    n: int,
    a: float = 0.95,
    eta: float = 1.0,
    mu: float = -1.0,
    rho: float = 0.7,
    sigma: float = 4.0,
    y0: float | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Gourieroux & Jasiak (2025) stochastic branching-tree bubble: a binomial
    tree whose branching intensity p_t = Phi(X_t) follows a latent Gaussian
    AR(1). Price floor eta/(1-a); no finite mean, so occasional huge values
    are expected. Defaults reproduce the source's Figure 2 example."""
    if not 0 < a < 1:
        raise ValueError("a must be in (0, 1)")
    if eta <= 0 or sigma <= 0:
        raise ValueError("eta and sigma must be positive")
    if not -1 < rho < 1:
        raise ValueError("rho must be in (-1, 1)")
    rng = np.random.default_rng(seed)
    y0 = eta / (1 - a) if y0 is None else y0

    x = np.empty(n)
    x[0] = mu
    u = rng.normal(size=n - 1)
    for t in range(1, n):
        x[t] = mu + rho * (x[t - 1] - mu) + sigma * math.sqrt(1 - rho**2) * u[t - 1]
    # clip away from exact 0/1: 0/0 would propagate NaN through every later y
    p = np.clip(_norm_cdf(x), 1e-10, 1 - 1e-10)
    z = rng.binomial(1, p)
    xi1 = z / (a * p)
    eps = (eta / (1 - a)) * (1 - xi1) + (eta / a) * (1 - z) / (1 - p)

    y = np.empty(n)
    y[0] = y0
    for t in range(1, n):
        y[t] = xi1[t] * y[t - 1] + eps[t]
    return y


def sim_mar(
    n: int,
    phi1: float = 0.7,
    psi1: float = 0.7,
    dist: str = "cauchy",
    df: float = 2,
    burn: int = 100,
    seed: int | None = None,
) -> np.ndarray:
    """Blasques, Koopman, Mingoli & Telg (2025) mixed causal-noncausal
    AR(1,1): (1 - phi1 L)(1 - psi1 L^-1) y_t = eps_t. The noncausal part is
    run backward from a zero boundary `burn` steps past the end, the causal
    part forward from `burn` steps before the start; both burn-ins dropped."""
    if dist not in ("cauchy", "t"):
        raise ValueError('dist must be "cauchy" or "t"')
    if not (0 < phi1 < 1 and 0 < psi1 < 1):
        raise ValueError("phi1 and psi1 must be in (0, 1)")
    if burn < 0:
        raise ValueError("burn must be non-negative")
    rng = np.random.default_rng(seed)
    m = n + 2 * burn
    eps = rng.standard_cauchy(m) if dist == "cauchy" else rng.standard_t(df, m)

    u = np.zeros(m + 1)
    for t in range(m - 1, -1, -1):
        u[t] = psi1 * u[t + 1] + eps[t]
    y = np.empty(m)
    y[0] = u[0]
    for t in range(1, m):
        y[t] = phi1 * y[t - 1] + u[t]
    return y[burn : burn + n]


def sim_common(
    n_series: int,
    n: int,
    te: float | None = None,
    tf: float | None = None,
    c: float = 1.0,
    alpha: float = 0.6,
    sigma: float = 6.79,
    sigma_e: float = 0.1,
    seed: int | None = None,
    return_factor: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Chen, Phillips & Shi (2023) common bubble: `n_series` observed series
    X_t = Lambda f_t + e_t driven by one latent sim_psy1() bubble factor,
    loadings Lambda ~ U[0, 2]. Returns an (n, n_series) array; with
    `return_factor=True`, also the latent factor (R's "factor" attribute)."""
    if sigma_e < 0:
        raise ValueError("sigma_e must be non-negative")
    rng = np.random.default_rng(seed)
    f = sim_psy1(n, te=te, tf=tf, c=c, alpha=alpha, sigma=sigma, seed=int(rng.integers(2**62)))
    loadings = rng.uniform(0, 2, size=n_series)
    x = np.outer(f, loadings) + rng.normal(scale=sigma_e, size=(n, n_series))
    return (x, f) if return_factor else x


def sim_coexplosive(
    n: int,
    lag: int = 0,
    phi_x: float = 1.0,
    phi_z: float = 0.0,
    mu_y: float = 0.0,
    sigma_y: float = 6.79,
    x_args: dict | None = None,
    z_args: dict | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Evripidou, Harvey, Leybourne & Sollis (2022) co-explosive pair:
    x from sim_psy1(), y_t = mu_y + phi_x x_{t-lag} + phi_z z_t + eps_t
    (z an independent sim_psy1() bubble, only drawn if phi_z != 0).
    lag > 0: x's episode leads y's. Returns an (n, 2) array [x, y]; y is
    NaN where x_{t-lag} falls outside the sample."""
    if sigma_y < 0:
        raise ValueError("sigma_y must be non-negative")
    rng = np.random.default_rng(seed)
    x = sim_psy1(n, seed=int(rng.integers(2**62)), **(x_args or {}))
    if phi_z != 0:
        z = sim_psy1(n, seed=int(rng.integers(2**62)), **(z_args or {}))
    else:
        z = np.zeros(n)
    idx = np.arange(n) - lag
    valid = (idx >= 0) & (idx < n)
    x_lag = np.full(n, np.nan)
    x_lag[valid] = x[idx[valid]]
    y = mu_y + phi_x * x_lag + phi_z * z + rng.normal(scale=sigma_y, size=n)
    return np.column_stack([x, y])


def sim_msbubble(
    n: int,
    p11: float = 0.98,
    p22: float = 0.90,
    lambda1: float = 0.98,
    lambda2: float = 1.03,
    sigma_b: float = 0.05,
    b0: float = 0.0,
    s0: int = 1,
    seed: int | None = None,
    return_regime: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Chan & Santi (2021) Markov-switching bubble: b_t = b_{t-1}/lambda_{S_t}
    + eps_t, S_t a two-state Markov chain (1 = surviving/explosive via
    lambda1 < 1, 2 = collapsing via lambda2 > 1). Uses the contemporaneous
    S_t, as R does (the source indexes by S_{t+1}). With
    `return_regime=True`, also the regime path (R's "regime" attribute).
    At the defaults the process is net-explosive (~1.2% average growth per
    step), so it overflows float64 after roughly 60,000 observations."""
    if not (0 < p11 < 1 and 0 < p22 < 1):
        raise ValueError("p11 and p22 must be in (0, 1)")
    if sigma_b <= 0 or s0 not in (1, 2):
        raise ValueError("sigma_b must be positive and s0 one of 1, 2")
    rng = np.random.default_rng(seed)
    s = np.empty(n, dtype=int)
    s[0] = s0
    unif = rng.uniform(size=n - 1)
    for t in range(1, n):
        stay = p11 if s[t - 1] == 1 else p22
        s[t] = s[t - 1] if unif[t - 1] < stay else 3 - s[t - 1]
    lam = np.where(s == 1, lambda1, lambda2)

    b = np.empty(n)
    b[0] = b0
    eps = rng.normal(scale=sigma_b, size=n - 1)
    for t in range(1, n):
        b[t] = b[t - 1] / lam[t] + eps[t - 1]
    return (b, s) if return_regime else b


def sim_falsebubble(
    n: int,
    t1: int | None = None,
    t2: int | None = None,
    kappa: int | None = None,
    shape: str = "triangular",
    amplitude: float = 1.0,
    mu: float = 0.02,
    sigma_d: float = 0.05,
    r: float = 0.05,
    d0: float = 0.0,
    seed: int | None = None,
    return_components: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chen, Chen, Huang, Li & Zhang (2026) false bubble: a deterministic
    hump-shaped technology shock in dividend growth makes a present-value
    fundamental look locally explosive, although there is no bubble at all. It
    serves as a stress test in which no bubble is present. t1, t2 and kappa
    are 1-indexed dates, as in R. With `return_components=True`, it returns
    (price, dividend, technology)."""
    t1 = math.floor(0.3 * n) if t1 is None else t1
    t2 = math.floor(0.7 * n) if t2 is None else t2
    kappa = math.floor((t2 - t1) / 2) if kappa is None else kappa
    if shape not in ("triangular", "gaussian"):
        raise ValueError('shape must be "triangular" or "gaussian"')
    if not (1 <= t1 <= n and t1 <= t2 <= n):
        raise ValueError("need 1 <= t1 <= t2 <= n")
    if not (0 < kappa < t2 - t1) or amplitude < 0 or r <= 0 or sigma_d <= 0:
        raise ValueError("need 0 < kappa < t2 - t1, amplitude >= 0, r > 0, sigma_d > 0")
    rng = np.random.default_rng(seed)

    tt = np.arange(1, n + 1)
    if shape == "triangular":
        tau = np.zeros(n)
        up = (tt >= t1) & (tt <= t1 + kappa)
        down = (tt > t1 + kappa) & (tt <= t2)
        tau[up] = (tt[up] - t1) / kappa
        tau[down] = (t2 - tt[down]) / (t2 - t1 - kappa)
    else:
        tau = np.exp(-0.5 * ((tt - (t1 + kappa)) / ((t2 - t1) / 4)) ** 2)
        tau[(tt < t1) | (tt > t2)] = 0
    tau = amplitude * tau

    d = np.empty(n)
    d[0] = d0
    eta = rng.normal(scale=sigma_d, size=n - 1)
    for t in range(1, n):
        d[t] = d[t - 1] + mu + tau[t] + eta[t - 1]

    # T_t = sum_{s>t} beta^(s-t) tau_s: the known-in-advance hump, discounted
    beta = 1 / (1 + r)
    bigt = np.array([np.sum(beta ** np.arange(1, n - t) * tau[t + 1 :]) for t in range(n)])
    p = mu * (1 + r) * r ** (-2) + d / r + bigt / r
    return (p, d, tau) if return_components else p
