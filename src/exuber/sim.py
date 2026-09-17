"""Bubble/DGP simulators. Ports of exuber's R/sim.R.

Note on reproducibility: R's rnorm()/rbinom() use R's own RNG algorithm.
These use numpy's Generator (PCG64), so a given `seed` will not reproduce
the same draws as the R functions of the same name -- only the algorithm
is ported, not the bit-stream. See exubercore's RNG design note.
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
    sim_psy1(..., e=sim_vol_break(...)) -- the non-stationary-volatility
    DGP the volatility-robust tests (radf_wb_cv etc.) target."""
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
