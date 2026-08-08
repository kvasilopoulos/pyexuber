"""Bubble/DGP simulators. Ports of exuber's R/sim.R.

Note on reproducibility: R's rnorm()/rbinom() use R's own RNG algorithm.
These use numpy's Generator (PCG64), so a given `seed` will not reproduce
the same draws as the R functions of the same name -- only the algorithm
is ported, not the bit-stream. See exubercore's RNG design note.
"""

import numpy as np


def sim_psy1(
    n: int,
    te: float | None = None,
    tf: float | None = None,
    c: float = 1.0,
    alpha: float = 0.6,
    sigma: float = 6.79,
    seed: int | None = None,
) -> np.ndarray:
    """Single-bubble process: martingale -> mildly explosive -> martingale."""
    te = te if te is not None else 0.4 * n
    tf = tf if tf is not None else 0.15 * n + te
    rng = np.random.default_rng(seed)
    delta = 1 + c * n ** (-alpha)

    y = np.empty(n)
    y[0] = 100.0
    for t in range(2, n + 1):
        i = t - 1
        if t < te:
            y[i] = y[i - 1] + rng.normal(scale=sigma)
        elif te <= t <= tf:
            y[i] = delta * y[i - 1] + rng.normal(scale=sigma)
        elif t == tf + 1:
            y[i] = y[int(te) - 1] + rng.normal(scale=sigma)
        else:
            y[i] = y[i - 1] + rng.normal(scale=sigma)
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
    seed: int | None = None,
) -> np.ndarray:
    """Blanchard (1979) rational bubble process."""
    rng = np.random.default_rng(seed)
    b = np.empty(n)
    b[0] = b0
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
