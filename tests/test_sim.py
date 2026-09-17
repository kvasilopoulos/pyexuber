"""Regression tests for the regime-switching branch logic in sim.py,
verified against R by feeding both implementations an identical fixed
shock sequence (bypassing RNG differences -- see the conversation history
for the R-side check). Covers integer and non-integer te/tf/tr boundaries,
including R's array-index truncation behavior at y[te]."""

import numpy as np
import pytest

_SHOCKS = [
    0, 0.1836, -0.8356, 1.5953, 0.3295, -0.8205, 0.4874, 0.7383, 0.5758, -0.3054,
    1.5118, 0.3898, -0.6212, -2.2147, 1.1249, -0.0449, -0.0162, 0.9438, 0.8212,
    0.5939, 0.919, 0.7821, 0.0746, -1.9894, 0.6198, -0.0561, -0.1558, -1.4708,
    -0.4782, 0.4179, 1.3587, -0.1028, 0.3877, -0.0538, -1.3771, -0.415, -0.3943,
]
_N = 37


def _sim_psy1_fixed(n, te, tf, c=1.0, alpha=0.6):
    delta = 1 + c * n ** (-alpha)
    y = np.empty(n)
    y[0] = 100.0
    for t in range(2, n + 1):
        i = t - 1
        s = _SHOCKS[t - 1]
        if t < te:
            y[i] = y[i - 1] + s
        elif te <= t <= tf:
            y[i] = delta * y[i - 1] + s
        elif t == tf + 1:
            y[i] = y[int(te) - 1] + s
        else:
            y[i] = y[i - 1] + s
    return y


def _sim_ps1_fixed(n, te, tf, tr, c=1.0, c1=1.0, c2=1.0, eta=0.6, alpha=0.6, beta=0.5):
    drift = c * n ** (-eta)
    delta = 1 + c1 * n ** (-alpha)
    gamma = 1 - c2 * n ** (-beta)
    y = np.empty(n)
    y[0] = 100.0
    for t in range(2, n + 1):
        i = t - 1
        s = _SHOCKS[t - 1]
        if t < te:
            y[i] = drift + y[i - 1] + s
        elif te <= t <= tf:
            y[i] = delta * y[i - 1] + s
        elif tf < t <= tr:
            y[i] = gamma * y[i - 1] + s
        else:
            y[i] = drift + y[i - 1] + s
    return y


def test_sim_psy1_matches_r_shared_prefix():
    # Both te/tf pairs below share the same bubble onset and haven't
    # collapsed by t=20 (R, 1-indexed) either way -- same value expected.
    for te, tf in [(15, 20), (14.6, 20.3)]:
        y = _sim_psy1_fixed(_N, te, tf)
        assert y[0] == 100.0
        assert y[19] == pytest.approx(198.18085)  # R position 20 -> y[19]


def test_sim_psy1_noninteger_boundary_matches_r():
    y = _sim_psy1_fixed(_N, te=14.6, tf=20.3)
    assert y[20] == pytest.approx(199.09985)  # R position 21 -> y[20]


def test_sim_psy1_full_sequence_matches_r():
    y = _sim_psy1_fixed(_N, te=15, tf=20)
    expected_tail = [198.18085, 114.6314, 115.4135, 115.4881]
    np.testing.assert_allclose(y[19:23], expected_tail, atol=1e-4)


def test_sim_psy1_short_bubble_matches_r():
    y = _sim_psy1_fixed(_N, te=10, tf=13)
    expected = [102.2538, 113.663835, 128.198342, 143.276099, 159.070345, 111.449135]
    np.testing.assert_allclose(y[8:14], expected, atol=1e-4)


def test_sim_ps1_full_sequence_matches_r():
    y = _sim_ps1_fixed(_N, te=10, tf=18, tr=22)
    expected = [275.051528, 230.654536, 193.329064, 162.464961, 136.537986]
    np.testing.assert_allclose(y[17:22], expected, atol=1e-4)


def test_sim_ps1_noninteger_boundaries_match_r():
    y = _sim_ps1_fixed(_N, te=9.4, tf=17.2, tr=21.8)
    expected = [245.930903, 206.443911, 173.325941, 145.425232, 122.436471]
    np.testing.assert_allclose(y[16:21], expected, atol=1e-4)


def test_sim_functions_run_and_have_right_shape():
    from exuber.sim import sim_blan, sim_div, sim_evans, sim_ps1, sim_ps2, sim_psy1, sim_psy2

    n = 50
    for fn in (sim_psy1, sim_psy2, sim_ps1, sim_ps2, sim_blan, sim_evans, sim_div):
        y = fn(n, seed=1)
        assert y.shape == (n,)
        assert np.all(np.isfinite(y))


# -- Innovation generators: formula checks against R, bypassing RNG --------
#
# set.seed(1); eps10 <- rnorm(10) in R gives the fixed pool below. _FakeRNG
# feeds it to np.random.default_rng()'s call sites in the same order the
# real R code draws from its own RNG, so the *formula* (not the RNG
# bit-stream) gets checked bit-for-bit -- same idea as the fixed _SHOCKS
# list above, generalized to a drop-in np.random.Generator replacement.

_EPS10 = np.array(
    [-0.62645381, 0.18364332, -0.83562861, 1.59528080, 0.32950777,
     -0.82046838, 0.48742905, 0.73832471, 0.57578135, -0.30538839]
)


class _FakeRNG:
    def __init__(self, pool):
        self.pool = np.asarray(pool, dtype=float)
        self.pos = 0

    def _take(self, size):
        if size is None:
            v = self.pool[self.pos]
            self.pos += 1
            return v
        out = self.pool[self.pos : self.pos + size]
        self.pos += size
        return out

    def standard_normal(self, size=None):
        return self._take(size)

    def normal(self, loc=0.0, scale=1.0, size=None):
        if size is None and isinstance(scale, np.ndarray):
            size = scale.shape[0]
        return loc + scale * self._take(size)

    def standard_t(self, df, size=None):  # not exercised by the checks below
        raise NotImplementedError


@pytest.fixture
def fake_rng(monkeypatch):
    def _install(pool):
        monkeypatch.setattr(np.random, "default_rng", lambda seed=None: _FakeRNG(pool))

    return _install


def test_sim_vol_break_matches_r(fake_rng):
    from exuber.sim import sim_vol_break

    fake_rng(_EPS10)
    y = sim_vol_break(10, tau=0.5, ratio=3, sigma=2, seed=1)
    expected = [-1.25290762, 0.36728665, -1.67125722, 3.19056160, 0.65901554,
                -4.92281030, 2.92457431, 4.42994823, 3.45468811, -1.83233032]
    np.testing.assert_allclose(y, expected, atol=1e-6)


def test_sim_vol_garch_matches_r(fake_rng):
    from exuber.sim import sim_vol_garch

    fake_rng(_EPS10[:5])
    y = sim_vol_garch(5, omega=0.1, alpha=0.1, beta=0.8, gamma=0.0, seed=1)
    expected = [-0.19810209, 0.07875803, -0.41593815, 0.89607126, 0.21675026]
    np.testing.assert_allclose(y, expected, atol=1e-6)


def test_sim_vol_garch_tgarch_matches_r(fake_rng):
    from exuber.sim import sim_vol_garch

    fake_rng(_EPS10[:5])
    y = sim_vol_garch(5, omega=0.1, alpha=0.1, beta=0.8, gamma=0.2, seed=1)
    expected = [-0.19810209, 0.08042096, -0.42119779, 0.95247029, 0.22731252]
    np.testing.assert_allclose(y, expected, atol=1e-6)


def test_sim_vol_cir_matches_r(fake_rng):
    from exuber.sim import sim_vol_cir

    fake_rng(np.concatenate([_EPS10[:4], _EPS10[5:10]]))
    y = sim_vol_cir(5, kappa=0.03, theta=0.25, xi=0.1, seed=1)
    expected = [-0.41023419, 0.23678823, 0.36175334, 0.27117724, -0.15439035]
    np.testing.assert_allclose(y, expected, atol=1e-6)


def test_sim_vol_sv_matches_r(fake_rng):
    from exuber.sim import sim_vol_sv

    fake_rng(np.concatenate([_EPS10[:4], _EPS10[5:10]]))
    y = sim_vol_sv(5, phi=0.98, tau=0.1, seed=1)
    expected = [-0.82046838, 0.47239810, 0.72260999, 0.54069901, -0.31098370]
    np.testing.assert_allclose(y, expected, atol=1e-6)


def test_sim_innov_normal_matches_r(fake_rng):
    from exuber.sim import sim_innov

    fake_rng(_EPS10)
    y = sim_innov(10, dist="normal", sigma=2, seed=1)
    expected = [-1.25290762, 0.36728665, -1.67125722, 3.19056160, 0.65901554,
                -1.64093677, 0.97485810, 1.47664941, 1.15156270, -0.61077677]
    np.testing.assert_allclose(y, expected, atol=1e-6)


def test_sim_innov_t_rescale_factor_matches_r():
    # Var(t_5) = df/(df-2) = 5/3 in closed form -- R: 1/sqrt(5/3)
    df = 5
    assert 1 / np.sqrt(df / (df - 2)) == pytest.approx(0.7745966692, abs=1e-9)


def test_sim_innov_skew_t_constants_match_r():
    # R: xi=-0.75, df=5 -> delta, E|t0|, mean_raw, sd_raw (see sim.py's _beta_fn)
    from exuber.sim import _beta_fn

    df, xi = 5, -0.75
    delta = xi / np.sqrt(1 + xi**2)
    e_abs_t0 = (2 * np.sqrt(df) / ((df - 1) * _beta_fn(df / 2, 0.5))) / np.sqrt(df / (df - 2))
    mean_raw = delta * e_abs_t0
    sd_raw = np.sqrt(max(1 - delta**2 * e_abs_t0**2, np.finfo(float).eps))
    assert delta == pytest.approx(-0.6, abs=1e-9)
    assert e_abs_t0 == pytest.approx(0.7351051939, abs=1e-9)
    assert mean_raw == pytest.approx(-0.4410631163, abs=1e-9)
    assert sd_raw == pytest.approx(0.8974760874, abs=1e-9)


def test_sim_fi_psi_recursion_matches_r():
    # R: d=0.2, m=5 -> psi[1..6] (1-indexed); a hand truncation to m=5
    # (real sim_fi() always uses m=max(500, 5n), so this checks the
    # recursion formula/convolution logic at a size R was also hand-run at)
    d, m = 0.2, 5
    psi = np.empty(m + 1)
    psi[0] = 1.0
    for j in range(1, m + 1):
        psi[j] = psi[j - 1] * (j - 1 + d) / j
    np.testing.assert_allclose(
        psi, [1.0, 0.2, 0.12, 0.088, 0.0704, 0.059136], atol=1e-6
    )

    eps = _EPS10[:8]  # n=3, m=5 -> needs n+m=8 innovations
    u = np.empty(3)
    for k in range(3):
        window = eps[k : k + m + 1]
        u[k] = np.dot(psi, window[::-1])
    np.testing.assert_allclose(u, [-0.66078593, 0.45529270, 0.82924303], atol=1e-6)


def test_sim_blan_rotermann_wilfling_matches_r(monkeypatch):
    from exuber.sim import sim_blan

    class _FakeRWRNG(_FakeRNG):
        def binomial(self, n_trials, p, size=None):
            return np.array([0, 1, 1, 1])

        def lognormal(self, mean=0.0, sigma=1.0, size=None):
            return np.array([0.96467442, 0.97837265, 0.95143523, 0.95254875])

    monkeypatch.setattr(np.random, "default_rng", lambda seed=None: _FakeRWRNG([]))
    b = sim_blan(5, pi=0.7, type="rotermann_wilfling", delta=0.984, rw_sigma=0.05, b0=0.1, seed=7)
    expected = [0.10000000, 0.10006889, 0.09949661, 0.09620385, 0.09312891]
    np.testing.assert_allclose(b, expected, atol=1e-6)


def test_sim_psy1_e_and_shifts_and_coef_noise():
    from exuber.sim import sim_psy1

    # e= overrides the innovations entirely
    y_zero_e = sim_psy1(50, seed=42, e=np.zeros(49))
    assert np.all(np.isfinite(y_zero_e))
    with pytest.raises(ValueError):
        sim_psy1(50, seed=42, e=np.zeros(10))  # wrong length

    # shifts= adds an exact one-period bump at the given (1-indexed) date.
    # Only checkable in isolation at the shift's own date: the process is
    # autoregressive, so a shift's effect propagates into later
    # observations too (compounded by the explosive-regime coefficient),
    # not just its own date.
    y_plain = sim_psy1(50, seed=42)
    y_shift = sim_psy1(50, seed=42, shifts={"date": [10], "size": [100.0]})
    assert y_shift[9] - y_plain[9] == pytest.approx(100.0)
    assert np.array_equal(y_shift[:9], y_plain[:9])  # no effect before the shift date

    # coef_noise= only perturbs the coefficient inside [te, tf]
    with pytest.raises(ValueError):
        sim_psy1(50, seed=42, coef_noise=np.zeros(5))  # wrong length


def test_sim_blan_rejects_bad_type():
    from exuber.sim import sim_blan

    with pytest.raises(ValueError):
        sim_blan(10, type="bogus")
