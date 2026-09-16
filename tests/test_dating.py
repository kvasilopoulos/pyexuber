"""Unit tests for dating.py, ported from exuber's test-rootstamp.R.
Formula-exact checks use pytest.approx with a tight tolerance (matches
this project's convention in test_datestamp.py etc.); Monte Carlo checks
use loose bounds, per the same finite-sample honesty this family's R
tests document (see docs/dating-and-root-inference.md)."""

import math

import numpy as np
import pytest

from exuber.datestamp import Episode
from exuber.dating import RootstampEpisode, rootstamp, rootstamp_episodes

# -- rootstamp() --------------------------------------------------------


def test_cauchy_percentiles_match_student_t_df1():
    # standard Cauchy IS Student's t at df=1 -- pure math identity, exact
    published = [6.314, 12.7, 63.65674]
    computed = [
        math.tan(math.pi * (p - 0.5)) for p in (0.95, 0.975, 0.995)
    ]
    for c, p in zip(computed, published, strict=True):
        assert c == pytest.approx(p, rel=1e-3)


def _simulate_ar1(rho, n, seed):
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    e = rng.normal(size=n)
    for t in range(1, n):
        y[t] = rho * y[t - 1] + e[t]
    return y


def test_rootstamp_recovers_known_rho():
    y = _simulate_ar1(1.03, 150, seed=1)
    fit = rootstamp(y)
    assert fit.rho == pytest.approx(1.03, abs=0.01)
    assert fit.se > 0
    assert fit.n == 149


def test_rootstamp_doubling_time_consistent_with_rho():
    y = _simulate_ar1(1.03, 150, seed=1)
    fit = rootstamp(y)
    assert fit.doubling_time == pytest.approx(math.log(2) / math.log(fit.rho))
    # doubling time decreases in rho, so the CI bounds are flipped
    assert fit.doubling_time_ci[0] < fit.doubling_time < fit.doubling_time_ci[1]
    assert fit.rho_ci[0] < fit.rho < fit.rho_ci[1]


def test_rootstamp_coverage_plausible_at_moderate_t():
    rho_true, n = 1.05, 200
    rng = np.random.default_rng(24601)
    covered = 0
    for _ in range(300):
        y = np.zeros(n)
        e = rng.normal(size=n)
        for t in range(1, n):
            y[t] = rho_true * y[t - 1] + e[t]
        ci = rootstamp(y)
        if ci.rho_ci[0] <= rho_true <= ci.rho_ci[1]:
            covered += 1
    # not the nominal 95% -- finite-sample undercoverage is expected and
    # documented (docs/dating-and-root-inference.md); a generous band only.
    assert covered / 300 > 0.80


def test_rootstamp_cauchy_brackets_point_estimate_and_matches_eq27():
    y = _simulate_ar1(1.05, 150, seed=1)
    ci = rootstamp(y, type="cauchy")
    assert ci.rho_ci[0] < ci.rho < ci.rho_ci[1]

    q = math.tan(math.pi * (0.975 - 0.5))
    half_width = q * (ci.rho**2 - 1) / ci.rho**ci.n
    assert ci.rho_ci == pytest.approx((ci.rho - half_width, ci.rho + half_width))


def test_rootstamp_rejects_bad_args():
    y = _simulate_ar1(1.03, 50, seed=1)
    with pytest.raises(ValueError):
        rootstamp(y, type="bogus")
    with pytest.raises(ValueError):
        rootstamp(y, sig_lvl=30)


def test_rootstamp_episodes_matches_direct_call():
    ep = Episode(start=10, peak=15, end=30, duration=20, ongoing=False)
    rng = np.random.default_rng(7)
    y = np.cumsum(rng.normal(size=40))
    ds = {"series1": [ep]}

    out = rootstamp_episodes(y, ds)
    direct = rootstamp(y[ep.start : ep.end])

    assert list(out.keys()) == ["series1"]
    row = out["series1"][0]
    assert isinstance(row, RootstampEpisode)
    assert row.rho == pytest.approx(direct.rho)
    assert row.rho_lower == pytest.approx(direct.rho_ci[0])
    assert row.rho_upper == pytest.approx(direct.rho_ci[1])


def test_rootstamp_episodes_ongoing_slices_to_series_end():
    ep = Episode(start=5, peak=8, end=None, duration=10, ongoing=True)
    rng = np.random.default_rng(8)
    y = np.cumsum(rng.normal(size=20))
    out = rootstamp_episodes(y, {"series1": [ep]})
    direct = rootstamp(y[5:])
    assert out["series1"][0].rho == pytest.approx(direct.rho)
