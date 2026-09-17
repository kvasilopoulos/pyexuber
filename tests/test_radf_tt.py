"""Tests for exuber.radf_tt -- the STADF/GSTADF time-transformed test.
Port of exuber's R/radf_tt.R; see docs/volatility-robustness.md (root
repo) for the source paper.

Formula-exact reference numbers below were produced by feeding the SAME
deterministic input series to both R and this port (no RNG involved at
the formula level, so results agree to numpy/R floating-point precision,
not just approximately) -- see
docs/replication/volatility-robustness/radf_tt_validation.py (root repo)
for the fuller narrative version and the exact R commands used.
"""

import numpy as np
import pytest

from exuber._gls_dfstat import gls_dfstat_grid
from exuber.radf import psy_minw
from exuber.radf_tt import radf_tt, radf_tt_cv, variance_profile

# set.seed(7); y <- round(cumsum(rnorm(40)), 8)
Y_VEC = np.array(
    [
        2.28724716, 1.09047548, 0.39618297, -0.01610998, -0.98678332, -1.93406327,
        -1.18592393, -1.30287915, -1.15022153, 1.03975658, 1.39674281, 4.11349459,
        6.39494652, 6.71896706, 8.61503413, 9.08271464, 8.18891391, 7.88158561,
        7.87676319, 8.86492734, 9.7046777, 10.41001953, 11.71598425, 10.32798804,
        11.6009049, 11.78509767, 12.53737757, 13.12912262, 12.14607002, 11.87000607,
        10.99915505, 11.7178656, 11.82851848, 11.75005171, 11.32956125, 10.76743537,
        11.76494882, 10.65981876, 10.51753093, 10.83252583,
    ]
)
MINW = 10


def test_gls_dfstat_grid_matches_r():
    res = gls_dfstat_grid(Y_VEC, MINW)
    assert res["sadf"] == pytest.approx(1.2107378397, abs=1e-6)
    assert res["gsadf"] == pytest.approx(1.9637889135, abs=1e-6)


def test_variance_profile_matches_r():
    eta_grid, omega2 = variance_profile(Y_VEC, kernel="uniform")
    assert eta_grid[-1] == pytest.approx(1.0, abs=1e-6)
    assert eta_grid[0] == pytest.approx(0.0, abs=1e-6)
    assert omega2 == pytest.approx(0.8233400111, abs=1e-6)


def test_radf_tt_matches_r():
    res = radf_tt(Y_VEC, minw=MINW, kernel="uniform")
    assert res.adf[0] == pytest.approx(0.3401194521, abs=1e-6)
    assert res.sadf[0] == pytest.approx(1.1571296746, abs=1e-6)
    assert res.gsadf[0] == pytest.approx(1.8910439255, abs=1e-6)
    assert res.minw == MINW
    assert res.lag == 0


def test_radf_tt_cv_against_published_whitehouse():
    """Pivotal (RNG-agnostic) asymptotic target -- comparable directly to
    Whitehouse (2019)'s published STADF triple even with numpy's RNG."""
    cv = radf_tt_cv(n=300, minw=30, nrep=800, seed=555)
    published = np.array([2.319, 2.626, 3.223])
    assert np.all(np.abs(cv.sadf_cv - published) < 0.4)


def test_radf_tt_cv_badf_cv_identity():
    cv = radf_tt_cv(n=60, minw=15, nrep=100, seed=1)
    np.testing.assert_allclose(cv.badf_cv[-1], cv.adf_cv)


def test_radf_tt_default_minw_uses_psy_minw():
    res = radf_tt(Y_VEC)
    assert res.minw == psy_minw(len(Y_VEC))
