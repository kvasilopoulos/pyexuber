"""Unit tests for dating_hlw.py, ported from exuber's test-dating_hlw.R.

_dating_hlw_from_episodes(): pure window-construction + per-window HLS
fitting, no radf()/datestamp()/the C++ extension needed (see module
docstring). dating_hlw() itself needs radf()/radf_wb_cv()/datestamp(),
only passes where the extension is built -- see module docstring, same
split as radf_recovery() in test_radf_recovery.py."""

import math

import numpy as np

from exuber.datestamp import Episode
from exuber.dating_hls import dating_hls
from exuber.dating_hlw import _dating_hlw_from_episodes, _hlw_local_to_global, dating_hlw

# -- dating_hlw() -----------------------------------------------------------


def _sim_hls_model4(seed, n1=60, n2=25, n3=25, n4=40, base=100.0, c_bubble=1.05):
    rng = np.random.default_rng(seed)
    unit1 = base + np.cumsum(rng.normal(size=n1))
    bubble = unit1[-1] * c_bubble ** np.arange(1, n2 + 1) + np.cumsum(rng.normal(size=n2))
    target = bubble[-1] * 0.5
    collapse = np.empty(n3)
    collapse[0] = bubble[-1] + rng.normal()
    for k in range(1, n3):
        collapse[k] = target + 0.85 * (collapse[k - 1] - target) + rng.normal()
    recovery = collapse[-1] + np.cumsum(rng.normal(size=n4))
    y = np.concatenate([unit1, bubble, collapse, recovery])
    return y, n1, n1 + n2


def test_hlw_local_to_global_arithmetic():
    # R's own check: local_tau=5, s=21 (1-indexed) -> i_index=25, position=26.
    # s0 (0-indexed window start) = s - 1 = 20.
    assert _hlw_local_to_global(local_tau=5, s0=20) == 26


def test_dating_hlw_single_episode_reduces_to_dating_hls():
    y, _t1, _t2 = _sim_hls_model4(11)
    n = len(y)
    ep = Episode(start=0, peak=0, end=None, duration=n, ongoing=True)

    hlw_eps = _dating_hlw_from_episodes(y, [ep], n, trim=0.1)
    hls_out = dating_hls(y, trim=0.1)

    assert len(hlw_eps) == 1
    last = hlw_eps[0]
    assert last.model == hls_out.model[0]
    assert last.origination == hls_out.origination[0]
    assert last.collapse == hls_out.collapse[0]
    if math.isnan(hls_out.recovery[0]):
        assert math.isnan(last.recovery)
    else:
        assert last.recovery == hls_out.recovery[0]


def test_dating_hlw_no_episodes_returns_empty_list():
    y = np.cumsum(np.random.default_rng(1).normal(size=50))
    assert _dating_hlw_from_episodes(y, [], n=50, trim=0.1) == []


def _sim_two_bubbles(seed, n1a=50, n2a=20, n3a=30, n1b=50, n2b=20, n3b=30):
    rng = np.random.default_rng(seed)
    e1 = 100 + np.cumsum(rng.normal(size=n1a))
    b1 = e1[-1] * 1.05 ** np.arange(1, n2a + 1) + np.cumsum(rng.normal(size=n2a))
    u1 = b1[-1] + np.cumsum(rng.normal(size=n3a))
    e2 = u1[-1] + np.cumsum(rng.normal(size=n1b))
    b2 = e2[-1] * 1.05 ** np.arange(1, n2b + 1) + np.cumsum(rng.normal(size=n2b))
    u2 = b2[-1] + np.cumsum(rng.normal(size=n3b))
    y = np.concatenate([e1, b1, u1, e2, b2, u2])
    true1 = (n1a, n1a + n2a)
    true2 = (n1a + n2a + n3a + n1b, n1a + n2a + n3a + n1b + n2b)
    return y, true1, true2


def test_dating_hlw_two_windows_recovers_both_bubbles_accurately():
    # hand-built "PSY-detected" episodes (a few observations of slack on
    # either side of the true regime, as a real step-1 pass would give)
    # rather than relying on datestamp()'s own detection noise -- keeps
    # this test independent of the C++ extension while still exercising
    # the real window-construction + sequential-adjustment logic.
    y, true1, true2 = _sim_two_bubbles(1)
    n = len(y)
    ep1 = Episode(start=true1[0] - 5, peak=0, end=true1[1] + 5, duration=0, ongoing=False)
    ep2 = Episode(start=true2[0] - 5, peak=0, end=true2[1] + 5, duration=0, ongoing=False)

    eps = _dating_hlw_from_episodes(y, [ep1, ep2], n, trim=0.1)

    assert len(eps) == 2
    assert eps[0].origination < eps[1].origination
    assert abs(eps[0].origination - true1[0]) < 10
    assert abs(eps[1].origination - true2[0]) < 10


def test_dating_hlw_end_to_end_well_formed():
    rng = np.random.default_rng(11)
    n1, n2, n3, n4 = 60, 25, 25, 40
    unit1 = 100 + np.cumsum(rng.normal(size=n1))
    bubble = unit1[-1] * 1.05 ** np.arange(1, n2 + 1) + np.cumsum(rng.normal(size=n2))
    target = bubble[-1] * 0.5
    collapse = np.empty(n3)
    collapse[0] = bubble[-1] + rng.normal()
    for k in range(1, n3):
        collapse[k] = target + 0.85 * (collapse[k - 1] - target) + rng.normal()
    recovery = collapse[-1] + np.cumsum(rng.normal(size=n4))
    y = np.concatenate([unit1, bubble, collapse, recovery])

    out = dating_hlw(y, trim=0.1, nboot=199, seed=1)
    assert "series1" in out.episodes
    assert isinstance(out.episodes["series1"], list)


def test_dating_hlw_h0_returns_zero_windows():
    rng = np.random.default_rng(2)
    y = 100 + np.cumsum(rng.normal(size=150))
    out = dating_hlw(y, trim=0.1, nboot=199, seed=1)
    assert out.episodes["series1"] == []
