"""Unit tests for radf_recovery.py, ported from exuber's test-recovery.R.

radf_recovery()/radf_recovery_cv() themselves need radf() (the C++
extension); their end-to-end tests below will only pass where that's
built (CI), same as test_cv.py's radf_mc_cv/radf_wb_cv tests. The
crossing-detection logic is factored into _recovery_dates_from_bsadf(),
tested separately below with synthetic bsadf arrays -- no extension
needed, same split as test_datestamp.py testing datestamp() with a
hand-built RadfResult/RadfCv."""

import numpy as np
import pytest

from exuber.radf_recovery import _recovery_dates_from_bsadf, radf_recovery, radf_recovery_cv

# -- radf_recovery() -------------------------------------------------------
# _recovery_dates_from_bsadf(): pure crossing-detection logic, no C++
# extension needed (see module docstring).


def _up_then_down(n_minw=8, up_at=2, down_at=5):
    bsadf = np.zeros((n_minw, 1))
    bsadf[up_at:down_at, 0] = 3.0
    cv = np.tile([1.0, 1.0, 1.0], (n_minw, 1))
    return bsadf, cv


def test_recovery_dates_up_then_down_crossing():
    bsadf, cv = _up_then_down(n_minw=8, up_at=2, down_at=5)
    res = _recovery_dates_from_bsadf(bsadf, cv, minw=5, lag=0, n=13, sig_lvl=95)

    assert res.detected[0]
    assert not res.censored[0]
    zadj = 5
    assert res.f_r[0] == 13 - 1 - (2 + zadj)
    assert res.f_c[0] == 13 - 1 - (5 + zadj)
    assert res.f_c[0] <= res.f_r[0]


def test_recovery_dates_censored_when_no_down_crossing():
    n_minw = 8
    bsadf = np.zeros((n_minw, 1))
    bsadf[2:, 0] = 3.0  # up-crossing, never comes back down
    cv = np.tile([1.0, 1.0, 1.0], (n_minw, 1))
    res = _recovery_dates_from_bsadf(bsadf, cv, minw=5, lag=0, n=13, sig_lvl=95)

    assert res.detected[0]
    assert res.censored[0]
    assert np.isnan(res.f_c[0])
    assert not np.isnan(res.f_r[0])


def test_recovery_dates_not_detected_when_never_crosses():
    n_minw = 8
    bsadf = np.zeros((n_minw, 1))
    cv = np.tile([1.0, 1.0, 1.0], (n_minw, 1))
    res = _recovery_dates_from_bsadf(bsadf, cv, minw=5, lag=0, n=13, sig_lvl=95)

    assert not res.detected[0]
    assert not res.censored[0]
    assert np.isnan(res.f_c[0])
    assert np.isnan(res.f_r[0])


def test_recovery_dates_rejects_bad_sig_lvl():
    bsadf, cv = _up_then_down()
    with pytest.raises(ValueError):
        _recovery_dates_from_bsadf(bsadf, cv, minw=5, lag=0, n=13, sig_lvl=80)


def test_recovery_dates_multiseries_independent():
    n_minw = 8
    bsadf = np.zeros((n_minw, 2))
    bsadf[2:5, 0] = 3.0  # series 0: up then down
    # series 1: never crosses
    cv = np.tile([1.0, 1.0, 1.0], (n_minw, 1))
    res = _recovery_dates_from_bsadf(bsadf, cv, minw=5, lag=0, n=13, sig_lvl=95)

    assert res.detected[0] and not res.detected[1]
    assert not np.isnan(res.f_r[0])
    assert np.isnan(res.f_r[1])


# radf_recovery()/radf_recovery_cv(): need the C++ extension (radf()),
# only pass where it's built -- see module docstring.


def test_radf_recovery_cv_shape():
    cv = radf_recovery_cv(n=100, minw=20, nrep=50, seed=1)
    assert cv.bsadf_cv.shape == (80, 3)
    assert cv.minw == 20
    assert cv.n == 100
    assert cv.iter == 50


def test_radf_recovery_runs_end_to_end():
    rng = np.random.default_rng(1)
    y = np.cumsum(rng.normal(size=100))
    with pytest.warns(UserWarning):
        out = radf_recovery(y, minw=20, nrep=50, seed=1)
    assert out.detected.dtype == bool
    assert out.censored.dtype == bool
    assert out.n == 100


def test_radf_recovery_rejects_bad_sig_lvl():
    rng = np.random.default_rng(1)
    y = np.cumsum(rng.normal(size=100))
    with pytest.raises(ValueError):
        radf_recovery(y, minw=20, nrep=50, sig_lvl=93)


def test_radf_recovery_f_c_never_exceeds_f_r():
    def run_once(seed):
        rng = np.random.default_rng(seed)
        n1, n2, n3 = 40, 25, 35
        expansion = 100 * 1.03 ** np.arange(1, n1 + 1) + np.cumsum(rng.normal(size=n1))
        collapse = expansion[-1] * 0.5 ** (np.arange(1, n2 + 1) / n2) + np.cumsum(
            rng.normal(size=n2)
        )
        recovery = collapse[-1] + np.cumsum(rng.normal(size=n3)) + np.arange(1, n3 + 1) * 0.5
        y = np.concatenate([expansion, collapse, recovery])
        with pytest.warns(UserWarning):
            out = radf_recovery(y, minw=15, nrep=50, seed=1)
        if not out.detected[0] or out.censored[0]:
            return None
        return out.f_c[0] <= out.f_r[0]

    results = [r for r in (run_once(s) for s in range(10)) if r is not None]
    assert all(results)
