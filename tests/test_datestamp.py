import numpy as np
import pytest

from exuber.cv import RadfCv
from exuber.datestamp import datestamp
from exuber.radf import RadfResult


def _make_result_and_cv(minw=5, lag=0):
    bsadf = np.array([[0.0], [0.0], [3.0], [5.0], [2.0], [0.0], [0.0], [4.0]])
    result = RadfResult(
        adf=np.array([0.0]),
        badf=np.zeros((8, 1)),
        sadf=np.array([0.0]),
        bsadf=bsadf,
        gsadf=np.array([10.0]),
        bsadf_panel=bsadf.mean(axis=1),
        gsadf_panel=10.0,
        minw=minw,
        lag=lag,
        n=8 + minw + lag,
        series_names=["x"],
    )
    cv = RadfCv(
        adf_cv=np.array([-1.0, -0.5, 0.0]),
        sadf_cv=np.array([0.0, 0.5, 1.0]),
        gsadf_cv=np.array([1.0, 2.0, 3.0]),
        badf_cv=np.zeros((8, 3)),
        bsadf_cv=np.tile(np.array([1.0, 1.0, 1.0]), (8, 1)),
        method="test", minw=minw, n=8 + minw + lag, iter=100,
    )
    return result, cv


def test_datestamp_finds_two_episodes_one_ongoing():
    result, cv = _make_result_and_cv(minw=5, lag=0)
    episodes = datestamp(result, cv, sig_lvl=95, option="gsadf")

    assert list(episodes.keys()) == ["x"]
    eps = episodes["x"]
    assert len(eps) == 2

    zadj = 5
    first, second = eps
    assert (first.start, first.peak, first.end, first.duration, first.ongoing) == (
        2 + zadj, 3 + zadj, 5 + zadj, 3, False,
    )
    assert (second.start, second.peak, second.duration, second.ongoing) == (
        7 + zadj, 7 + zadj, 1, True,
    )
    assert second.end is None


def test_datestamp_min_duration_filters_short_episodes():
    result, cv = _make_result_and_cv()
    episodes = datestamp(result, cv, sig_lvl=95, option="gsadf", min_duration=2)

    eps = episodes["x"]
    assert len(eps) == 1
    assert eps[0].duration == 3


def test_datestamp_drops_series_that_dont_reject_overall():
    result, cv = _make_result_and_cv()
    result.gsadf[0] = 0.5  # below even the 90% cv (1.0)
    episodes = datestamp(result, cv, sig_lvl=95, option="gsadf")
    assert episodes == {}


def test_datestamp_rejects_bad_args():
    result, cv = _make_result_and_cv()

    with pytest.raises(ValueError):
        datestamp(result, cv, sig_lvl=80)
    with pytest.raises(ValueError):
        datestamp(result, cv, option="badf")
    with pytest.raises(ValueError):
        datestamp(result, cv, min_duration=-1)
