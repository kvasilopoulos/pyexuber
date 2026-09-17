import numpy as np
import pytest

from exuber.cv import RadfCv
from exuber.datestamp import datestamp, svadf_threshold
from exuber.radf import RadfResult, psy_ds


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


def test_datestamp_requires_cv_unless_svadf():
    result, _cv = _make_result_and_cv()
    with pytest.raises(ValueError, match="cv is required"):
        datestamp(result, option="gsadf")


def test_svadf_threshold_formula():
    t = np.array([100.0])
    np.testing.assert_allclose(svadf_threshold(t, "origination"), np.log(100) / 10)
    np.testing.assert_allclose(svadf_threshold(t, "collapse"), np.log(100) / 2)
    with pytest.raises(ValueError):
        svadf_threshold(t, "bogus")


def _make_svadf_result(badf_col, minw=5, lag=0):
    badf = np.asarray(badf_col)[:, None]
    n = badf.shape[0] + minw + lag
    return RadfResult(
        adf=np.array([0.0]), badf=badf, sadf=np.array([0.0]),
        bsadf=badf, gsadf=np.array([0.0]),
        bsadf_panel=badf.mean(axis=1), gsadf_panel=0.0,
        minw=minw, lag=lag, n=n, series_names=["x"],
    )


def test_datestamp_svadf_detects_origination_and_collapse():
    # thresholds at these small t are well inside (-1, 2): -1 stays below
    # orig_thresh, 2.0 clears it; -2.0 later clears (goes below) coll_thresh.
    badf_path = [-1] * 5 + [2] * 5 + [-2] * 10
    result = _make_svadf_result(badf_path)
    with pytest.warns(UserWarning, match="non-peer-reviewed preprint"):
        out = datestamp(result, option="svadf")

    assert list(out.keys()) == ["x"]
    ep = out["x"][0]
    zadj = 5
    assert ep.start == 5 + zadj
    assert ep.end == 10 + zadj
    assert ep.ongoing is False
    assert ep.duration == 5


def test_datestamp_svadf_ongoing_when_no_collapse():
    result = _make_svadf_result([-1] * 5 + [3] * 15)
    out = datestamp(result, option="svadf")
    ep = out["x"][0]
    assert ep.ongoing is True
    assert ep.end is None


def test_datestamp_svadf_no_episode_when_never_exceeds_threshold():
    result = _make_svadf_result([-5.0] * 20)
    out = datestamp(result, option="svadf")
    assert out == {}


def test_datestamp_svadf_collapse_never_before_origination():
    """Structural invariant: the search for a collapse only starts after
    the origination row, so End can never precede Start when both are
    found -- checked across several synthetic badf paths."""
    rng = np.random.default_rng(3)
    for _ in range(20):
        badf = rng.normal(scale=2, size=30)
        result = _make_svadf_result(badf, minw=20)
        out = datestamp(result, option="svadf", min_duration=psy_ds(50))
        if "x" in out:
            ep = out["x"][0]
            if not ep.ongoing:
                assert ep.end is not None
                assert ep.end > ep.start
