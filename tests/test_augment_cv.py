"""augment() for RadfCv/RadfSbCv and augment_join(), checked against R's
own output (exuber 2.0.0 dev), same objects as test_tidy_cv.py:
mc <- radf_mc_cv(30, minw=10, nrep=300, seed=123);
wb <- radf_wb_cv(dta, minw=10, nboot=20, seed=1);
sb <- radf_sb_cv(dta, minw=10, lag=1, nboot=20, seed=1).
Only the leading rows of each cv sequence are R's; the rest are filler,
since the checks are about layout (row order, keys, padding) plus those
leading values. R's `index`/`data` columns are not carried (see tidy.py)."""

import numpy as np
import pytest
from test_tidy import RESULT

from exuber.cv import RadfCv, RadfSbCv
from exuber.tidy import augment, augment_join

pytest.importorskip("pandas")


def _seq(rows, head):
    out = np.zeros((rows, *np.shape(head)[1:]))
    out[: len(head)] = head
    return out


MC = RadfCv(
    adf_cv=np.zeros(3), sadf_cv=np.zeros(3), gsadf_cv=np.zeros(3),
    badf_cv=np.tile([-0.44, -0.08, 0.60], (20, 1)),
    bsadf_cv=_seq(20, [[-0.25725705, 0.3032252, 0.6426675],
                       [0.04567163, 0.5246360, 1.5498861]]),
    method="Monte Carlo", minw=10, n=30, iter=300,
)
# (rows, sig, series)
WB_BADF = _seq(20, np.stack([[[-1.0608085, -0.8208085, -0.5185410],
                              [-0.8790170, -0.8, -0.5]],
                             [[-0.2720711, -0.2269112, -0.002063997],
                              [-0.2, -0.1, 0.0]]], axis=2))
WB_BSADF = _seq(20, np.stack([[[-1.0608085, -0.8208085, -0.5185410],
                               [-0.7529651, -0.6991336, -0.3405395]],
                              [[-0.2720711, -0.2269112, -0.002063997],
                               [-0.2549559, -0.1191308, 0.042151482]]], axis=2))
WB = RadfCv(
    adf_cv=np.zeros((2, 3)), sadf_cv=np.zeros((2, 3)), gsadf_cv=np.zeros((2, 3)),
    badf_cv=WB_BADF, bsadf_cv=WB_BSADF,
    method="Wild Bootstrap", minw=10, n=30, iter=20, series_names=["a", "b"],
)
SB = RadfSbCv(
    gsadf_panel_cv=np.zeros(3),
    bsadf_panel_cv=_seq(19, [[-1.252444, -1.0084789, -0.9671652],
                             [-1.024851, -0.8575255, -0.8513641]]),
    method="Sieve Bootstrap", minw=10, n=30, iter=20, lag=1,
)


def test_augment_mc_cv():
    wide = augment(MC)
    assert wide.columns.tolist() == ["key", "sig", "badf", "bsadf"]
    assert len(wide) == 60
    assert wide.iloc[3].tolist() == [12, "90", -0.44, pytest.approx(0.04567163)]
    assert wide.iloc[-1][["key", "sig"]].tolist() == [30, "99"]
    long = augment(MC, "long")
    assert long.columns.tolist() == ["key", "stat", "sig", "crit"]
    assert len(long) == 120
    assert long["stat"].tolist() == ["badf"] * 60 + ["bsadf"] * 60
    full = augment(MC, trunc=False)
    assert len(full) == 90 and full["key"].iloc[0] == 1 and np.isnan(full["badf"].iloc[0])


def test_augment_wb_cv():
    wide = augment(WB)
    assert wide.columns.tolist() == ["key", "id", "sig", "badf", "bsadf"]
    assert len(wide) == 120
    assert wide.iloc[3].tolist() == [12, "a", "90", pytest.approx(-0.8790170),
                                     pytest.approx(-0.7529651)]
    assert wide.iloc[-1][["key", "id", "sig"]].tolist() == [30, "b", "99"]
    long = augment(WB, "long")
    assert long.columns.tolist() == ["key", "id", "stat", "sig", "crit"]
    assert len(long) == 240
    assert long.iloc[1].tolist() == [11, "a", "bsadf", "90", pytest.approx(-1.0608085)]
    assert long.iloc[2][["stat", "sig"]].tolist() == ["badf", "95"]


def test_augment_sb_cv():
    wide = augment(SB)
    assert wide.columns.tolist() == ["key", "sig", "bsadf_panel"]
    assert len(wide) == 57 and wide["key"].iloc[0] == 12
    long = augment(SB, "long")
    assert long.iloc[0].tolist() == [12, "panel", "bsadf_panel", "90", pytest.approx(-1.252444)]


def test_augment_join_sb():
    aj = augment_join(RESULT, SB)
    assert aj.columns.tolist() == ["key", "id", "stat", "tstat", "sig", "crit"]
    assert len(aj) == 57
    assert aj.iloc[0].tolist() == [12, "panel", "bsadf_panel", pytest.approx(-2.57239567),
                                   "90", pytest.approx(-1.252444)]
    assert aj.iloc[1].tolist() == [13, "panel", "bsadf_panel", pytest.approx(-2.34023288),
                                   "90", pytest.approx(-1.024851)]


def test_augment_join_wb_order():
    aj = augment_join(RESULT, WB)
    # R: arrange(id, stat, sig), keys ascending within
    assert aj[["id", "stat", "sig"]].iloc[0].tolist() == ["a", "badf", "90"]
    assert aj[["id", "stat", "sig"]].iloc[-1].tolist() == ["b", "bsadf", "99"]
    assert aj["key"].is_monotonic_increasing is False
    first = aj[(aj["id"] == "a") & (aj["stat"] == "badf") & (aj["sig"] == "90")]
    assert first["key"].tolist() == list(range(11, 31))
