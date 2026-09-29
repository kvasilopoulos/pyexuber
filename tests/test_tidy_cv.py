"""tidy() for RadfCv/RadfDistr and tidy_join(), checked against R's own
output (exuber 2.0.0 dev): same `dta`/`radf(dta, minw=10, lag=1)` as
test_diagnostics.py; mc <- radf_mc_cv(30, minw=10, nrep=300, seed=123);
wb <- radf_wb_cv(dta, minw=10, nboot=20, seed=1);
sb <- radf_sb_cv(dta, minw=10, lag=1, nboot=20, seed=1);
wd <- radf_wb_distr(dta, minw=10, nboot=3, seed=1)."""

import numpy as np
import pytest
from test_diagnostics import MC, RESULT

from exuber.cv import RadfCv, RadfDistr, RadfSbCv, RadfSbDistr
from exuber.tidy import tidy, tidy_join

pytest.importorskip("pandas")

WB = RadfCv(
    adf_cv=np.array([[-0.8624607078, -0.7863651059, -0.4092731238],
                     [-1.016034371, -0.9475845117, -0.2459603129]]),
    sadf_cv=np.array([[0.06902419281, 0.263186663, 0.3010102565],
                      [-0.09672821289, 0.0745500493, 0.3846641397]]),
    gsadf_cv=np.array([[1.406854759, 1.604526677, 3.406507451],
                       [0.4777904159, 0.5640575295, 1.157662296]]),
    badf_cv=np.zeros((19, 3, 2)), bsadf_cv=np.zeros((19, 3, 2)),
    method="Wild Bootstrap", minw=10, n=30, iter=20, series_names=["a", "b"],
)
SB = RadfSbCv(
    gsadf_panel_cv=np.array([0.3883875264, 0.6090607544, 0.8972463401]),
    bsadf_panel_cv=np.zeros((19, 3)), method="Sieve Bootstrap", minw=10, n=30, iter=20, lag=1,
)


def test_tidy_mc_cv():
    wide = tidy(MC)
    assert wide.columns.tolist() == ["sig", "adf", "sadf", "gsadf"]
    assert wide["sig"].tolist() == ["90", "95", "99"]
    np.testing.assert_allclose(wide["gsadf"], [1.457097501, 1.80047179, 2.500302695])
    long = tidy(MC, "long")
    assert long.columns.tolist() == ["stat", "sig", "crit"]
    assert long["stat"].tolist() == ["adf"] * 3 + ["sadf"] * 3 + ["gsadf"] * 3
    assert long["crit"].iloc[4] == pytest.approx(1.302069063)


def test_tidy_wb_cv():
    wide = tidy(WB)
    assert wide.columns.tolist() == ["id", "sig", "adf", "sadf", "gsadf"]
    assert wide["id"].tolist() == ["a", "b"] * 3
    assert wide["sig"].tolist() == ["90", "90", "95", "95", "99", "99"]
    np.testing.assert_allclose(wide["adf"], [-0.8624607078, -1.016034371, -0.7863651059,
                                             -0.9475845117, -0.4092731238, -0.2459603129])
    long = tidy(WB, "long")
    assert long.columns.tolist() == ["id", "stat", "sig", "crit"]
    assert long.iloc[7].tolist() == ["b", "sadf", "90", pytest.approx(-0.09672821289)]


def test_tidy_sb_cv():
    assert tidy(SB).columns.tolist() == ["id", "sig", "gsadf_panel"]
    long = tidy(SB, "long")
    assert long.iloc[2].tolist() == ["panel", "gsadf_panel", "99", pytest.approx(0.8972463401)]


def test_tidy_distr():
    wd = RadfDistr(
        adf_distr=np.array([[-1.112824118, -2.915467599], [-2.223014919, -3.130538564],
                            [-3.624708728, -2.126122471]]),
        sadf_distr=np.zeros((3, 2)), gsadf_distr=np.zeros((3, 2)),
        method="Wild Bootstrap", minw=10, n=30, iter=3,
    )
    df = tidy(wd)
    assert df.columns.tolist() == ["id", "adf", "sadf", "gsadf"]
    assert df["id"].tolist() == ["series1"] * 3 + ["series2"] * 3
    np.testing.assert_allclose(df["adf"], [-1.112824118, -2.223014919, -3.624708728,
                                           -2.915467599, -3.130538564, -2.126122471])
    mc = RadfDistr(adf_distr=np.arange(4.0), sadf_distr=np.ones(4), gsadf_distr=np.ones(4),
                   method="Monte Carlo", minw=10, n=30, iter=4)
    assert tidy(mc).shape == (4, 3)
    sb = RadfSbDistr(gsadf_panel_distr=np.ones(5), method="Sieve Bootstrap",
                     minw=10, n=30, iter=5, lag=1)
    assert tidy(sb).columns.tolist() == ["gsadf_panel"]


def test_tidy_join():
    mc = tidy_join(RESULT, MC)
    assert mc.columns.tolist() == ["id", "stat", "tstat", "sig", "crit"]
    assert len(mc) == 18
    assert mc.iloc[10].tolist() == ["b", "adf", pytest.approx(-4.225519136), "95",
                                    pytest.approx(-0.01850857045)]
    wb = tidy_join(RESULT, WB)
    assert wb.iloc[17].tolist() == ["b", "gsadf", pytest.approx(-1.258915002), "99",
                                    pytest.approx(1.157662296)]
    sb = tidy_join(RESULT, SB)
    assert sb["stat"].tolist() == ["gsadf_panel"] * 3
    assert sb["tstat"].iloc[0] == pytest.approx(-0.47777465)
