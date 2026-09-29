"""diagnostics()/summary() are pure comparisons (no RNG, no _core call).
Reference numbers are R's own output (exuber 2.0.0 dev): set.seed(5);
dta <- data.frame(a=cumsum(rnorm(30)), b=cumsum(rnorm(30)));
r <- radf(dta, minw=10, lag=1); cv <- radf_mc_cv(30, minw=10, nrep=300, seed=123);
diagnostics(r, cv, option, sig_lvl) / summary(r, cv)."""

import numpy as np
import pytest

from exuber.cv import RadfCv, RadfSbCv
from exuber.diagnostics import diagnostics, summary
from exuber.radf import RadfResult

RESULT = RadfResult(
    adf=np.array([-1.588393268, -4.225519136]),
    badf=np.zeros((19, 2)),
    sadf=np.array([0.8868840249, -2.6798471430]),
    bsadf=np.zeros((19, 2)),
    gsadf=np.array([1.061278277, -1.258915002]),
    bsadf_panel=np.zeros(19),
    gsadf_panel=-0.47777465,
    minw=10,
    lag=1,
    n=30,
    series_names=["a", "b"],
)
MC = RadfCv(
    adf_cv=np.array([-0.26111718664, -0.01850857045, 0.66709085216]),
    sadf_cv=np.array([0.8154876151, 1.3020690630, 1.9061565892]),
    gsadf_cv=np.array([1.457097501, 1.800471790, 2.500302695]),
    badf_cv=np.zeros((19, 3)),
    bsadf_cv=np.zeros((19, 3)),
    method="Monte Carlo",
    minw=10,
    n=30,
    iter=300,
)


@pytest.mark.parametrize(
    ("option", "sig_lvl", "positive", "dummy"),
    [
        ("gsadf", 90, [], [0, 0]),
        ("gsadf", 95, [], [0, 0]),
        ("sadf", 90, ["a"], [1, 0]),
        ("sadf", 95, [], [0, 0]),
        ("sadf", 99, [], [0, 0]),
    ],
)
def test_diagnostics_matches_r(option, sig_lvl, positive, dummy):
    dg = diagnostics(RESULT, MC, option=option, sig_lvl=sig_lvl)
    assert dg.positive == positive
    assert dg.dummy.tolist() == dummy
    # R: "Reject" (= not rejected) -> None here
    assert dg.sig == (["10%", None] if option == "sadf" else [None, None])


def test_diagnostics_per_series_cv():
    wb = RadfCv(**{**MC.__dict__, "gsadf_cv": np.array([[0.5, 0.9, 1.0], [-2.0, -1.5, -1.0]])})
    dg = diagnostics(RESULT, wb)
    assert dg.positive == ["a", "b"]
    assert dg.sig == ["1%", "5%"]


def test_diagnostics_panel_sb_cv():
    sb = RadfSbCv(
        gsadf_panel_cv=np.array([-1.0, -0.5, 0.0]), bsadf_panel_cv=np.zeros((19, 3)),
        method="Sieve Bootstrap", minw=10, n=30, iter=100, lag=1,
    )
    dg = diagnostics(RESULT, sb)
    assert (dg.panel, dg.option, dg.positive, dg.sig) == (True, "gsadf_panel", ["panel"], ["5%"])
    with pytest.raises(ValueError, match="sadf"):
        diagnostics(RESULT, sb, option="sadf")


def test_diagnostics_rejects_bad_input():
    with pytest.raises(ValueError, match="sig_lvl"):
        diagnostics(RESULT, MC, sig_lvl=80)
    with pytest.raises(ValueError, match="n/minw"):
        diagnostics(RESULT, RadfCv(**{**MC.__dict__, "n": 31}))


def test_summary_matches_r():
    pytest.importorskip("pandas")
    sm = summary(RESULT, MC)
    assert list(sm) == ["a", "b"]
    a = sm["a"]
    assert a.columns.tolist() == ["stat", "tstat", "90", "95", "99"]
    assert a["stat"].tolist() == ["adf", "sadf", "gsadf"]
    np.testing.assert_allclose(a["tstat"], [-1.588393268, 0.8868840249, 1.061278277])
    np.testing.assert_allclose(sm["b"]["95"], [-0.01850857045, 1.3020690630, 1.800471790])
