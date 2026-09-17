"""tidy()/augment() are pure reshaping (no RNG, no _core call) -- checked
bit-for-bit against R here. RadfResult is built by hand from R's own
radf() output rather than calling pyexuber's radf() (which needs the C++
extension, unbuildable on this dev machine -- see
docs/replication/core-workflow/tidy_validation.py, which this test
module also wires into pytest."""

import numpy as np
import pandas as pd

from exuber.radf import RadfResult
from exuber.tidy import augment, tidy

# Same reference RadfResult as tidy_validation.py (R: set.seed(5); dta <-
# data.frame(a=cumsum(rnorm(30)), b=cumsum(rnorm(30))); radf(dta, minw=10, lag=1)).
BADF_A = np.array(
    [-2.46494420, -2.55609820, -2.16484802, -0.91445335, -0.70673367, -0.29629491,
     0.88688402, 0.54431127, 0.28992220, -0.42269212, -0.99670492, -1.42628725,
     -1.54181668, -1.56355826, -1.62076636, -1.40214773, -1.17097835, -1.38785977, -1.58839327]
)
BADF_B = np.array(
    [-2.67984714, -2.89097940, -3.01985212, -3.04928798, -3.01988765, -3.20036243,
     -3.33449015, -3.33320416, -3.42786100, -3.67806731, -3.75406292, -3.61676812,
     -3.85527436, -3.86434474, -4.02064720, -3.98642045, -4.08715517, -4.17231099, -4.22551914]
)
BSADF_A = np.array(
    [-2.46494420, -2.55609820, -2.16484802, -0.68907303, -0.67193106, -0.23548812,
     1.06127828, 0.55655744, 0.31930170, -0.36296915, -0.85058297, -1.21636247,
     -1.29172845, -0.85233681, -1.19198336, -0.20590389, 0.51885401, -0.32902425, -0.98206675]
)
BSADF_B = np.array(
    [-2.67984714, -2.12436757, -2.38926579, -2.20851663, -2.19738005, -2.32439711,
     -2.44772642, -1.78353260, -1.79480972, -1.65031186, -1.75241119, -1.25891500,
     -1.33694666, -1.36992530, -1.67054688, -1.53454847, -1.47440332, -1.59087090, -1.63594756]
)
BSADF_PANEL = np.array(
    [-2.57239567, -2.34023288, -2.27705690, -1.44879483, -1.43465556, -1.27994262,
     -0.69322407, -0.61348758, -0.73775401, -1.00664051, -1.30149708, -1.23763873,
     -1.31433756, -1.11113106, -1.43126512, -0.87022618, -0.47777465, -0.95994758, -1.30900716]
)

RESULT = RadfResult(
    adf=np.array([-1.58839327, -4.22551914]),
    badf=np.column_stack([BADF_A, BADF_B]),
    sadf=np.array([0.88688402, -2.67984714]),
    bsadf=np.column_stack([BSADF_A, BSADF_B]),
    gsadf=np.array([1.06127828, -1.25891500]),
    bsadf_panel=BSADF_PANEL,
    gsadf_panel=-0.47777465,
    minw=10,
    lag=1,
    n=30,
    series_names=["a", "b"],
)


def test_tidy_wide_matches_r():
    df = tidy(RESULT)
    assert list(df.columns) == ["id", "adf", "sadf", "gsadf"]
    assert list(df["id"]) == ["a", "b"]
    np.testing.assert_allclose(df["adf"], [-1.58839327, -4.22551914], atol=1e-6)
    np.testing.assert_allclose(df["gsadf"], [1.06127828, -1.25891500], atol=1e-6)


def test_tidy_long_matches_r():
    df = tidy(RESULT, format="long")
    assert list(df["id"]) == ["a", "a", "a", "b", "b", "b"]
    assert list(df["stat"]) == ["adf", "sadf", "gsadf", "adf", "sadf", "gsadf"]


def test_tidy_panel_matches_r():
    wide = tidy(RESULT, panel=True)
    assert list(wide.columns) == ["gsadf_panel"]
    np.testing.assert_allclose(wide["gsadf_panel"], [-0.47777465], atol=1e-6)

    long = tidy(RESULT, panel=True, format="long")
    assert long["id"].iloc[0] == "panel"
    assert long["stat"].iloc[0] == "gsadf_panel"


def test_tidy_rejects_bad_format():
    import pytest

    with pytest.raises(ValueError):
        tidy(RESULT, format="bogus")


def test_augment_wide_matches_r():
    df = augment(RESULT)
    pointer = RESULT.n - RESULT.minw - RESULT.lag  # 19
    assert len(df) == pointer * 2
    row0, row1 = df.iloc[0], df.iloc[1]
    assert (row0["key"], row0["id"]) == (12, "a")
    assert (row1["key"], row1["id"]) == (12, "b")
    np.testing.assert_allclose([row0["badf"], row0["bsadf"]], [-2.46494420, -2.46494420], atol=1e-6)


def test_augment_long_matches_r():
    df = augment(RESULT, format="long")
    pointer = RESULT.n - RESULT.minw - RESULT.lag
    assert len(df) == pointer * 2 * 2
    first4 = df.iloc[:4]
    assert list(zip(first4["key"], first4["id"], first4["stat"], strict=True)) == [
        (12, "a", "badf"), (12, "a", "bsadf"), (12, "b", "badf"), (12, "b", "bsadf"),
    ]


def test_augment_panel_matches_r():
    df = augment(RESULT, panel=True)
    assert list(df.columns) == ["key", "bsadf_panel"]
    assert len(df) == RESULT.n - RESULT.minw - RESULT.lag
    np.testing.assert_allclose(
        df["bsadf_panel"].iloc[:3], [-2.57239567, -2.34023288, -2.27705690], atol=1e-6
    )


def test_augment_trunc_false_pads_with_nan():
    df = augment(RESULT, trunc=False)
    assert len(df) == RESULT.n * 2
    assert df.iloc[0]["key"] == 1
    assert pd.isna(df.iloc[0]["badf"])
    first_defined = df[df["badf"].notna()].iloc[0]
    assert first_defined["key"] == 12
