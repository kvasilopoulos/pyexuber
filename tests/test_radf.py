from pathlib import Path

import numpy as np
import pytest

from exuber import _core

FIXTURES = Path(__file__).parent / "fixtures" / "golden"

CASES = ["small_lag0", "small_lag1", "medium_lag0", "medium_lag2"]


def _load(name: str):
    yxmat = np.loadtxt(FIXTURES / f"{name}_input.csv", delimiter=",", skiprows=1)
    min_win, lag, _n_rows, _n_cols = np.loadtxt(
        FIXTURES / f"{name}_params.csv", delimiter=",", skiprows=1
    )
    expected = np.loadtxt(FIXTURES / f"{name}_output.csv", delimiter=",", skiprows=1)
    return yxmat, int(min_win), int(lag), expected


@pytest.mark.parametrize("name", CASES)
def test_radf_stat_matches_golden_fixture(name):
    yxmat, min_win, lag, expected = _load(name)
    actual = _core.radf_stat(yxmat, min_win, lag)

    # Same cross-toolchain floating-point tolerance rationale as
    # exubercore's own test_radf.cpp: closed-form lag==0 path matches
    # tightly, the O(n^2) sequential lag>0 path gets a looser bound.
    tol = 1e-9 if lag > 0 else 1e-12
    np.testing.assert_allclose(actual, expected, atol=tol, equal_nan=True)


def test_radf_stat_rejects_bad_min_win():
    yxmat, _min_win, lag, _expected = _load("small_lag0")
    with pytest.raises(ValueError):
        _core.radf_stat(yxmat, 1, lag)
