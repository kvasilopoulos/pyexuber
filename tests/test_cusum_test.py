"""Tests for exuber.cusum_test -- Kurozumi & Nishi (2025)'s CUSUM-type
bubble tests. Port of exuber's R/cusum_test.R; R reference values come
from the same deterministic input (test_ssu_test.Y_VEC)."""

import numpy as np
import pytest
from test_ssu_test import Y_VEC

from exuber.cusum_test import cusum_test

R_REF = {
    "cs": (1.7059972635, None),
    "gcs": (2.3702314238, None),
    "cssq": (1.1433215401, -0.3831771014),
    "gcssq": (1.5264986415, -1.1433215401),
}


def _brute(y: np.ndarray, type: str) -> tuple[float, float]:
    d = np.diff(y)
    nd = len(d)
    vals = []
    for k in range(1, nd + 1):
        for j in [0] if type in ("cs", "cssq") else range(k):
            w = d[j:k]
            if type in ("cs", "gcs"):
                vals.append(w.sum() / np.sqrt(np.mean(d**2) * nd))
            else:
                num = np.sum(w**2) - (k - j) / nd * np.sum(d**2)
                vals.append(num / np.sqrt((np.mean(d**4) - np.mean(d**2) ** 2) * nd))
    return max(vals), min(vals)


@pytest.mark.parametrize("type", ["cs", "gcs", "cssq", "gcssq"])
def test_cusum_test_matches_r_and_brute_force(type):
    res = cusum_test(Y_VEC, type=type)
    sup_r, inf_r = R_REF[type]
    assert res.sup[0] == pytest.approx(sup_r, abs=1e-8)
    b_sup, b_inf = _brute(Y_VEC, type)
    assert res.sup[0] == pytest.approx(b_sup, abs=1e-10)
    if inf_r is not None:
        assert res.inf is not None
        assert res.inf[0] == pytest.approx(inf_r, abs=1e-8)
        assert res.inf[0] == pytest.approx(b_inf, abs=1e-10)


def test_cusum_test_critical_values_and_two_sided_rule():
    y = np.cumsum(np.random.default_rng(8).normal(size=100))
    assert cusum_test(y, type="cs").crit == pytest.approx((1.93,))
    sq = cusum_test(y, type="cssq", sig_lvl=90)
    assert sq.crit == pytest.approx((1.19, -1.21))
    assert sq.inf is not None
    assert bool(sq.detected[0]) == bool(sq.sup[0] >= 1.19 or sq.inf[0] <= -1.21)
    with pytest.raises(ValueError):
        cusum_test(y, type="cusum")
    with pytest.raises(ValueError):
        cusum_test(y, sig_lvl=80)


def test_cusum_test_size_under_h0():
    rng = np.random.default_rng(1)
    ys = [np.cumsum(rng.normal(size=200)) for _ in range(200)]
    for type in ("cs", "gcs", "cssq", "gcssq"):
        fa = np.mean([cusum_test(y, type=type).detected[0] for y in ys])
        assert fa < 0.10, type
