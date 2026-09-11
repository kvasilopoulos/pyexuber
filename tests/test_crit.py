import lzma
import struct
import urllib.error

import numpy as np
import pytest

from exuber.crit import parse_crit_bin, radf_crit


def _synthetic(n=20, minw=5, lag=2) -> bytes:
    nrows = n - minw
    body = struct.pack("<4i", n, minw, lag, nrows)
    body += struct.pack("<9d", *range(1, 10))
    body += np.arange(1, nrows * 3 + 1, dtype="<f8").tobytes()
    return lzma.compress(body)


def test_parse_crit_bin_round_trip():
    cv = parse_crit_bin(_synthetic())
    assert (cv.n, cv.minw, cv.lag) == (20, 5, 2)
    np.testing.assert_array_equal(cv.adf_cv, [1, 2, 3])
    np.testing.assert_array_equal(cv.gsadf_cv, [7, 8, 9])
    assert cv.bsadf_cv.shape == (15, 3)
    np.testing.assert_array_equal(cv.bsadf_cv[1], [4, 5, 6])
    np.testing.assert_allclose(cv.badf_cv[0], [-0.44, -0.08, 0.6])


def test_parse_crit_bin_rejects_truncated():
    with pytest.raises(ValueError):
        parse_crit_bin(lzma.compress(lzma.decompress(_synthetic())[:-8]))


def test_radf_crit_unreachable_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    with pytest.raises(urllib.error.URLError):
        radf_crit(9999, lag=4, base_url="http://127.0.0.1:9/crit2")


@pytest.mark.network
def test_radf_crit_live(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    cv = radf_crit(700)
    assert cv is not None and cv.n == 700 and cv.lag == 0
    assert cv.bsadf_cv.shape == (700 - cv.minw, 3)
    assert (tmp_path / "exuber" / "lag0-n700.bin.xz").exists()
    assert radf_crit(4999, lag=4) is None
