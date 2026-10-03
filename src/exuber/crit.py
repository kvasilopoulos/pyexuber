"""Precomputed Monte Carlo critical values from the shared exuber store.

The store is the same object store that exuber's R client
(exuber/R/crit-bucket.R) reads. It holds one small xz-compressed binary file
for each (lag, n), served read-only by the Railway proxy
(exubercrit/scripts/exuber-fn.ts). Fetched tables are cached on disk, so a
given (n, lag) is downloaded once per machine.

Binary layout (little-endian): four int32 values (n, minw, lag, nrows);
three float64 values each for adf, sadf and gsadf (90, 95 and 99%); then
nrows*3 float64 values for bsadf in row-major order. badf_cv is not stored,
because it is the constant PWY asymptotic tiling.
"""

import lzma
import os
import struct
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from exuber.cv import RadfCv

CRIT_BASE_URL = "https://exuber.up.railway.app/crit2"
_ASY_ADF = np.array([-0.44, -0.08, 0.6])


def _cache_dir() -> Path:
    root = os.environ.get("XDG_CACHE_HOME") or os.environ.get("LOCALAPPDATA") or "~/.cache"
    d = Path(root).expanduser() / "exuber"
    d.mkdir(parents=True, exist_ok=True)
    return d


def parse_crit_bin(raw: bytes) -> RadfCv:
    """Decode one xz-compressed critical-value table (see module docstring)."""
    buf = lzma.decompress(raw)
    n, minw, lag, nrows = struct.unpack_from("<4i", buf)
    body = np.frombuffer(buf, dtype="<f8", offset=16)
    if body.size != 9 + nrows * 3:
        raise ValueError(
            f"crit table for n={n} lag={lag}: expected {9 + nrows * 3} doubles, got {body.size}"
        )
    return RadfCv(
        adf_cv=body[0:3].copy(),
        sadf_cv=body[3:6].copy(),
        gsadf_cv=body[6:9].copy(),
        badf_cv=np.tile(_ASY_ADF, (nrows, 1)),
        bsadf_cv=body[9:].reshape(nrows, 3).copy(),
        method="Monte Carlo", minw=minw, n=n, iter=2000, lag=lag,
    )


def radf_crit(n: int, lag: int = 0, base_url: str = CRIT_BASE_URL) -> RadfCv | None:
    """Fetch precomputed critical values for (n, lag), disk-cached.

    Returns None if the store answers 404 (that combination hasn't been
    simulated yet); raises URLError/HTTPError for anything else so a
    transient network failure isn't mistaken for a missing table.
    """
    path = _cache_dir() / f"lag{lag}-n{n}.bin.xz"
    if path.exists():
        return parse_crit_bin(path.read_bytes())
    try:
        with urllib.request.urlopen(f"{base_url}/{lag}/{n}", timeout=30) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    cv = parse_crit_bin(raw)  # validate before caching
    path.write_bytes(raw)
    return cv
