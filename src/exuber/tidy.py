"""Tidy accessors for RadfResult. Ports exuber's R/radf-tidiers.R
tidy.radf_obj()/augment.radf_obj() (the radf_obj methods only -- the
radf_cv/radf_distr tidiers, tidy_join()/augment_join(), and
summary()/diagnostics() built on top of them are not ported here, see
__init__.py's module docstring).

Divergence from R: RadfResult doesn't carry the original input data or a
date index (radf() never stored either -- see radf.py), so augment()'s
table has no `data`/`index` columns, unlike R's augment.radf_obj(). Row
selection, column names and ordering otherwise match R exactly (verified
against R's own tidy()/augment() output structure -- see
docs/replication/core-workflow/tidy_validation.py).

Requires pandas (an optional extra, `pip install pyexuber[pandas]`) --
lazily imported, like radf()'s C++ extension.
"""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

    from exuber.radf import RadfResult


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError(
            "tidy()/augment() need pandas: pip install pyexuber[pandas]"
        ) from e
    return pd


def _series_names(result: "RadfResult") -> list[str]:
    return result.series_names or [f"series{i + 1}" for i in range(len(result.adf))]


def tidy(result: "RadfResult", format: str = "wide", panel: bool = False) -> "pd.DataFrame":
    """Port of R's tidy.radf_obj(): the scalar adf/sadf/gsadf test
    statistics as a tidy DataFrame, one row per series (`format="wide"`)
    or one row per series-statistic pair (`format="long"`). `panel=True`
    returns the single panel gsadf_panel statistic instead."""
    if format not in ("wide", "long"):
        raise ValueError('format must be "wide" or "long"')
    pd = _require_pandas()

    if panel:
        if format == "wide":
            return pd.DataFrame({"gsadf_panel": [result.gsadf_panel]})
        return pd.DataFrame(
            {"id": ["panel"], "stat": ["gsadf_panel"], "tstat": [result.gsadf_panel]}
        )

    names = _series_names(result)
    if format == "wide":
        return pd.DataFrame(
            {"id": names, "adf": result.adf, "sadf": result.sadf, "gsadf": result.gsadf}
        )

    rows = [
        {"id": name, "stat": stat, "tstat": value}
        for name, a, s, g in zip(names, result.adf, result.sadf, result.gsadf, strict=True)
        for stat, value in (("adf", a), ("sadf", s), ("gsadf", g))
    ]
    return pd.DataFrame(rows)


def augment(
    result: "RadfResult", format: str = "wide", panel: bool = False, trunc: bool = True
) -> "pd.DataFrame":
    """Port of R's augment.radf_obj(): the full badf/bsadf test-statistic
    sequences, one row per observation (`key`, 1-indexed like R) per
    series. `trunc=True` (R's default) drops the pre-minw+lag rows where
    badf/bsadf aren't defined; `trunc=False` keeps them as NaN, aligned
    to the full 1..n range. `panel=True` returns the panel bsadf_panel
    sequence instead (one row per key, no `id` split)."""
    if format not in ("wide", "long"):
        raise ValueError('format must be "wide" or "long"')
    pd = _require_pandas()

    trunc_key = result.minw + result.lag
    pointer = result.n - trunc_key
    keys = np.arange(trunc_key + 1, result.n + 1)  # matches R's 1-indexed key

    if panel:
        bsadf_panel = np.full(result.n, np.nan)
        bsadf_panel[trunc_key:] = result.bsadf_panel
        df = pd.DataFrame({"key": np.arange(1, result.n + 1), "bsadf_panel": bsadf_panel})
        if trunc:
            df = df[df["key"] > trunc_key].reset_index(drop=True)
        if format == "long":
            df = df.assign(id="panel", stat="bsadf_panel").rename(columns={"bsadf_panel": "tstat"})
            df = df[["key", "id", "stat", "tstat"]]
        return df

    names = _series_names(result)
    if trunc:
        rows = [
            {
                "key": int(keys[r]),
                "id": name,
                "badf": result.badf[r, j],
                "bsadf": result.bsadf[r, j],
            }
            for r in range(pointer)
            for j, name in enumerate(names)
        ]
    else:
        rows = []
        for k in range(1, result.n + 1):
            r = k - trunc_key - 1  # row into badf/bsadf, valid for r >= 0
            for j, name in enumerate(names):
                badf_val = result.badf[r, j] if r >= 0 else np.nan
                bsadf_val = result.bsadf[r, j] if r >= 0 else np.nan
                rows.append({"key": k, "id": name, "badf": badf_val, "bsadf": bsadf_val})

    df = pd.DataFrame(rows)
    if format == "long":
        df = df.melt(
            id_vars=["key", "id"], value_vars=["badf", "bsadf"], var_name="stat", value_name="tstat"
        )
        df["stat"] = pd.Categorical(df["stat"], categories=["badf", "bsadf"])
        df = df.sort_values(["key", "id", "stat"], kind="stable").reset_index(drop=True)
    return df
