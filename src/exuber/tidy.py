"""Tidy accessors. Ports exuber's R/radf-tidiers.R: tidy()/augment() for
radf_obj and radf_cv, tidy() for radf_distr, tidy_join() and
augment_join().

tidy() dispatches on its argument's type like R's S3 generic. Monte
Carlo vs wild bootstrap is told apart by shape (shared (3,) critical
values vs per-series (nc, 3)), not by class, since both are RadfCv here.

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

from exuber.cv import RadfCv, RadfDistr, RadfSbCv, RadfSbDistr

if TYPE_CHECKING:
    import pandas as pd

    from exuber.radf import RadfResult

SIGS = ("90", "95", "99")
STATS = ("adf", "sadf", "gsadf")


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError(
            "tidy()/augment() need pandas: pip install pyexuber[pandas]"
        ) from e
    return pd


def _series_names(result, nc: int | None = None) -> list[str]:
    nc = len(result.adf) if nc is None else nc
    return result.series_names or [f"series{i + 1}" for i in range(nc)]


def tidy(result, format: str = "wide", panel: bool = False) -> "pd.DataFrame":
    """Port of R's tidy(). For a RadfResult (tidy.radf_obj): the scalar
    adf/sadf/gsadf statistics, one row per series (`format="wide"`) or per
    series-statistic pair (`format="long"`); `panel=True` returns the
    panel gsadf_panel statistic instead. For a RadfCv/RadfSbCv
    (tidy.radf_cv): the 90/95/99% critical values. For a
    RadfDistr/RadfSbDistr (tidy.radf_distr): the simulated statistics,
    one row per replication (`format` is ignored)."""
    if format not in ("wide", "long"):
        raise ValueError('format must be "wide" or "long"')
    pd = _require_pandas()
    if isinstance(result, (RadfCv, RadfSbCv)):
        return _tidy_cv(pd, result, format)
    if isinstance(result, (RadfDistr, RadfSbDistr)):
        return _tidy_distr(pd, result)

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
    result: "RadfResult | RadfCv | RadfSbCv",
    format: str = "wide",
    panel: bool = False,
    trunc: bool = True,
) -> "pd.DataFrame":
    """Port of R's augment.radf_obj(): the full badf/bsadf test-statistic
    sequences, one row per observation (`key`, 1-indexed like R) per
    series. `trunc=True` (R's default) drops the pre-minw+lag rows where
    badf/bsadf aren't defined; `trunc=False` keeps them as NaN, aligned
    to the full 1..n range. `panel=True` returns the panel bsadf_panel
    sequence instead (one row per key, no `id` split).

    For a RadfCv/RadfSbCv (augment.radf_cv): the badf/bsadf critical-value
    sequences (bsadf_panel for a sieve-bootstrap cv), one row per key and
    significance level. No `index` column, as above."""
    if format not in ("wide", "long"):
        raise ValueError('format must be "wide" or "long"')
    pd = _require_pandas()
    if isinstance(result, (RadfCv, RadfSbCv)):
        return _augment_cv(pd, result, format, trunc)

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


def _tidy_cv(pd, cv: "RadfCv | RadfSbCv", format: str) -> "pd.DataFrame":
    if isinstance(cv, RadfSbCv):
        df = pd.DataFrame({"id": "panel", "sig": SIGS, "gsadf_panel": cv.gsadf_panel_cv})
        if format == "long":
            df = df.assign(stat="gsadf_panel").rename(columns={"gsadf_panel": "crit"})
            df = df[["id", "stat", "sig", "crit"]]
        return df

    arrs = {stat: np.asarray(getattr(cv, f"{stat}_cv")) for stat in STATS}
    if arrs["adf"].ndim == 1:  # Monte Carlo: shared across series
        if format == "wide":
            return pd.DataFrame({"sig": SIGS, **arrs})
        return pd.DataFrame(
            [(stat, sig, arrs[stat][k]) for stat in STATS for k, sig in enumerate(SIGS)],
            columns=["stat", "sig", "crit"],
        )

    names = _series_names(cv, arrs["adf"].shape[0])  # bootstrap: (nc, 3), per series
    if format == "wide":
        return pd.DataFrame(
            [(name, sig, *(arrs[stat][j, k] for stat in STATS))
             for k, sig in enumerate(SIGS) for j, name in enumerate(names)],
            columns=["id", "sig", *STATS],
        )
    return pd.DataFrame(
        [(name, stat, sig, arrs[stat][j, k])
         for stat in STATS for k, sig in enumerate(SIGS) for j, name in enumerate(names)],
        columns=["id", "stat", "sig", "crit"],
    )


def _tidy_distr(pd, distr: "RadfDistr | RadfSbDistr") -> "pd.DataFrame":
    if isinstance(distr, RadfSbDistr):
        return pd.DataFrame({"gsadf_panel": distr.gsadf_panel_distr})
    arrs = {stat: np.asarray(getattr(distr, f"{stat}_distr")) for stat in STATS}
    if arrs["adf"].ndim == 1:  # Monte Carlo
        return pd.DataFrame(arrs)
    nrep, nc = arrs["adf"].shape  # bootstrap: one column per series, stacked
    names = [f"series{i + 1}" for i in range(nc)]
    return pd.DataFrame(
        {"id": np.repeat(names, nrep), **{stat: a.T.ravel() for stat, a in arrs.items()}}
    )


def _augment_cv(pd, cv: "RadfCv | RadfSbCv", format: str, trunc: bool) -> "pd.DataFrame":
    if isinstance(cv, RadfSbCv):
        seqs = {"bsadf_panel": np.asarray(cv.bsadf_panel_cv)[:, :, None]}
        names = ["panel"]
    else:
        seqs = {"badf": np.asarray(cv.badf_cv), "bsadf": np.asarray(cv.bsadf_cv)}
        names: list[str] = []
    first = next(iter(seqs.values()))
    rows_cv = first.shape[0]
    offset = cv.n - rows_cv  # minw (+ lag): keys before it have no critical value
    keys = np.arange(1, cv.n + 1) if not trunc else np.arange(offset + 1, cv.n + 1)

    def value(arr, k, s, j):
        r = k - offset - 1
        if r < 0:
            return np.nan
        return arr[r, s] if arr.ndim == 2 else arr[r, s, j]

    per_series = first.ndim == 3
    if per_series and not names:
        names = _series_names(cv, first.shape[2])
    stats = list(seqs)
    if not per_series:  # Monte Carlo: shared across series, no id column
        df = pd.DataFrame(
            [(k, sig, *(value(seqs[st], k, s, 0) for st in stats))
             for k in keys for s, sig in enumerate(SIGS)],
            columns=["key", "sig", *stats],
        )
        if format == "long":
            df = df.melt(id_vars=["key", "sig"], value_vars=stats, var_name="stat",
                         value_name="crit")[["key", "stat", "sig", "crit"]]
        return df

    df = pd.DataFrame(
        [(k, name, sig, *(value(seqs[st], k, s, j) for st in stats))
         for j, name in enumerate(names) for k in keys for s, sig in enumerate(SIGS)],
        columns=["key", "id", "sig", *stats],
    )
    if format == "long":
        df = pd.DataFrame(
            [(r.key, r.id, st, r.sig, getattr(r, st)) for r in df.itertuples() for st in stats],
            columns=["key", "id", "stat", "sig", "crit"],
        )
    elif isinstance(cv, RadfSbCv):
        df = df.drop(columns="id")  # R: the id column only appears in the long form
    return df


def augment_join(
    result: "RadfResult", cv: "RadfCv | RadfSbCv", trunc: bool = True
) -> "pd.DataFrame":
    """Port of R's augment_join.radf_obj(): the badf/bsadf statistic
    sequences next to their critical-value sequences, one row per
    key-series-statistic-significance (key, id, stat, tstat, sig, crit) --
    the table autoplot() is built on. With a sieve-bootstrap cv, the panel
    bsadf_panel sequence only. Like augment(), no `index`/`data` columns."""
    pd = _require_pandas()
    panel = isinstance(cv, RadfSbCv)
    x = augment(result, "long", panel=panel, trunc=trunc)
    y = augment(cv, "long", trunc=trunc)
    by = ["key", "stat"] if "id" not in y.columns else ["key", "id", "stat"]
    x["stat"] = x["stat"].astype(str)
    df = x.merge(y, on=by, how="outer")
    ids = ["panel"] if panel else _series_names(result)
    df["id"] = pd.Categorical(df["id"], categories=ids)
    df = df.sort_values(["id", "stat", "sig", "key"], kind="stable").reset_index(drop=True)
    df["id"] = df["id"].astype(str)
    return df[["key", "id", "stat", "tstat", "sig", "crit"]]


def tidy_join(result: "RadfResult", cv: "RadfCv | RadfSbCv") -> "pd.DataFrame":
    """Port of R's tidy_join.radf_obj(): each statistic next to its
    critical values, one row per series-statistic-significance triple
    (id, stat, tstat, sig, crit). With a sieve-bootstrap cv, the panel
    gsadf_panel statistic only."""
    pd = _require_pandas()
    cols = ["id", "stat", "tstat", "sig", "crit"]
    if isinstance(cv, RadfSbCv):
        return pd.DataFrame(
            [("panel", "gsadf_panel", result.gsadf_panel, sig, cv.gsadf_panel_cv[k])
             for k, sig in enumerate(SIGS)],
            columns=cols,
        )
    rows = []
    for j, name in enumerate(_series_names(result)):
        for stat in STATS:
            crit = np.asarray(getattr(cv, f"{stat}_cv"))
            crit = crit if crit.ndim == 1 else crit[j]
            tstat = getattr(result, stat)[j]
            rows += [(name, stat, tstat, sig, crit[k]) for k, sig in enumerate(SIGS)]
    return pd.DataFrame(rows, columns=cols)
