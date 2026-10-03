"""Common-bubble detection by PCA and PSY (Chen, Phillips & Shi 2023). Ported
from exuber's R/radf_common.R. See docs/multivariate.md for the evaluation
that this code implements.
"""

from dataclasses import dataclass

import numpy as np

from exuber.cv import PCNT, RadfCv
from exuber.radf import RadfResult, _to_2d_array, psy_minw, radf


@dataclass
class RadfCommonResult(RadfResult):
    """RadfResult plus the PCA info radf_common() extracted it from."""

    loadings: np.ndarray | None = None
    explained_variance_ratio: np.ndarray | None = None


def _pca(x: np.ndarray, r: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """First r principal components of x (centered, not scaled) via SVD.
    This is the numerical core of radf_common(). It is a separate function so
    that it can be tested without the compiled _core extension that radf()
    needs. It returns
    (loadings, scores, explained_variance_ratio): loadings is R's
    prcomp()$rotation, scores is prcomp()$x, restricted to r columns.
    """
    xc = x - x.mean(axis=0)
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    r = min(r, vt.shape[0])
    loadings = vt[:r].T
    scores = xc @ loadings
    explained = (s[:r] ** 2) / np.sum(s**2)
    return loadings, scores, explained


def radf_common(data, minw: int | None = None, r: int = 1) -> RadfCommonResult:
    """Common-bubble detection via PCA + PSY (Chen, Phillips & Shi 2023).

    Extracts the panel's first principal component (PCA via SVD on the
    centered panel) and runs the ordinary radf() on its scores. Theorem 4.2 and footnote 3 of
    the paper say that PC1 is "sufficient... for the purpose of bubble
    identification." Setting `r` > 1 only changes how many
    components are kept in `.loadings`/`.explained_variance_ratio` for
    inspection; detection always uses PC1.

    Use radf_common_cv() for the critical values, and not radf_mc_cv().
    radf_mc_cv() does not depend on the panel width N, and independent
    validation found it badly undersized here once N grows past a handful of
    series (docs/multivariate.md, "Independent validation (2026-08-09)").
    """
    x, columns = _to_2d_array(data)
    if x.shape[1] < 2:
        raise ValueError("radf_common needs a panel of at least 2 series.")
    if np.isnan(x).any():
        raise ValueError("radf_common: data contains NaN.")

    loadings, scores, explained = _pca(x, r)
    res = radf(scores[:, 0], minw=minw)
    return RadfCommonResult(
        adf=res.adf,
        badf=res.badf,
        sadf=res.sadf,
        bsadf=res.bsadf,
        gsadf=res.gsadf,
        bsadf_panel=res.bsadf_panel,
        gsadf_panel=res.gsadf_panel,
        minw=res.minw,
        lag=res.lag,
        n=res.n,
        series_names=columns,
        loadings=loadings,
        explained_variance_ratio=explained,
    )


@dataclass
class RadfCommonCv(RadfCv):
    """RadfCv plus the panel width N its null was simulated at."""

    N: int = 0


def radf_common_cv(
    n: int, N: int, minw: int | None = None, nrep: int = 1000, seed: int | None = None
) -> RadfCommonCv:
    """Critical values for radf_common(), simulated under its own null.

    Theorem 4.3 of Chen, Phillips & Shi (2023) claims the PSY-on-PC1
    statistic's limiting null is asymptotically identical to plain
    univariate GSADF's, independent of the panel width N. Independent
    validation found that this does not hold at practical N. The true null
    quantile grows with N, which is the opposite of what a naive reading of
    the finite-sample section of the paper suggests, and it reaches more than
    double the 95% quantile of radf_mc_cv() at N=100. See
    docs/multivariate.md, "Independent validation (2026-08-09)", for the full
    finding.

    Simulates the null radf_common() actually needs: an N-column panel of
    *independent* random walks, which has no true common factor and is the
    sharpest possible null. It extracts PC1 and tests it exactly as
    radf_common() does. `N` must match the panel on which radf_common() was
    actually run, because, unlike radf_mc_cv(), this null distribution
    depends on it.
    """
    if n <= 5:
        raise ValueError("n must be greater than 5")
    if N <= 1:
        raise ValueError("N must be greater than 1")
    if nrep <= 0:
        raise ValueError("nrep must be positive")
    minw = minw if minw is not None else psy_minw(n)
    if minw <= 2:
        raise ValueError("minw must be greater than 2")

    rng = np.random.default_rng(seed)
    n_minw = n - minw

    adf = np.empty(nrep)
    sadf = np.empty(nrep)
    gsadf = np.empty(nrep)
    bsadf = np.empty((n_minw, nrep))

    for i in range(nrep):
        panel = np.cumsum(rng.normal(size=(n, N)), axis=0)
        res = radf_common(panel, minw=minw)
        adf[i] = res.adf[0]
        sadf[i] = res.sadf[0]
        gsadf[i] = res.gsadf[0]
        bsadf[:, i] = res.bsadf[:, 0]

    adf_cv = np.quantile(adf, PCNT)
    sadf_cv = np.quantile(sadf, PCNT)
    gsadf_cv = np.quantile(gsadf, PCNT)
    # This matches R's apply(bsadf_mat, 2, cummax) |> apply(1, quantile). We
    # take the cummax of the simulated bsadf path of each replication and
    # then the quantiles across replications. radf_mc_cv() builds its values
    # differently, taking the cummax of badf. We port the construction of
    # the R source for this function.
    bsadf_cv = np.quantile(np.maximum.accumulate(bsadf, axis=0), PCNT, axis=1).T
    asy_adf_crit = np.array([-0.44, -0.08, 0.6])
    badf_cv = np.tile(asy_adf_crit, (n_minw, 1))

    return RadfCommonCv(
        adf_cv=adf_cv,
        sadf_cv=sadf_cv,
        gsadf_cv=gsadf_cv,
        badf_cv=badf_cv,
        bsadf_cv=bsadf_cv,
        method="Monte Carlo",
        minw=minw,
        n=n,
        iter=nrep,
        N=N,
    )
