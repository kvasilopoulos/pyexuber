# Changelog

## Unreleased

- `monitor_quantile(boundary="bootstrap")` implements Algorithm 1 of Wu, Shi and Wu (2025) for the whole QPWY or QPSY path. It resamples the centred first differences, cumulates them, recomputes the statistic path and takes the quantile of the path maxima. It corrects most of the oversizing of the asymptotic boundary away from the median. Each replicate costs a full statistic path, so QPSY is slow with the IRLS solver. The result has a new `boundary_type` field, and the asymptotic caveat warning now points to the bootstrap.

## 0.1.0 (2026-10-05)

First PyPI release.

- `radf()` computes the recursive ADF, SADF, GSADF and BSADF statistics for
  univariate and panel data, using exubercore v0.3.1 (C++).
- `radf_crit()` returns precomputed Monte Carlo critical values from the
  shared exuber store (lags 0 to 4, n up to 4000) and caches them on disk.
- `radf_mc_cv` and `radf_mc_distr` simulate critical values and
  distributions locally, as do `radf_wb_cv` and `radf_wb_distr` for the
  Harvey, Leybourne, Sollis and Taylor (HLST) wild bootstrap.
- `datestamp()` reports the start, peak, end and duration of each explosive
  episode.
- `monitor(boundary="bootstrap")` uses the wild-bootstrap monitoring
  boundary of Phillips and Shi (2020), computed by `radf_wb_ps_cv()` on the
  training window.
- `dating_hlw(join=3)` applies the run-joining rule of Harvey, Leybourne and
  Whitehouse (HLW) to fragmented step-1 detections. It matches `dating_hlw()` in
  exuber.
- `sim_tree`, `sim_mar`, `sim_common`, `sim_coexplosive`, `sim_msbubble` and
  `sim_falsebubble` (in `exuber.sim`) complete the set of bubble DGPs added
  in 2026-08. Extra R attributes come back through `return_*` flags, and
  multi-series DGPs return 2-D arrays.
- `diagnostics()` and `summary()` give a reject or not-reject verdict for
  each series and a table of statistics against critical values. They are
  ported from R's `diagnostics()` and `summary()` for `radf_obj`, and we
  checked them against R's output.
- `tidy()` works on critical values and distributions (`RadfCv`,
  `RadfSbCv`, `RadfDistr`, `RadfSbDistr`), and `augment()` on critical
  values. `tidy_join()` and `augment_join()` follow R's row and column
  layout.
- `ssu_test(type="gssu", union=True)` and `cusum_test()` implement the GSSU,
  the UR/GUR union of rejections, and the CS, GCS, CSSQ and GCSSQ tests of
  Kurozumi and Nishi (2025). All use the critical values in Table I of the
  paper.
- `dating_knp(breaks=m)` implements the multi-bubble dynamic programme of
  Kejriwal, Nguyen and Perron (2025). It finds the exact global minimiser in
  O(m n^2) time.
- `monitor_quantile(type="qpsy")` adds the QPSY monitor of Wu, Shi and Wu
  (2025) next to QPWY. It uses the asymptotic boundary, which is oversized
  away from the median in small samples, and issues a `UserWarning` for
  QPSY to say so.
- `sim_psy1`, `sim_psy2`, `sim_ps1`, `sim_ps2`, `sim_blan`, `sim_evans` and
  `sim_div` simulate bubble DGPs.
- The package requires Python 3.10 or later. Wheels cover Linux x86_64,
  macOS 15 or later (arm64 and x86_64) and Windows x86_64. The source
  distribution builds against a system Armadillo.
