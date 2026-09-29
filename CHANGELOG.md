# Changelog

## 0.1.0 (unreleased)

First PyPI release.

- `radf()`: recursive ADF/SADF/GSADF/BSADF statistics via exubercore v0.3.1
  (C++), univariate and panel.
- `radf_crit()`: precomputed Monte Carlo critical values from the shared
  exuber store (lag 0-4, n <= 4000), disk-cached.
- `radf_mc_cv`/`radf_mc_distr`, `radf_wb_cv`/`radf_wb_distr` (HLST wild
  bootstrap): locally simulated critical values and distributions.
- `datestamp()`: explosive-episode start/peak/end/duration.
- `monitor(boundary="bootstrap")`: Phillips & Shi (2020)'s wild-bootstrap
  monitoring boundary, via `radf_wb_ps_cv()` on the training window.
- `diagnostics()`/`summary()`: per-series reject/not-reject verdict and the
  statistic-vs-critical-value table, ported from R's `diagnostics()`/
  `summary()` for `radf_obj` (verified against R's output).
- `tidy()` for critical values and distributions (`RadfCv`, `RadfSbCv`,
  `RadfDistr`, `RadfSbDistr`), `augment()` for critical values,
  `tidy_join()` and `augment_join()`, matching R's row/column layout.
- `sim_psy1`, `sim_psy2`, `sim_ps1`, `sim_ps2`, `sim_blan`, `sim_evans`,
  `sim_div`: bubble DGP simulators.
- Requires Python >= 3.10. Wheels for Linux x86_64, macOS 15+ (arm64 and
  x86_64), Windows x86_64; sdist builds against a system Armadillo.
