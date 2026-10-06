# pyexuber

<!-- badges: start -->
[![PyPI](https://img.shields.io/pypi/v/pyexuber.svg)](https://pypi.org/project/pyexuber/)
[![Python](https://img.shields.io/pypi/pyversions/pyexuber.svg)](https://pypi.org/project/pyexuber/)
[![CI](https://github.com/kvasilopoulos/pyexuber/actions/workflows/ci.yml/badge.svg)](https://github.com/kvasilopoulos/pyexuber/actions/workflows/ci.yml)
[![License: GPL v3](https://img.shields.io/badge/license-GPL--3.0--or--later-blue.svg)](LICENSE)
[![JSS](https://img.shields.io/badge/JSS-10.18637%2Fjss.v103.i10-b31b1b.svg)](https://doi.org/10.18637/jss.v103.i10)
<!-- badges: end -->

Testing for and dating periods of explosive dynamics (exuberance) in time
series, in Python. A series is explosive when it grows faster than a random
walk, as asset prices do in a bubble. The package runs the univariate and
panel recursive right-tailed unit root tests of
[Phillips, Shi and Yu (2015)](https://doi.org/10.1111/iere.12132) and
[Pavlidis et al. (2016)](https://doi.org/10.1007/s11146-015-9531-2): ADF,
SADF, GSADF and the backward SADF (BSADF) sequence, then dates each episode
and says how fast it grows.

pyexuber is distributed on PyPI as `pyexuber` and imported as `exuber`. It
is the Python edition of the R package
[exuber](https://github.com/kvasilopoulos/exuber), which is on CRAN, and it
shares the C++ statistic [exubercore](https://github.com/kvasilopoulos/exubercore)
with it. For the same input the two return the same numbers. The simulation,
critical-value and dating code is written in Python and numpy, so it needs no
R installation.

### Overview

Testing for explosive dynamics has two parts:

- Estimation of the test statistics.
- Critical values to compare them with.

Conventional tests take their critical values from a standard distribution.
The statistics used for explosive dynamics follow non-standard
distributions, so the critical values have to be simulated.

#### Estimation

The central function is `radf()`, the recursive augmented Dickey–Fuller
test. It accepts a single series or several, and it also runs the panel
versions of the tests when you pass more than one column. The C++ core uses
the matrix inversion lemma in the recursive least-squares step, so no matrix
is inverted for each window, which makes the tests fast.

#### Critical values

- `radf_crit()`: precomputed Monte Carlo critical values.
- `radf_mc_cv()`: Monte Carlo critical values simulated locally.
- `radf_wb_cv()`, `radf_wb_ps_cv()`: wild bootstrap (Harvey et al. 2016;
  Phillips & Shi 2020).
- `radf_sb_cv()`: sieve bootstrap (panel).

`radf_crit()` reads a shared store that covers `lag = 0` to `4` and every
sample size up to 4000. This is the same store that exuber uses when you
omit `cv` in R. Each `(n, lag)` combination is fetched once and cached on
disk. The simulating functions work offline, and they are the option for
other lags or larger samples.

### Analysis

The analysis needs the output of the estimation (`res`) and the critical
values (`cv`). Small steps break it down:

- `summary(res, cv)` tabulates each statistic against its critical values.
- `diagnostics(res, cv)` shows which series reject the null hypothesis.
- `datestamp(res, cv)` gives the origination, peak, termination and duration
  of each episode.
- `rootstamp(...)` estimates how fast a detected episode grows (the explosive
  root and its doubling time).
- `tidy()` and `augment()` return DataFrames of results, critical values and
  distributions.

### Beyond `radf()`

The recursive ADF test is the core of the package, but not all of it. Each
family below follows a published method, and `docs/parity.md` in the
[project repository](https://github.com/kvasilopoulos/exuber-project) lists
which functions each implementation has.

- Volatility-robust tests (`radf_tt`, `radf_kp`, `radf_sbz`, `radf_sign`,
  `ssu_test`, `cusum_test`) ask the same question as `radf()` when the
  innovation variance changes over time.
- Dating procedures (`dating_hls`, `dating_hlw`, `dating_knp`, `dating_pdc`,
  `radf_recovery`) use regime models to estimate when a bubble that you
  already believe in starts and ends. They need no critical value.
- Real-time monitoring (`monitor`, `monitor_cusum`, `monitor_lbi`,
  `monitor_quantile`) is calibrated on a training window and raises an alarm
  as new observations arrive.
- Multivariate tools (`radf_common`, `cobubble_test`, `contagion_reg`) look
  for shared and transmitted bubbles across series.
- Simulation (`sim_psy1`, `sim_psy2`, `sim_ps1`, `sim_ps2`, `sim_blan`,
  `sim_evans`, `sim_div` and the later additions) provides the bubble
  processes on which the tests are validated.

The functions `radf_tt`, `radf_kp`, `radf_sbz`, `radf_sign`, `ssu_test`,
`cusum_test` and the `sim_*` simulators are imported from their own modules,
for example `from exuber.radf_tt import radf_tt`. The docstring of
`exuber/__init__.py` lists everything and says what is deferred.

### Installation

```sh
pip install pyexuber
```

pandas and polars are optional. `pip install pyexuber[pandas]` adds the
DataFrame outputs of `tidy()` and `summary()`.

If you find a clear bug, please file a reproducible example on
[GitHub](https://github.com/kvasilopoulos/pyexuber/issues).

### Usage

```python
import exuber
from exuber.sim import sim_psy1

y = sim_psy1(200, seed=1)                 # single-bubble DGP
res = exuber.radf(y)                      # ADF / SADF / GSADF + BSADF sequence
cv = exuber.radf_crit(n=200)              # precomputed Monte Carlo critical values

exuber.summary(res, cv)                   # statistics against 90/95/99% values
exuber.diagnostics(res, cv)               # reject or not, per series
exuber.datestamp(res, cv)                 # {'series1': [Episode(start=..., peak=..., end=...)]}
```

`radf()` accepts a numpy array, a 1-D sequence, or any object with a
`.to_numpy()` method, such as a pandas or polars DataFrame. Column names
become series names, and a multi-column input also runs the panel tests
(`bsadf_panel`, `gsadf_panel`).

### Notes

- **Critical values.** For a combination of `lag` and `n` outside the store,
  `radf_crit()` returns `None`. Use `radf_mc_cv()` instead.
- **Reproducibility.** The simulators and bootstraps use numpy's `Generator`
  and not R's random number generator. A given `seed` reproduces the same
  draws across Python runs, but not the draws of the R function with the
  same name.
- **Numerics.** The statistic is built by sequential recursive updates, so
  for `lag > 0` the result may differ between compilers at about 1e-12. The
  tests use a tolerance of 1e-9 there. With `lag == 0` the result matches R
  to 1e-12.

### Building from source

Wheels are published for Linux x86_64, macOS (arm64 and x86_64) and Windows
x86_64. Building from the source distribution needs a C++17 compiler, CMake
3.16 or later, and a system [Armadillo](https://arma.sourceforge.net/)
installation with BLAS and LAPACK. Install it with `apt install
libarmadillo-dev`, `brew install armadillo`, or `vcpkg install armadillo`
on Windows with MSVC. CMake fetches exubercore at a pinned tag during the
build.

```sh
uv sync --dev
uv run pytest
```

### Citation

pyexuber is the product of ongoing research. If it is useful in your own
work, please cite the paper that accompanies exuber, in the *Journal of
Statistical Software*:

> Vasilopoulos, K., Pavlidis, E., & Martínez-García, E. (2022). exuber:
> Recursive Right-Tailed Unit Root Testing with R. *Journal of Statistical
> Software*, 103(10), 1–26.
> [doi:10.18637/jss.v103.i10](https://doi.org/10.18637/jss.v103.i10)

```bibtex
@Article{,
  title = {{exuber}: Recursive Right-Tailed Unit Root Testing with {R}},
  author = {Kostas Vasilopoulos and Efthymios Pavlidis and Enrique Mart{'i}nez-Garc{'i}a},
  journal = {Journal of Statistical Software},
  year = {2022},
  volume = {103},
  number = {10},
  pages = {1--26},
  doi = {10.18637/jss.v103.i10},
}
```

### License

GPL-3.0-or-later, the same as exuber.
