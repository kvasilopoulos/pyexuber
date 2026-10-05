# pyexuber

pyexuber detects explosive behaviour (bubbles) in time series. It runs the
recursive right-tailed unit root tests of Phillips, Shi and Yu (2015):
ADF, SADF, GSADF and the backward SADF (BSADF) sequence. A series is
explosive when it grows faster than a random walk, and each of these tests
asks whether the data contain a stretch of such growth.

This is the Python counterpart of the R package
[exuber](https://github.com/kvasilopoulos/exuber). Both packages compute the
statistic with the same C++ core,
[exubercore](https://github.com/kvasilopoulos/exubercore), so they give
identical numbers for the same input.

The package is distributed as `pyexuber` and imported as `exuber`.

```sh
pip install pyexuber
```

## Usage

```python
import exuber
from exuber.sim import sim_psy1

y = sim_psy1(200, seed=1)                 # single-bubble DGP
res = exuber.radf(y)                      # ADF / SADF / GSADF + BSADF sequence
cv = exuber.radf_crit(n=200)              # precomputed Monte Carlo critical values
exuber.datestamp(res, cv)                 # {'series1': [Episode(start=..., peak=..., end=...)]}
```

`radf()` accepts a numpy array, a 1-D sequence, or any object with a
`.to_numpy()` method, such as a pandas or polars DataFrame (neither is a
dependency). Column names become series names. A multi-column input also
runs the panel version of the tests (`bsadf_panel`, `gsadf_panel`).

## What the package contains

| | |
|---|---|
| `radf(data, minw=None, lag=0)` | recursive ADF/SADF/GSADF/BSADF statistics (C++) |
| `radf_crit(n, lag=0)` | precomputed Monte Carlo critical values from the shared store that the R package also reads; fetched once and cached on disk |
| `radf_mc_cv` / `radf_mc_distr` | Monte Carlo critical values and distributions, simulated locally |
| `radf_wb_cv` / `radf_wb_distr` | wild-bootstrap critical values (Harvey, Leybourne, Sollis & Taylor 2016) |
| `datestamp(result, cv, ...)` | start, peak, end and duration of each explosive episode |
| `sim_psy1`, `sim_psy2`, `sim_ps1`, `sim_ps2`, `sim_blan`, `sim_evans`, `sim_div` | bubble DGP simulators |
| `psy_minw`, `psy_ds` | the PSY default minimum window and minimum duration rules |

Beyond this core, the package ports many methods from exuber and the wider
literature: other bootstrap variants (`radf_wb_ps_cv`, `radf_sb_cv`), several
dating and monitoring procedures, and `diagnostics()`, `summary()` and
`tidy()`. The docstring of `exuber/__init__.py` lists them all and says what
is deferred.

## Notes

- **Critical values.** `radf_crit()` covers lags 0 to 4 and sample sizes up
  to 4000. For a combination that has not been simulated it returns `None`,
  and you can use `radf_mc_cv` instead. The store is described at
  [exuber.kvasilopoulos.com](https://exuber.kvasilopoulos.com/).
- **Reproducibility.** The simulators and bootstraps use numpy's `Generator`
  and not R's random number generator. A given `seed` reproduces the same
  draws across Python runs, but not the draws of the R function with the
  same name.
- **Numerics.** The statistic is built by sequential recursive updates, so
  for `lag > 0` the result may differ between compilers at about 1e-12. The
  tests use a tolerance of 1e-9 there. With `lag == 0` the result matches R
  to 1e-12.

## Building from source

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

## License

GPL-3.0-or-later, the same as exuber.
