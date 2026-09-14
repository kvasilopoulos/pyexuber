# pyexuber

Recursive right-tailed unit root tests (ADF/SADF/GSADF/BSADF, Phillips,
Shi & Yu 2015) for detecting explosive dynamics -- bubbles -- in time
series. The Python counterpart of the R package
[exuber](https://github.com/kvasilopoulos/exuber): the statistic is
computed by the same C++ core, [exubercore](https://github.com/kvasilopoulos/exubercore),
so both packages produce identical numbers for the same input.

Distributed as `pyexuber`, imported as `exuber`.

```sh
pip install pyexuber
```

## Usage

```python
import exuber

y = exuber.sim_psy1(200, seed=1)          # single-bubble DGP
res = exuber.radf(y)                      # ADF / SADF / GSADF + BSADF sequence
cv = exuber.radf_crit(n=200)              # precomputed Monte Carlo critical values
exuber.datestamp(res, cv)                 # {'series1': [Episode(start=..., peak=..., end=...)]}
```

`radf()` accepts a numpy array, a 1-D sequence, or anything with
`.to_numpy()` (a pandas or polars DataFrame -- neither is a dependency;
column names become series names). Multi-column input runs the panel
version too (`bsadf_panel`, `gsadf_panel`).

## What's in the box

| | |
|---|---|
| `radf(data, minw=None, lag=0)` | recursive ADF/SADF/GSADF/BSADF statistics (C++) |
| `radf_crit(n, lag=0)` | precomputed Monte Carlo critical values from the shared store exuber's R package also reads; fetched once, cached on disk |
| `radf_mc_cv` / `radf_mc_distr` | Monte Carlo critical values / distributions, simulated locally |
| `radf_wb_cv` / `radf_wb_distr` | wild-bootstrap critical values (Harvey, Leybourne, Sollis & Taylor 2016) |
| `datestamp(result, cv, ...)` | start / peak / end / duration of each explosive episode |
| `sim_psy1`, `sim_psy2`, `sim_ps1`, `sim_ps2`, `sim_blan`, `sim_evans`, `sim_div` | bubble DGP simulators |
| `psy_minw`, `psy_ds` | the PSY default minimum window and minimum duration rules |

Not yet ported from exuber (deferred, see `exuber/__init__.py`'s
docstring for why): the Phillips & Shi wild-bootstrap variant, the sieve
bootstrap, and `summary()`/`tidy()`-style DataFrame outputs.

## Notes

- **Critical values.** `radf_crit()` covers lag 0-4 and n up to 4000; it
  returns `None` for combinations that haven't been simulated (fall back
  to `radf_mc_cv`). The store is documented at
  [exuber.kvasilopoulos.com](https://exuber.kvasilopoulos.com/).
- **Reproducibility.** The simulators and bootstraps use numpy's
  `Generator`, not R's RNG: a given `seed` gives the same draws across
  Python runs, not the same draws as the R function of the same name.
- **Numerics.** The `lag > 0` path of the statistic isn't guaranteed
  bit-identical across compilers at 1e-12 (sequential recursive updates);
  tests use 1e-9 there. `lag == 0` matches R to 1e-12.

## Building from source

Wheels are published for Linux x86_64, macOS (arm64, x86_64) and Windows
x86_64. Building from the sdist needs a C++17 compiler, CMake >= 3.16 and
a system [Armadillo](https://arma.sourceforge.net/) (with BLAS/LAPACK) --
`apt install libarmadillo-dev`, `brew install armadillo`, or
`vcpkg install armadillo` on Windows with MSVC. CMake fetches exubercore
at a pinned tag during the build.

```sh
uv sync --dev
uv run pytest
```

## License

GPL-3.0-or-later, same as exuber.
