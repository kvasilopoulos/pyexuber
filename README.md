# pyexuber

Python bindings for [exubercore](https://github.com/kvasilopoulos/exubercore),
the C++ core behind the R package
[exuber](https://github.com/kvasilopoulos/exuber) (recursive right-tailed
unit root tests -- ADF/SADF/GSADF/BSADF -- for detecting explosive dynamics
in time series). Distributed as `pyexuber`, imported as `exuber`.

## Scope

This currently binds only `exubercore::radf()`, the recursive least-squares
ADF/SADF/GSADF/BSADF statistic -- the numerically expensive routine.
Everything RNG-driven (Monte Carlo/wild/sieve bootstrap critical values,
date-stamping, the bubble DGP simulators) is exuber R-side orchestration
around repeated calls to that routine, not yet ported here; that's future
work, mirroring whatever exuber itself ends up doing for those pieces.

```python
import numpy as np
import exuber

data = np.cumsum(np.random.randn(200))
result = exuber.radf(data)
result.adf, result.sadf, result.gsadf
```

`exuber.radf()` accepts a numpy array, a 1-D sequence, or anything exposing
`.to_numpy()` (pandas/polars DataFrames) -- neither pandas nor polars is a
hard dependency.

## Build

CMake fetches [exubercore](https://github.com/kvasilopoulos/exubercore) at
a pinned tag and [CARMA](https://github.com/RUrlus/carma) for numpy<->
Armadillo conversion; requires a system Armadillo (pulls BLAS/LAPACK).

```sh
uv sync --dev
uv run pytest
```

**Windows/Rtools-only dev note:** on a box with only Rtools' MinGW
toolchain and no MSVC (this project's original dev machine), the build
needs an explicit generator/compiler override and currently fails at the
final static link against Rtools' `libarmadillo.a` (missing LAPACK symbols
from a static-archive linking quirk specific to that toolchain layout) --
unresolved, not chased further since it's a local-only limitation: real
Windows builds should use MSVC + vcpkg (see CI), the same combination
already proven working for `exubercore` itself.

## Numerics

Same tolerance caveat as exubercore: the `lag > 0` path isn't guaranteed
bit-identical across toolchains at 1e-12 (cross-compiler floating-point
drift over its O(n^2) sequential updates); tests use 1e-9 there.
