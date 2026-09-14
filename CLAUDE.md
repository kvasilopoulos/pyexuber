# pyexuber

Python bindings for [exubercore](../exubercore) (C++ `radf()`, via
pybind11) plus pure-Python/numpy orchestration mirroring `exuber`'s R side.
Distributed as `pyexuber`, imported as `exuber`.

**Scope is ahead of README.md** — the README still says "currently binds
only `radf()`", but `src/exuber/__init__.py`'s docstring (the actual source
of truth, kept current) shows `cv.py`/`datestamp.py`/`sim.py` already port
Monte Carlo + wild-bootstrap critical values, date-stamping, and the bubble
DGP simulators as pure Python. Read that docstring, not the README, for
current scope — including what's deliberately deferred (PS wild-bootstrap
variant, sieve bootstrap, `.summary()`/`.tidy()` methods) and why.

## Methodology record: `../docs/`

Shared with exuber and exubercore; `../docs/README.md` is the map.
`../docs/parity.md` is this package's port checklist — one row per
method with the R function, the Python function (or `deferred`/`—`) and
why. When you port something: update its row, and if the cross-check
against R is worth keeping re-runnable, archive it as
`../docs/replication/<family>/<function>_validation.py` next to the R
script (convention in `../docs/replication/README.md`). The method's
formulas, published constants and validated numbers are in
`../docs/<family>.md` — port from there and from `exuber/R/`, not from
the paper alone.

## Build & test

```sh
uv sync --dev
uv run pytest
uv run ruff check src/ tests/
uv run ty check src/
```

CMake `FetchContent`-fetches exubercore at a pinned tag (`EXUBERCORE_TAG`
in `CMakeLists.txt`, currently `v0.3.1`) — bump it and `exuber`'s own pin
together, per exubercore/CLAUDE.md's release-mechanism note.

**Windows/Rtools-only dev note:** a box with only Rtools' MinGW toolchain
(no MSVC) fails at the final static link against Rtools' `libarmadillo.a`
— unresolved, not chased further, local-only limitation. Use MSVC + vcpkg
(see `.github/workflows/ci.yml`'s Windows job) instead, same as
exubercore's own CI.

Windows extension-module DLL resolution doesn't consult PATH (bpo-36085,
Python 3.8+) — `tests/conftest.py` calls `os.add_dll_directory()` via the
`EXUBER_DLL_DIR` env var; set it to vcpkg's `installed/x64-windows/bin`
before running tests on Windows with a fresh vcpkg install.

## Release mechanism

PyPI, via `.github/workflows/release.yml`: pushing a `vX.Y.Z` tag builds
sdist + wheels (cibuildwheel, config in `pyproject.toml`) and publishes
through trusted publishing; `workflow_dispatch` does a build-only dry run
or a TestPyPI push. `.claude/skills/pypi-release/checklist.md` is the
per-release checklist (run by the `pypi-release` skill) -- follow it,
don't improvise. Version is static in `pyproject.toml` and the
workflow refuses a tag that doesn't match it. Pre-1.0: no deprecation
cycle, breaking API changes just land with a CHANGELOG entry.
