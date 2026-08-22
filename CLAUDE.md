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

## Build & test

```sh
uv sync --dev
uv run pytest
uv run ruff check src/ tests/
uv run ty check src/
```

CMake `FetchContent`-fetches exubercore at a pinned tag (`EXUBERCORE_TAG`
in `CMakeLists.txt`, currently `v0.1.0`) — bump it and `exuber`'s own pin
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

No PyPI publish exists yet (`pyproject.toml` version is `0.1.0`, no
`.github/workflows/` publish/release job) — CI is lint + build-and-test
only. When a release workflow gets added, mirror whatever `exuber` ends up
doing for the same methods rather than inventing a separate versioning
scheme; until then there's no deprecation cycle to maintain either (no
external consumers, pre-1.0, breaking API changes just land).
