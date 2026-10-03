# pyexuber

Python bindings for [exubercore](../exubercore), which holds the C++
`radf()`, written with pybind11. The package adds pure Python and numpy code
that mirrors the R side of `exuber`. It is distributed as `pyexuber` and
imported as `exuber`.

**The package does more than its README table suggests.** The docstring of
`src/exuber/__init__.py` is the source of truth for scope, and it is kept
current. It shows that the pure Python modules already port the Monte Carlo,
wild-bootstrap and sieve-bootstrap critical values, date-stamping, several
dating and monitoring procedures, the `diagnostics()` and `summary()` outputs
and the bubble DGP simulators. Read that docstring for the current scope. It
also lists what is deliberately deferred and explains why.

## Methodology record: `../docs/`

The record is shared with exuber and exubercore, and `../docs/README.md` is
its map. `../docs/parity.md` is the port checklist for this package. It has
one row per method, giving the R function, the Python function (or
`deferred`, or a dash) and the reason. When you port a method, update its
row. If the cross-check against R is worth re-running later, archive it as
`../docs/replication/<family>/<function>_validation.py` next to the R script,
following the convention in `../docs/replication/README.md`. The formulas,
published constants and validated numbers for each method are in
`../docs/<family>.md`. Port from there and from `exuber/R/`, and do not rely
on the paper alone.

## Build and test

```sh
uv sync --dev
uv run pytest
uv run ruff check src/ tests/
uv run ty check src/
```

CMake fetches exubercore with `FetchContent` at a pinned tag (`EXUBERCORE_TAG`
in `CMakeLists.txt`, currently `v0.3.1`). When you bump it, bump the pin in
`exuber` at the same time, as the release notes in exubercore/CLAUDE.md
describe.

**Windows with Rtools only.** A machine that has only the Rtools MinGW
toolchain and no MSVC fails at the final static link against Rtools'
`libarmadillo.a`. We did not pursue this, because it affects only local
development. Use MSVC with vcpkg instead, as the Windows job in
`.github/workflows/ci.yml` does and as exubercore's own CI does.

On Windows, DLL resolution for extension modules does not consult PATH
(bpo-36085, Python 3.8 and later). `tests/conftest.py` therefore calls
`os.add_dll_directory()` with the directory in the `EXUBER_DLL_DIR`
environment variable. Set it to vcpkg's `installed/x64-windows/bin` before
you run the tests on Windows with a fresh vcpkg install.

## Release mechanism

Releases go to PyPI through `.github/workflows/release.yml`. Pushing a
`vX.Y.Z` tag builds the sdist and the wheels (with cibuildwheel, configured
in `pyproject.toml`) and publishes them through trusted publishing. A
`workflow_dispatch` run does a build-only dry run or a TestPyPI upload.
`.claude/skills/pypi-release/checklist.md` is the per-release checklist,
which the `pypi-release` skill runs. Follow it as written. The version is
static in `pyproject.toml`, and the workflow refuses a tag that does not
match it. The package is pre-1.0, so there is no deprecation cycle:
breaking API changes land with an entry in the CHANGELOG.

## Writing style (all user-facing text)

Applies to READMEs, vignettes, the website, `docs/`, NEWS/CHANGELOG,
roxygen and docstrings, and any prose a reader sees. Code comments and
CLAUDE.md files follow it too.

**Voice.** An applied economist writing for colleagues who also want
ordinary readers to be able to run the test. Precise, sober, a little
plain-spoken. Define a term at first use (what "explosive" means, what a
critical value is for) and give the idea in words before the formula.

**Rewrite, do not substitute.** Swapping an em dash for a comma, colon or
hyphen keeps the machine-written rhythm and is not acceptable. If a
sentence needed a dash, it was carrying two thoughts: split it into two
sentences, or fold the aside into the grammar (a relative clause, a
parenthesis only for a true aside, or a separate sentence). No U+2014 and
no spaced hyphen standing in for one. En dashes stay for numeric ranges
and joint names (Phillips–Shi–Yu).

**Patterns to remove at the sentence level.**
- Fragments stacked for effect, and "X, not Y" or "not just X, but Y"
  framings. State the claim directly.
- Triplets used for rhythm, and sentences that announce what they are about
  to say ("Importantly,", "It is worth noting that", "In essence").
- Telegraphic notes (dropped articles, arrows, semicolon chains, "confirmed,
  zero new code"). Write full sentences with a subject and a verb.
- Status-report voice: "genuinely", "confirmed", "now done", "picked
  clean", "the most topical candidate". Say what is true and give the date
  if it matters.
- Marketing and filler words: seamlessly, robust (unless a statistical
  sense is stated), leverage, delve, comprehensive, powerful, crucial,
  landscape, journey, "under the hood", "a rich set of".
- Hedge stacks, and bold used as emphasis inside running prose.
- Self-reference to the writing process ("this resolves the question this
  file flagged", "an earlier pass"). Keep history in dated notes, not in
  the body of explanations.

**Do keep.** Formulas, numbers, citations, function names and every fact.
This is a change of language, not of content. Vary sentence length. Prefer
"we" or the imperative to the passive, and say what a function does and
