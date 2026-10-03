---
name: pypi-release
description: Audit a Python project against current PyPI/PyPA publishing rules and drive a release (metadata, sdist/wheel build, checks, TestPyPI, trusted publishing). Use when asked to publish to PyPI, prepare/cut a release, check PyPI readiness, fix package metadata, or when a twine/uv publish/cibuildwheel upload fails.
---

# PyPI release

This skill has two modes. Choose one from the request, and use audit mode when the request is unclear.

- **audit.** Read the repo, compare it against `reference.md`, and report the
  gaps as a checklist that gives the file and line, the problem and the fix.
  Do not edit anything unless asked.
- **release.** Run `checklist.md`, which sits next to this file and is the
  per-release list, and tick items off as you go. Use it together with
  `.github/workflows/release.yml`. Fill any gaps from this skill, and do not
  invent a parallel process.

## Who does what

Claude runs the release from start to finish except for three human gates.
Do everything else without asking. At a gate, stop, hand over the exact
artifact or command, and wait. `checklist.md` marks which items are gates.

| Human gate | What Claude hands over |
|---|---|
| **1. Description check.** The text that users see: `[project] description`, the README (the PyPI landing page), and the classifiers including `Development Status` | a diff of proposed wording changes; commit only after approval |
| **2. Release notes.** The `CHANGELOG.md` section for the version, which becomes the GitHub release body | a drafted section from `git log vLAST..`; the human edits/approves |
| **3. Final submission.** `git push origin vX.Y.Z` and approval of the protected `pypi` environment | the ready tag command and the Actions URL; never push a `v*` tag or approve a deployment yourself |

Claude owns: version bump, metadata fixes, `sdist.exclude`/cibuildwheel
config, all local checks, the build-only dry run and the TestPyPI push
(`gh workflow run release.yml -f target=testpypi`), watching CI
(`gh run watch`, `gh run view --log-failed`), TestPyPI install
verification, post-publish verification, the GitHub release
(`gh release create vX.Y.Z --notes-from-tag` or from the CHANGELOG
section), and opening the next `(unreleased)` CHANGELOG section.

**Monitoring by email.** Use mail only for what does not reach `gh`, such as
PyPI notices and requests to approve an environment. The release mailbox is
the `authors` email in `pyproject.toml`, and the claude.ai Gmail connector is
on that account. Search with `mcp__claude_ai_Gmail__search_threads` (`query`,
Gmail syntax), then read a thread with `mcp__claude_ai_Gmail__get_thread`
and `messageFormat: PLAIN_TEXT`. Before you act on a hit, check that its
`to_recipients` is that address. If it is not, the connector is on another
account, so fall back to `gh` and say so.

Queries (Gmail syntax):

```
# GitHub Actions failure / environment approval request
from:notifications@github.com pyexuber (subject:"Run failed" OR subject:"review required" OR "requires approval") newer_than:1d
# PyPI: trusted publisher added, new release published, security notices
from:pypi.org pyexuber newer_than:7d
```

While a workflow runs, poll about every 10 minutes (or use `gh run watch`).
Check for PyPI notices daily.

## Audit procedure

1. Read `pyproject.toml`, `README*`, `LICENSE*`, `CHANGELOG*`,
   `.gitignore`, `.github/workflows/*`, the package `__init__.py`.
2. Check each row below. `reference.md` gives the reasons and the exact rules.

| Area | Must hold |
|---|---|
| Name | `[project] name` free on PyPI (`curl -s -o /dev/null -w '%{http_code}' https://pypi.org/pypi/NAME/json` → 404) or already owned; normalized name (PEP 503) is what PyPI stores. |
| Version | PEP 440; static in `pyproject.toml`, or `dynamic = ["version"]` with a backend provider, but not both. The tag must equal the version if CI checks it. |
| Python | `requires-python` set, no upper bound. `Programming Language :: Python :: 3.X` classifiers match the versions wheels are built for. |
| License | `license = "<SPDX>"` (PEP 639) + `license-files = [...]`. No `license = {file=...}` table, no `License ::` classifier. Backend must support metadata 2.4 (setuptools ≥ 77, hatchling ≥ 1.27, scikit-build-core ≥ 0.11, flit-core ≥ 3.12). |
| README | `readme = "README.md"`; renders on PyPI (`twine check dist/*`); absolute URLs only; no relative image/links. |
| Classifiers | `Development Status`, `Programming Language :: Python :: 3` + each minor, `Operating System`, `Typing :: Typed` if `py.typed` shipped, topic. Use valid names only, because PyPI rejects unknown ones. |
| URLs | `[project.urls]` with well-known labels: `Homepage`, `Documentation`, `Repository`, `Issues`, `Changelog`. |
| Deps | `dependencies` lower-bounded, not pinned. Extras in `[project.optional-dependencies]`; dev tools in `[dependency-groups]` (not shipped). |
| Build | `[build-system]` present with pinned-lower-bound `requires`. Compiled ext: wheels via cibuildwheel, native libs bundled by auditwheel/delocate/delvewheel, `MACOSX_DEPLOYMENT_TARGET` as low as the deps allow. |
| sdist | Contains source + LICENSE + README + tests; excludes CI configs, `.claude/`, `CLAUDE.md`, lockfiles unless wanted, build dirs. `tar tzf dist/*.tar.gz` to verify. |
| Size | < 100 MB per file, < 10 GB per project (defaults; request more via pypi/support). |
| Publishing | GitHub Actions: build job separate from publish job; publish job has `permissions: id-token: write`, an `environment`, `pypa/gh-action-pypi-publish@release/v1`; trusted publisher registered on PyPI (and TestPyPI) for owner/repo/workflow-file/environment. No API tokens in secrets if OIDC works. |
| Immutability | Version/filename can never be re-uploaded, even after delete. Botched release → yank, bump patch, re-release. |
| Account | 2FA on. Release approval via protected environment if the repo has >1 pusher. |

3. Output a table of findings. Give each a severity (`blocks upload`,
   `blocks install for some users` or `cosmetic`) and a one-line fix.

## Release procedure (when no project checklist exists)

```sh
uv build --no-sources                 # or: python -m build
uvx twine check dist/*                # README + metadata validity
tar tzf dist/*.tar.gz                 # sdist hygiene
uvx check-wheel-contents dist/*.whl   # stray files / missing py.typed
# fresh-venv smoke test of the wheel:
uv run --no-project --with dist/*.whl -- python -c "import PKG; print(PKG.__version__)"
```

TestPyPI first for a first release: `uv publish --index testpypi` (needs
`[[tool.uv.index]] name="testpypi" url="https://test.pypi.org/simple/"
publish-url="https://test.pypi.org/legacy/" explicit=true`) or push
through the workflow's TestPyPI path. Verify with
`pip install --index-url https://test.pypi.org/simple/ --no-deps PKG==X.Y.Z`
(`--no-deps`: TestPyPI's dependency set is incomplete).

Real release (gate 3): the human tags and pushes `vX.Y.Z`, and CI builds the
distributions. The human then approves the protected `pypi` environment, and
`pypa/gh-action-pypi-publish` uploads them (PEP 740 attestations are created
automatically from version 1.11). After that, Claude installs
`PKG==X.Y.Z` with pip in a clean venv, creates the GitHub release from the
approved CHANGELOG section, and opens the next `(unreleased)` section.

Use a manual upload only as a fallback: `uv publish --token pypi-...`, with a
project-scoped token that is never account-wide and never committed.

## Failure playbook

| Symptom | Cause / fix |
|---|---|
| `400 File already exists` | filename immutable; bump version |
| `400 ... license classifier ... deprecated` | remove `License ::` classifier, keep SPDX `license` |
| `Invalid value for classifiers` | typo/unknown classifier; check https://pypi.org/classifiers/ |
| README not rendering | `twine check`; fix Markdown, set `readme = {file=..., content-type="text/markdown"}` |
| `invalid-publisher` (OIDC) | owner/repo/workflow filename/environment on PyPI don't match the running workflow exactly |
| auditwheel `not manylinux_2_28 compatible` | system lib linked from outside the container's glibc; build the dep in `before-all` |
| delocate `is newer than the wheel target` | `MACOSX_DEPLOYMENT_TARGET` lower than the Homebrew bottle's; raise it or build the dep from source |
| Windows `ImportError: DLL load failed` | dependency DLL not bundled; `delvewheel repair --add-path <bin dir>` |
| Install falls back to sdist | no wheel for that Python/platform tag; add it to `[tool.cibuildwheel] build` or accept + document the system deps |
