---
name: pypi-release
description: Audit a Python project against current PyPI/PyPA publishing rules and drive a release (metadata, sdist/wheel build, checks, TestPyPI, trusted publishing). Use when asked to publish to PyPI, prepare/cut a release, check PyPI readiness, fix package metadata, or when a twine/uv publish/cibuildwheel upload fails.
---

# PyPI release

Two modes. Pick from the request; default to **audit** if unclear.

- **audit** – read the repo, compare against `reference.md`, report gaps as a
  checklist (file:line, what, fix). Don't edit unless asked.
- **release** – run the project's own checklist if one exists
  (`RELEASING.md`, `CONTRIBUTING.md`, `.github/workflows/release*.yml`) and
  fill gaps from this skill. Never invent a parallel process.

## Audit procedure

1. Read `pyproject.toml`, `README*`, `LICENSE*`, `CHANGELOG*`,
   `.gitignore`, `.github/workflows/*`, the package `__init__.py`.
2. Check each row below. `reference.md` has the why and the exact rules.

| Area | Must hold |
|---|---|
| Name | `[project] name` free on PyPI (`curl -s -o /dev/null -w '%{http_code}' https://pypi.org/pypi/NAME/json` → 404) or already owned; normalized name (PEP 503) is what PyPI stores. |
| Version | PEP 440; static in `pyproject.toml` or `dynamic = ["version"]` with a backend provider — not both. Tag == version if CI checks it. |
| Python | `requires-python` set, no upper bound. `Programming Language :: Python :: 3.X` classifiers match the versions wheels are built for. |
| License | `license = "<SPDX>"` (PEP 639) + `license-files = [...]`. No `license = {file=...}` table, no `License ::` classifier. Backend must support metadata 2.4 (setuptools ≥ 77, hatchling ≥ 1.27, scikit-build-core ≥ 0.11, flit-core ≥ 3.12). |
| README | `readme = "README.md"`; renders on PyPI (`twine check dist/*`); absolute URLs only; no relative image/links. |
| Classifiers | `Development Status`, `Programming Language :: Python :: 3` + each minor, `Operating System`, `Typing :: Typed` if `py.typed` shipped, topic. Valid names only — PyPI rejects unknown ones. |
| URLs | `[project.urls]` with well-known labels: `Homepage`, `Documentation`, `Repository`, `Issues`, `Changelog`. |
| Deps | `dependencies` lower-bounded, not pinned. Extras in `[project.optional-dependencies]`; dev tools in `[dependency-groups]` (not shipped). |
| Build | `[build-system]` present with pinned-lower-bound `requires`. Compiled ext: wheels via cibuildwheel, native libs bundled by auditwheel/delocate/delvewheel, `MACOSX_DEPLOYMENT_TARGET` as low as the deps allow. |
| sdist | Contains source + LICENSE + README + tests; excludes CI configs, `.claude/`, `CLAUDE.md`, lockfiles unless wanted, build dirs. `tar tzf dist/*.tar.gz` to verify. |
| Size | < 100 MB per file, < 10 GB per project (defaults; request more via pypi/support). |
| Publishing | GitHub Actions: build job separate from publish job; publish job has `permissions: id-token: write`, an `environment`, `pypa/gh-action-pypi-publish@release/v1`; trusted publisher registered on PyPI (and TestPyPI) for owner/repo/workflow-file/environment. No API tokens in secrets if OIDC works. |
| Immutability | Version/filename can never be re-uploaded, even after delete. Botched release → yank, bump patch, re-release. |
| Account | 2FA on. Release approval via protected environment if the repo has >1 pusher. |

3. Output: table of findings, severity (`blocks upload` / `blocks install
   for some users` / `cosmetic`), and the one-line fix each.

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

Real release: tag `vX.Y.Z` → CI builds → protected `pypi` environment
approval → `pypa/gh-action-pypi-publish` (PEP 740 attestations are
automatic ≥ 1.11). Then `pip install PKG==X.Y.Z` in a clean venv, GitHub
release from the CHANGELOG section, start the next `(unreleased)` section.

Manual fallback only: `uv publish --token pypi-...` (project-scoped token,
never account-wide, never committed).

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
