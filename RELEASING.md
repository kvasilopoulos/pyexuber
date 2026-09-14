# Releasing pyexuber to PyPI

Copy this list into the release PR/issue and tick it. Every item is
either a command or a yes/no check; nothing here is optional unless
marked.

## 1. One-time setup (skip once done)

- [ ] PyPI project `pyexuber` exists, with a **trusted publisher** for
      `kvasilopoulos/pyexuber`, workflow `release.yml`, environment `pypi`.
      (PyPI -> project -> Publishing -> add GitHub publisher. For the very
      first release use "pending publisher" on pypi.org/manage/account/publishing/.)
- [ ] Same on test.pypi.org with environment `testpypi`.
- [ ] GitHub repo -> Settings -> Environments: `pypi` and `testpypi` exist
      (`pypi` with required reviewer = you, so a stray tag can't publish).

## 2. Before tagging

- [ ] `main` is green: <https://github.com/kvasilopoulos/pyexuber/actions>
- [ ] `EXUBERCORE_TAG` in `CMakeLists.txt` matches the exubercore tag
      `exuber` (R) vendors -- the two must not drift.
- [ ] `version` in `pyproject.toml` bumped (semver; pre-1.0 breaking
      changes bump minor).
- [ ] `CHANGELOG.md`: rename "(unreleased)" to today's date, entries
      cover every user-visible change since the last tag
      (`git log --oneline vLAST..`).
- [ ] `README.md` still describes what the package does (scope table,
      supported platforms, Python floor) -- it is the PyPI landing page.
- [ ] `classifiers` in `pyproject.toml`: Python versions match
      `[tool.cibuildwheel] build`; `Development Status` still accurate.
- [ ] `src/exuber/__init__.py` docstring's "not yet ported" list is current.
- [ ] Local: `uv run --no-sync ruff check src/ tests/ && uv run --no-sync ty check src/`
- [ ] Local: `uv build --sdist && uvx twine check --strict dist/* && tar tzf dist/*.tar.gz`
      -- metadata/README render OK, no stray files (build/, .venv,
      caches, CLAUDE.md, .github).
- [ ] Dry run: Actions -> Release -> Run workflow (target = `none`) on
      `main`; all four wheel jobs + sdist green. Download a wheel artifact
      and `uvx check-wheel-contents <wheel>`, then `pip install` it into
      a clean venv and run:
      `python -c "import exuber; print(exuber.__version__, exuber.radf(exuber.sim_psy1(100, seed=1)).gsadf)"`
- [ ] Optional: same with target = `testpypi`, then
      `pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple pyexuber==X.Y.Z`.

## 3. Tag and publish

```sh
git tag -a vX.Y.Z -m "pyexuber X.Y.Z"
git push origin vX.Y.Z
```

- [ ] Release workflow ran on the tag; approve the `pypi` environment
      when prompted.
- [ ] <https://pypi.org/project/pyexuber/> shows X.Y.Z with sdist + all
      wheels.
- [ ] `pip install pyexuber==X.Y.Z` in a clean venv on at least one
      platform, run the one-liner above.

## 4. After

- [ ] GitHub release created from the tag, body = the CHANGELOG section.
- [ ] `CHANGELOG.md`: add a new "(unreleased)" section at the top.
- [ ] Website (`website/`, the suite page) mentions the PyPI package and
      the version if it names one.
- [ ] If the release also bumped `EXUBERCORE_TAG`, `exuber` (R) got the
      same bump (or an issue exists for it).

## If a wheel job fails

- Linux: Armadillo comes from EPEL 8 (`armadillo-devel`) inside the
  manylinux_2_28 container; if EPEL drops/renames it, build Armadillo from
  source in `before-all` instead (it's a small CMake project needing only
  `openblas-devel`).
- macOS: `MACOSX_DEPLOYMENT_TARGET` in `[tool.cibuildwheel.macos]` must
  equal the runner's macOS major version (Homebrew bottles target the host
  OS; delocate rejects newer libraries than the wheel's tag). Bumping the
  runner label means bumping that value.
- Windows: `C:/vcpkg/...` paths in `[tool.cibuildwheel.windows]` are
  GitHub's hosted-runner layout; delvewheel bundles openblas/lapack/
  armadillo DLLs from `installed/x64-windows/bin`.
- Never re-upload a version to PyPI: fix, bump the patch version, re-tag.
