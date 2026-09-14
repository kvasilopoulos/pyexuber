# PyPI / PyPA publishing rules — reference (checked 2026-09-14)

Condensed from packaging.python.org (tutorial, "Writing your pyproject.toml",
"Publishing with GitHub Actions"), docs.pypi.org (storage limits, yanking,
2FA), cibuildwheel 4.2 docs, scikit-build-core 1.0 docs, uv "Building and
publishing" guide. Tool versions current at check time: scikit-build-core
1.0.3, cibuildwheel 4.2.1, pybind11 3.1.0, `pypa/gh-action-pypi-publish`
release/v1 (attestations since 1.11.0).

## 1. `[project]` metadata (PEP 621 / core metadata 2.4)

Required: `name`, `version` (or `dynamic = ["version"]`).

Strongly recommended, all read by PyPI's project page:

| Key | Rule |
|---|---|
| `name` | letters, digits, `.`, `_`, `-`; PyPI normalizes (PEP 503: lowercase, runs of `.-_` → `-`). Distribution name may differ from import name (e.g. `pyexuber` / `exuber`) — say so in the README's first lines. Abandoned names go through PEP 541. |
| `version` | PEP 440 (`1.2.0`, `1.2.0rc1`, `1.2.0.post1`, `1.2.0.dev3`). Local versions (`+abc`) are rejected by PyPI. |
| `description` | one sentence; shown in search results. |
| `readme` | file path; content type inferred from `.md`/`.rst`, or `{file=, content-type=}`. Rendered with GitHub-flavoured Markdown; unrenderable → PyPI shows raw text (not a rejection; `twine check` catches it). |
| `requires-python` | e.g. `">=3.10"`. Installer-enforced. Don't add an upper bound: it makes every future Python resolve to old releases. |
| `license` | SPDX expression string: `"MIT"`, `"GPL-3.0-or-later"`, `"MIT AND (Apache-2.0 OR BSD-2-Clause)"`, `"LicenseRef-Proprietary"`. Deprecated: `license = {file=...}` / `{text=...}` table and `License :: OSI Approved :: ...` classifiers — PyPI rejects an upload that has both a license expression and a License classifier. |
| `license-files` | globs, forward slashes, no `..`: `["LICENSE*", "AUTHORS*"]`. Default in most backends is `LICEN[CS]E*`, `COPYING*`, `NOTICE*`, `AUTHORS*`. |
| `authors` / `maintainers` | `[{name=, email=}]`. |
| `keywords` | list; only affects search. |
| `classifiers` | exact strings from https://pypi.org/classifiers/. Unknown ones are rejected at upload. Useful: `Development Status :: 3 - Alpha` … `5 - Production/Stable`; `Programming Language :: Python :: 3` + one per supported minor (PyPI sidebar and badge tooling read these; `requires-python` is what actually restricts installs); `Operating System :: ...`; `Intended Audience :: Science/Research`; `Topic :: Scientific/Engineering`; `Typing :: Typed` (only if `py.typed` ships in the wheel). `Private :: Do Not Upload` blocks accidental upload of internal packages. |
| `dependencies` | PEP 508 strings with markers. Lower bounds, not `==` pins (pins belong in lockfiles). |
| `[project.optional-dependencies]` | extras (`pip install pkg[pandas]`); an extra may reference others: `all = ["pkg[pandas,polars]"]`. |
| `[dependency-groups]` (PEP 735) | dev/test tooling; not published, not installable from PyPI — the right place for pytest/ruff. |
| `[project.urls]` | PyPI recognises (case-insensitive) `Homepage`, `Documentation`, `Repository`/`Source`, `Issues`/`Bug Tracker`, `Changelog`/`Release Notes`, `Funding`/`Sponsor` and gives them icons. Anything else is listed as a plain link. |
| `[project.scripts]` / `[project.entry-points."group"]` | console scripts / plugins. |
| `dynamic` | list any field the backend fills in (typically `version`); a field may not be both static and dynamic. |

## 2. `[build-system]`

Always present, even for pure-Python. `requires` lists the backend with a
lower bound that has the features used (PEP 639 needs setuptools >= 77,
hatchling >= 1.27, flit-core >= 3.12, scikit-build-core >= 0.11, pdm-backend
>= 2.4, uv-build >= 0.6). Backends: hatchling (pure-Python default),
setuptools, flit-core, pdm-backend, uv-build; compiled: scikit-build-core
(CMake), meson-python, maturin (Rust), setuptools + extension modules.

Build in isolation exactly as pip will: `python -m build` / `uv build
--no-sources` (uv: verifies the package builds without `tool.uv.sources`
overrides). Output `dist/*.tar.gz` (sdist) + `dist/*.whl`.

## 3. sdist and wheel contents

- Always upload **both** an sdist and wheels. sdist = provenance and the
  fallback for platforms without a wheel; wheel = what users actually get.
- sdist should hold: sources, tests, `LICENSE*`, `README`, `pyproject.toml`,
  anything the build needs (CMakeLists, headers). Should not hold: CI
  config, `.claude/`, `CLAUDE.md`, editor config, `build/`, `dist/`,
  caches, large fixtures. Backends default to "all git-tracked / not
  gitignored files", so files like `RELEASING.md` and `.github/` end up in
  the sdist unless excluded (`[tool.scikit-build] sdist.exclude`,
  `[tool.hatch.build.targets.sdist] exclude`, `MANIFEST.in` for setuptools).
- Verify: `tar tzf dist/*.tar.gz`, `unzip -l dist/*.whl` or
  `check-wheel-contents dist/*.whl` (flags stray files, missing
  `py.typed`, top-level junk, duplicate modules).
- `twine check dist/*`: metadata validity + README render check. Run it
  in CI before upload; `pypa/gh-action-pypi-publish` also runs it.
- Reproducible builds: backends honour `SOURCE_DATE_EPOCH`.

## 4. Compiled extensions (pybind11 / Cython / Rust)

- Build wheels with **cibuildwheel** (`pypa/cibuildwheel@v4`, config in
  `[tool.cibuildwheel]`). Defaults (4.x): builds CPython 3.9-3.15 + PyPy +
  GraalPy for every arch the runner has; Linux image `manylinux_2_28`
  (AlmaLinux 8, glibc 2.28) and `musllinux_1_2`; repair step on by default:
  auditwheel (Linux), delocate (macOS), delvewheel (Windows) copy the
  extension's shared-library dependencies into the wheel and rewrite
  rpaths so the user needs nothing but `pip install`.
- Narrow with `build = "cp310-* cp311-* ..."`, `skip`, `archs`; enable
  free-threaded/prerelease via `enable = ["cpython-prerelease", "cpython-freethreading"]`.
- Native deps (BLAS, Armadillo, ...): install in `before-all` per platform
  (`[tool.cibuildwheel.linux] before-all = "dnf install -y ..."` runs inside
  the manylinux container; `brew install` on macOS; `vcpkg` on Windows).
  Windows needs `before-build = "pip install delvewheel"` and
  `repair-wheel-command = "delvewheel repair -w {dest_dir} {wheel} --add-path <dll dir>"`.
- macOS: `MACOSX_DEPLOYMENT_TARGET` sets the wheel's minimum OS
  (`macosx_11_0_arm64` etc.). Homebrew bottles are built for the runner's
  OS, so delocate refuses them for a lower target: either raise the
  target (excludes older macOS users) or build the dep from source with
  the low target. `macos-15` runners are arm64, `macos-15-intel` x86_64;
  `macos-13` is the last Intel image with older-OS bottles.
- Test each wheel: `test-command = "pytest {project}/tests"`,
  `test-requires` / `test-groups`. Use `{project}` (or `test-sources`): the
  test runs from a temp dir so the wheel, not the source tree, is imported.
- Stable ABI (one wheel per platform for all Pythons >= X): scikit-build-core
  `wheel.py-api = "cp312"` + pybind11 `Py_LIMITED_API`; only worth it if
  the binding code supports it. PEP 803 free-threaded stable ABI is
  supported in scikit-build-core 1.0.
- Config-settings from cibuildwheel to the backend:
  `[tool.cibuildwheel.config-settings] "cmake.define.X" = "Y"` or env
  `SKBUILD_*`.
- Keep the sdist buildable from source: document system deps in README;
  users on unsupported platforms/Pythons fall back to it silently.

## 5. Accounts and access

- 2FA is mandatory for every PyPI account (TOTP or WebAuthn). Keep
  recovery codes.
- Username/password upload is gone. Options: **trusted publishing**
  (OIDC, preferred) or **API tokens** (`pypi-...`, project-scoped after the
  first upload; account-scoped only for the very first release if not
  using a pending publisher). Tokens go in CI secrets, never in the repo.
- TestPyPI (https://test.pypi.org) is a separate account/token/publisher.
  Not permanent: packages and accounts get purged. Install from it with
  `--index-url https://test.pypi.org/simple/ --no-deps` (its dependency
  index is incomplete; add `--extra-index-url https://pypi.org/simple`
  if deps are needed).
- Collaborators: add a second owner for bus-factor; the "maintainer" role
  can upload but not manage.

## 6. Trusted publishing (GitHub Actions, OIDC)

Register on PyPI: project -> Publishing (or, before the first upload,
Account -> Publishing -> "pending publisher", which also reserves the
name): owner, repository, workflow filename (`release.yml`, not the path),
environment name. All four must match the running job exactly.

Workflow rules from the PyPA guide:

- Build job and publish job **separate**; the publishing job downloads the
  artifacts (`actions/download-artifact` with `merge-multiple: true`).
  "Building distributions in a publishing job is unsupported."
- Publish job: `permissions: id-token: write` (mandatory), `environment:
  pypi` (make it a protected environment with required reviewers so a
  stray tag can't publish), step `uses: pypa/gh-action-pypi-publish@release/v1`.
  TestPyPI: same with `repository-url: https://test.pypi.org/legacy/`
  and environment `testpypi`.
- Trigger PyPI on tags only (`on: push: tags: ["v*"]` or
  `if: startsWith(github.ref, 'refs/tags/')`).
- Attestations (PEP 740, Sigstore) are generated automatically by
  gh-action-pypi-publish >= 1.11 under trusted publishing; `uv publish`
  uploads `*.publish.attestation` files it finds. No manual signing.
- `uv publish` also does trusted publishing automatically inside GitHub
  Actions (no token needed). `--check-url <index>` makes retries
  idempotent by skipping files already uploaded.

## 7. Immutable releases, yanking, deletion

- A filename (`pkg-1.2.0-cp312-...whl`, `pkg-1.2.0.tar.gz`) can be
  uploaded **once, ever**: not after deleting the file, not after
  deleting and re-creating the project. Fix -> bump version -> re-tag.
- **Yank** (release page -> Options -> Yank) for broken releases: existing
  pins `==X.Y.Z` still resolve (with a warning), unpinned resolution skips
  it. Give a reason. Prefer yank over delete; delete only for leaked
  secrets or legal problems, and it still doesn't free the filename.
- Post-releases (`1.2.0.post1`) are for packaging-only fixes; PyPI orders
  them after `1.2.0`.

## 8. Limits and policies

- 100 MB per file, 10 GB per project by default. Larger: open a "Limit
  Request" issue at https://github.com/pypi/support (needs justification;
  typical for GPU/ML wheels). Nightly-style version spam counts against
  the project total.
- Keep large images out of the README; it's part of the upload payload.
- Organisation accounts exist for shared ownership; free for community
  projects.
- Don't publish placeholder/name-squat releases (acceptable-use policy);
  a pending trusted publisher reserves the name without an upload.

## 9. Release checklist (generic)

1. Version bumped (`pyproject.toml` or tag-derived), CHANGELOG has the
   section, README accurate (it *is* the PyPI page).
2. `uv build --no-sources && twine check dist/* && tar tzf dist/*.tar.gz`.
3. Fresh-venv install of the wheel + one smoke command.
4. First release: TestPyPI end-to-end, then
   `pip install --index-url https://test.pypi.org/simple/ --no-deps pkg==X.Y.Z`.
5. Tag `vX.Y.Z` (annotated), push the tag, approve the protected environment.
6. Verify https://pypi.org/project/NAME/X.Y.Z/ lists the sdist + every
   expected wheel; `pip install NAME==X.Y.Z` on one clean machine.
7. GitHub release from the tag; open the next `(unreleased)` section.

## Sources

- https://packaging.python.org/en/latest/tutorials/packaging-projects/
- https://packaging.python.org/en/latest/guides/writing-pyproject-toml/
- https://packaging.python.org/en/latest/specifications/license-expression/
- https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/
- https://docs.pypi.org/project-management/storage-limits/
- https://docs.pypi.org/project-management/yanking/
- https://docs.pypi.org/trusted-publishers/
- https://docs.pypi.org/attestations/
- https://cibuildwheel.pypa.io/en/stable/options/
- https://scikit-build-core.readthedocs.io/en/latest/configuration/index.html
- https://docs.astral.sh/uv/guides/package/
- https://pypi.org/classifiers/
