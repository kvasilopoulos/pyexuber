"""Thin pytest wiring for the standalone replication scripts in the
umbrella repo's docs/replication/dating-and-root-inference/, per
docs/replication/README.md's convention (each .py script cross-checks the
R script of the same name and is runnable standalone). Skipped when run
from a standalone pyexuber checkout where that umbrella docs/ tree isn't
present (e.g. this repo's own GitHub Actions CI, which checks out only
kvasilopoulos/pyexuber) -- exercised for real from within the umbrella
checkout, locally or by anyone building the train.
"""

import runpy
from pathlib import Path

import pytest

_REPLICATION_DIR = (
    Path(__file__).resolve().parents[2] / "docs" / "replication" / "dating-and-root-inference"
)


def _run_script(name: str) -> None:
    path = _REPLICATION_DIR / name
    if not path.exists():
        pytest.skip(f"{path} not present (standalone pyexuber checkout, not the umbrella repo)")
    runpy.run_path(str(path), run_name="__main__")


def test_rootstamp_validation_script():
    _run_script("rootstamp_validation.py")


def test_radf_pdc_validation_script():
    _run_script("radf_pdc_validation.py")


def test_radf_recovery_validation_script():
    _run_script("radf_recovery_validation.py")


def test_radf_hls_validation_script():
    _run_script("radf_hls_validation.py")


def test_radf_hlw_validation_script():
    _run_script("radf_hlw_validation.py")
