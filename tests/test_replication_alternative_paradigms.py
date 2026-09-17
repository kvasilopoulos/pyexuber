"""Thin wiring so CI's pytest run also exercises the standalone
docs/replication/alternative-paradigms/*.py scripts -- see
tests/test_replication_monitoring.py's module docstring for the
skip-when-absent rationale (this checkout doesn't include the umbrella
repo's docs/ folder in pyexuber's own standalone CI)."""

import runpy
from pathlib import Path

import pytest

REPL_DIR = Path(__file__).resolve().parents[2] / "docs" / "replication" / "alternative-paradigms"

pytestmark = pytest.mark.skipif(
    not REPL_DIR.is_dir(),
    reason="docs/replication/alternative-paradigms not present (standalone pyexuber checkout)",
)


@pytest.mark.parametrize(
    "script",
    [
        "radf_quantile_validation.py",
        "radf_qpwy_validation.py",
    ],
)
def test_replication_script_runs_clean(script):
    runpy.run_path(str(REPL_DIR / script), run_name="__main__")
