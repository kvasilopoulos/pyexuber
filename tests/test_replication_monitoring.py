"""Thin wiring so CI's pytest run also exercises the standalone
docs/replication/monitoring/*.py scripts (see docs/replication/README.md
for the convention). Those scripts live in the umbrella exuber-project
root repo, one level above this standalone pyexuber checkout -- present
when running the full monorepo locally, absent in pyexuber's own
standalone CI clone (kvasilopoulos/pyexuber has no docs/ folder), so each
test is skipped, not failed, when the path isn't there."""

import runpy
from pathlib import Path

import pytest

REPL_DIR = Path(__file__).resolve().parents[2] / "docs" / "replication" / "monitoring"

pytestmark = pytest.mark.skipif(
    not REPL_DIR.is_dir(),
    reason="docs/replication/monitoring not present (standalone pyexuber checkout)",
)


@pytest.mark.parametrize(
    "script",
    [
        "radf_monitor_kurozumi_boundary_validation.py",
        "radf_monitor_fluc_boundary_validation.py",
        "radf_monitor_gsadf_s0_validation.py",
        "radf_cusum_validation.py",
        "radf_cusum_finite_boundary_validation.py",
        "radf_cusumv_kernel_validation.py",
    ],
)
def test_replication_script_runs_clean(script):
    runpy.run_path(str(REPL_DIR / script), run_name="__main__")
