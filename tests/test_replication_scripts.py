"""Wires docs/replication/volatility-robustness/*.py (root exuber-project
repo's replication scripts, cross-checked against the matching R script)
into pytest, per docs/replication/README.md's convention.

Those scripts live in a *different* git repo (the umbrella exuber-project,
not pyexuber) at ``../../docs/replication`` relative to this file, so
they're only present when pyexuber is checked out as part of a full
umbrella clone -- pyexuber's own standalone CI checkout (github.com/
kvasilopoulos/pyexuber) won't have them. Skip rather than fail when
that's the case; the scripts are also runnable standalone (see their own
``if __name__ == "__main__"`` blocks) for whenever they *are* present.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

REPLICATION_DIR = (
    Path(__file__).resolve().parents[2] / "docs" / "replication" / "volatility-robustness"
)


def _load(name: str):
    path = REPLICATION_DIR / f"{name}.py"
    if not path.exists():
        pytest.skip(f"{path} not present (standalone pyexuber checkout, not the umbrella repo)")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_radf_wb_ps_validation_script():
    m = _load("radf_wb_ps_validation")
    m.check_lag_select_and_adf_res()
    m.check_radf_wb_ps_cv_shapes()
    m.check_tb_mode()
