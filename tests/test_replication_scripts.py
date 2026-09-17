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

REPLICATION_ROOT = Path(__file__).resolve().parents[2] / "docs" / "replication"


def _load(family: str, name: str):
    path = REPLICATION_ROOT / family / f"{name}.py"
    if not path.exists():
        pytest.skip(f"{path} not present (standalone pyexuber checkout, not the umbrella repo)")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_radf_wb_ps_validation_script():
    m = _load("volatility-robustness", "radf_wb_ps_validation")
    m.check_lag_select_and_adf_res()
    m.check_radf_wb_ps_cv_shapes()
    m.check_tb_mode()


def test_radf_sb_cv_aic_bic_validation_script():
    m = _load("volatility-robustness", "radf_sb_cv_aic_bic_validation")
    m.check_lag_select_bit_for_bit()
    m.check_radf_sb_cv_fixed_vs_default()
    m.check_radf_sb_cv_shapes_lag_gt_0()


def test_tidy_validation_script():
    m = _load("core-workflow", "tidy_validation")
    m.check_tidy_wide()
    m.check_tidy_long()
    m.check_tidy_panel()
    m.check_augment_wide()
    m.check_augment_long()
    m.check_augment_panel()
    m.check_augment_trunc_false()


def test_sim_vol_innovations_validation_script():
    m = _load("simulation-dgps", "sim_vol_innovations_validation")
    m.check_sim_vol_break()
    m.check_sim_vol_garch()
    m.check_sim_vol_cir()
    m.check_sim_vol_sv()
    m.check_sim_innov_normal()
    m.check_sim_innov_t_skew_t_constants()
    m.check_sim_fi_psi_and_convolution()


def test_sim_psy1_axes_and_blan_rw_validation_script():
    m = _load("simulation-dgps", "sim_psy1_axes_and_blan_rw_validation")
    m.check_sim_psy1_e()
    m.check_sim_psy1_shifts()
    m.check_sim_blan_rotermann_wilfling()
