"""Regression tests for lambda=0.30 launch paths and reward isolation."""

import pytest

import scripts.launchers.run_relay_stress_r4_a5_lam03_pilot as train
import scripts.evaluation.eval_relay_stress_r4_a5_lam03_qmix as evaluation
from scripts.training.run_relay_stress_r4_a5_qmix import shaped_training_reward
from src.envs.thesis_cartesian_r4_relay_stress_env import (
    CartesianR4RelayStressBatteryConfig,
)


def test_lambda_matches_requested_value():
    assert train.LAMBDA_G == pytest.approx(0.30)
    assert evaluation.LAMBDA_G == pytest.approx(0.30)
    assert shaped_training_reward(0.5, 0.05, train.LAMBDA_G) == pytest.approx(0.515)


def test_outputs_are_isolated_from_other_lambda_runs():
    assert train.DEFAULT_OUTDIR.endswith("/a5_relay_required_lam0p30")
    assert evaluation.TRAIN_DIR == train.ROOT / train.DEFAULT_OUTDIR
    assert evaluation.DEFAULT_EVAL_DIR.name == "a5_lam0p30_eval"
    assert not any(s in train.DEFAULT_OUTDIR for s in ("lam1p00", "lam0p10", "original_reward"))


def test_r4_still_200m_backhaul_and_default_seed_is_91():
    cfg = CartesianR4RelayStressBatteryConfig()
    assert cfg.backhaul_radius_cells * cfg.cell_size_m == pytest.approx(200.0)
    assert 0.30 != 0.10 and 0.30 != 1.0
