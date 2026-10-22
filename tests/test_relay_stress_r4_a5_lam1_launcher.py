import pytest

import scripts.launchers.run_relay_stress_r4_a5_lam1_pilot as launcher
import scripts.training.run_relay_stress_r4_a5_qmix as a5


def test_lambda1_launcher_is_exactly_one():
    assert launcher.LAMBDA_G == pytest.approx(1.0)


def test_lambda1_has_separate_output_folder():
    assert "lam1p00" in launcher.DEFAULT_OUTDIR
    assert "lam0p10" not in launcher.DEFAULT_OUTDIR


def test_reward_scale_lambda1():
    # A score of 0.01 contributes 0.01 reward when lambda=1.0.
    assert a5.shaped_training_reward(0.5, 0.01, launcher.LAMBDA_G) == pytest.approx(0.51)
