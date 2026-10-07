import numpy as np
import pytest

import scripts.run_relay_stress_r4_qmix as r4run
import scripts.run_thesis_no_curriculum_fulltask as r6run
from scripts.eval_relay_stress_r4_qmix import direct_access_cells, relay_required_ue_mask


def test_r4_runner_uses_separate_200m_environment():
    env = r4run.make_env(3101, 5)
    assert env.cfg.backhaul_radius_cells == pytest.approx(4.0)
    assert env.cfg.backhaul_radius_cells * env.cfg.cell_size_m == pytest.approx(200.0)
    assert env.cfg.access_radius_cells == pytest.approx(4.0)
    assert env.n_actions == 10


def test_r4_runner_keeps_locked_training_protocol():
    assert r4run.TRAIN_SEEDS == r6run.TRAIN_SEEDS
    assert r4run.VALIDATION_SEEDS == r6run.VALIDATION_SEEDS
    assert r4run.TEST_SEEDS == r6run.TEST_SEEDS
    assert r4run.BLOCK_SPECS == r6run.BLOCK_SPECS


def test_r4_seed3101_contains_relay_required_ues():
    env = r4run.make_env(3101, 5)
    env.reset(seed=3101)
    mask = relay_required_ue_mask(env, env.ue_positions, direct_access_cells(env))
    assert mask.shape == (env.cfg.num_ues,)
    assert int(mask.sum()) > 0


def test_r4_original_reward_is_environment_utility():
    env = r4run.make_env(3101, 5)
    out = env.reset(seed=3101)
    rng = np.random.default_rng(1)
    # Legal random actions; reward must remain the environment's normalized utility.
    actions = []
    for mask in out['action_mask']:
        valid = np.flatnonzero(mask)
        actions.append(int(rng.choice(valid)))
    nxt = env.step(actions)
    assert 0.0 <= float(nxt['reward']) <= 1.0
    assert np.isfinite(float(nxt['reward']))
