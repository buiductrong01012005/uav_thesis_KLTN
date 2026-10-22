import numpy as np
import pytest

import scripts.run_relay_stress_r4_a5_qmix as a5
import scripts.run_relay_stress_r4_qmix as r4


def test_a5_keeps_r4_geometry_and_action_space():
    env = a5.make_env(3101, 5)
    assert env.cfg.backhaul_radius_cells == pytest.approx(4.0)
    assert env.cfg.backhaul_radius_cells * env.cfg.cell_size_m == pytest.approx(200.0)
    assert env.cfg.access_radius_cells == pytest.approx(4.0)
    assert env.n_actions == 10


def test_a5_keeps_same_training_protocol_as_r4():
    assert a5.TRAIN_SEEDS == r4.TRAIN_SEEDS
    assert a5.VALIDATION_SEEDS == r4.VALIDATION_SEEDS
    assert a5.TEST_SEEDS == r4.TEST_SEEDS
    assert a5.BLOCK_SPECS == r4.BLOCK_SPECS


def test_normalized_required_service_partial_and_capped():
    rates = np.array([0.0, 2.0, 4.0, 8.0])
    req = np.array([False, True, True, True])
    # demand=4 -> satisfaction [0.5, 1.0, 1.0]
    assert a5.normalized_required_service(rates, req, 4.0) == pytest.approx(2.5 / 3.0)


def test_normalized_required_service_zero_when_no_required_ues():
    assert a5.normalized_required_service(np.array([1.0, 2.0]), np.array([False, False]), 4.0) == 0.0


def test_a5_reward_formula_lambda_point_one():
    assert a5.shaped_training_reward(0.5, 0.25, 0.10) == pytest.approx(0.525)


def test_a5_environment_reward_remains_original_utility_but_metric_is_captured():
    env = a5.make_env(3101, 5)
    out = env.reset(seed=3101)
    # Stay+Access for all UAVs is action 1 in Cartesian encoding (move 0=Stay, role 1?);
    # avoid assuming encoding by picking the first legal action per UAV.
    actions = [int(np.flatnonzero(mask)[0]) for mask in out['action_mask']]
    nxt = env.step(actions)
    assert 0.0 <= float(nxt['reward']) <= 1.0
    assert 0.0 <= float(env.last_required_service_score) <= 1.0
    assert env.last_relay_required_count > 0
    assert env.last_relay_required_served_count >= 0


def test_seed3101_has_relay_required_ues_under_a5_geometry():
    env = a5.make_env(3101, 5)
    env.reset(seed=3101)
    mask = a5.relay_required_ue_mask(env, env.ue_positions, env._a5_direct_cells)
    assert mask.shape == (env.cfg.num_ues,)
    assert int(mask.sum()) > 0
