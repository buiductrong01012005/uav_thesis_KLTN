import numpy as np

from src.envs.v0_env import ACCESS, RELAY
from scripts.training.run_v4_a1_relay_dropout_100k import (
    initial_roles_for_residual,
    make_env,
    reset_env,
)


def test_initial_roles_follow_residual_identity():
    for residual in range(5):
        roles = initial_roles_for_residual(residual, 5)
        assert roles[residual] == RELAY
        assert np.sum(roles == RELAY) == 1
        assert np.all(roles[np.arange(5) != residual] == ACCESS)


def test_reset_keeps_static_ues_but_rotates_low_battery_relay():
    env = make_env()
    baseline_ues = None
    for residual in range(env.cfg.num_uavs):
        reset_env(env, residual)
        if baseline_ues is None:
            baseline_ues = env.ue_positions.copy()
        else:
            assert np.array_equal(env.ue_positions, baseline_ues)
        assert env.residual_uav_idx == residual
        assert env.battery[residual] == env.cfg.residual_initial_battery
        assert env.roles[residual] == RELAY
        assert np.sum(env.roles == RELAY) == 1


def test_dynamic_role_actions_are_not_masked():
    env = make_env()
    out = reset_env(env, 2)
    for i in range(env.cfg.num_uavs):
        roles = {env.decode_action(int(a))[1] for a in np.flatnonzero(out["action_mask"][i])}
        assert ACCESS in roles
        assert RELAY in roles
