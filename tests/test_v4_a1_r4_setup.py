from dataclasses import asdict
import numpy as np

from src.envs.v0_env import ACCESS, RELAY
from scripts.training.run_v4_a1_relay_dropout_100k import make_env as make_r5_env
from scripts.training.run_v4_a1_r4_100k import make_env, reset_env


def test_r4_changes_only_backhaul_radius_from_r5_a1():
    r4 = asdict(make_env().cfg)
    r5 = asdict(make_r5_env().cfg)
    diffs = {k: (r5[k], r4[k]) for k in r5 if r5[k] != r4[k]}
    assert diffs == {"backhaul_radius_cells": (5.0, 4.0)}


def test_r4_relay_dropout_initial_condition_rotates_identity():
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
        assert np.all(env.roles[np.arange(env.cfg.num_uavs) != residual] == ACCESS)


def test_r4_dynamic_role_actions_remain_unmasked():
    env = make_env()
    out = reset_env(env, 2)
    for i in range(env.cfg.num_uavs):
        roles = {env.decode_action(int(a))[1] for a in np.flatnonzero(out["action_mask"][i])}
        assert ACCESS in roles
        assert RELAY in roles
