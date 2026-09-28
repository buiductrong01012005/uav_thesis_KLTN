import numpy as np

from src.envs.v0_env import ACCESS, RELAY
from src.envs.v4_fixed_mobile_battery_env import RETURNING
from src.envs.v4_toggle_role_env import (
    STAY,
    TOGGLE_ROLE,
    V4ToggleRoleA1Config,
    V4ToggleRoleA1Env,
    V4ToggleRoleEnv,
)
from src.envs.v4_final_fixed_map_env import V4FinalFixedMapConfig


def make_no_battery_env():
    env = V4ToggleRoleEnv(V4FinalFixedMapConfig(
        seed=1,
        enable_ue_mobility=False,
        enable_battery_cycle=False,
    ))
    env.reset(seed=1)
    return env


def test_toggle6_action_count_and_shapes():
    env = make_no_battery_env()
    out = env.reset(seed=1)
    assert env.n_actions == 6
    assert out["action_mask"].shape == (5, 6)
    assert out["obs"].shape == (5, 65)
    assert out["state"].shape == (60,)


def test_movement_preserves_role():
    env = make_no_battery_env()
    env.roles[:] = ACCESS
    env.roles[0] = RELAY
    before = env.roles.copy()
    env.step([STAY] * 5)
    np.testing.assert_array_equal(env.roles, before)


def test_toggle_changes_role_without_moving():
    env = make_no_battery_env()
    env.roles[:] = ACCESS
    before_pos = env.uav_positions.copy()
    env.step([TOGGLE_ROLE, STAY, STAY, STAY, STAY])
    assert env.roles[0] == RELAY
    np.testing.assert_array_equal(env.uav_positions[0], before_pos[0])


def test_toggle_is_reversible():
    env = make_no_battery_env()
    env.roles[:] = ACCESS
    env.step([TOGGLE_ROLE, STAY, STAY, STAY, STAY])
    assert env.roles[0] == RELAY
    env.step([TOGGLE_ROLE, STAY, STAY, STAY, STAY])
    assert env.roles[0] == ACCESS


def test_unavailable_uav_only_has_stay_action():
    env = make_no_battery_env()
    env.status[0] = RETURNING
    mask = env.get_action_mask()
    assert mask[0].sum() == 1
    assert mask[0, STAY] == 1
    assert mask[0, TOGGLE_ROLE] == 0


def test_a1_reset_has_residual_as_sole_relay_at_anchor():
    env = V4ToggleRoleA1Env(V4ToggleRoleA1Config(seed=1101, scenario_seed=81))
    out = env.reset_with_scenario(2, 70002, layout_seed=1101)
    assert env.residual_uav_idx == 2
    assert env.roles[2] == RELAY
    assert int(np.sum(env.roles == RELAY)) == 1
    assert env.uav_positions[2].tolist() == [11, 6]
    assert env.battery[2] == 30.0
    assert np.all(env.battery[[0, 1, 3, 4]] == 100.0)
    assert out["action_mask"].shape == (5, 6)


def test_a1_identity_permutation_preserves_start_cell_set():
    env = V4ToggleRoleA1Env(V4ToggleRoleA1Config(seed=1101, scenario_seed=81))
    env.reset_with_scenario(4, 70004, layout_seed=1101)
    got = {tuple(map(int, x)) for x in env.uav_positions}
    expected = {tuple(map(int, x)) for x in env.cfg.uav_start_cells}
    assert got == expected


def test_spatial_qmix_supports_six_actions_end_to_end():
    from src.rl.v4_hardradius_spatial_qmix import V4HardRadiusSpatialQMIXLearner

    env = V4ToggleRoleA1Env(V4ToggleRoleA1Config(seed=1101, scenario_seed=81))
    out = env.reset_with_scenario(0, 70000, layout_seed=1101)
    learner = V4HardRadiusSpatialQMIXLearner(
        env.cfg.num_uavs,
        env.obs_dim,
        env.state_dim,
        env.n_actions,
        grid_size=env.cfg.grid_size,
        coarse_size=env.cfg.coarse_obs_size,
        device="cpu",
    )
    actions = learner.act(out["obs"], out["action_mask"], 0.0, np.random.default_rng(1))
    assert len(actions) == 5
    assert all(0 <= a < 6 for a in actions)
    nxt = env.step(actions)
    assert nxt["action_mask"].shape == (5, 6)
