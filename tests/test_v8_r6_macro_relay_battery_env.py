import numpy as np

from src.envs.v0_env import ACCESS
from src.envs.v4_fixed_mobile_battery_env import ACTIVE, CHARGING, RETURNING
from src.envs.v8_r6_macro_relay_battery_env import (
    V8R6MacroRelayBatteryConfig,
    V8R6MacroRelayBatteryEnv,
)


def make_env(*, horizon=150, randomize=False):
    return V8R6MacroRelayBatteryEnv(
        V8R6MacroRelayBatteryConfig(
            seed=1101,
            horizon=horizon,
            randomize_residual_uav=randomize,
        )
    )


def test_v8_keeps_v7_dimensions_and_10_action_semantics():
    env = make_env()
    out = env.reset(seed=1101)
    assert env.n_actions == 10
    assert out["state"].shape == (105,)
    assert out["obs"].shape == (5, 110)
    assert out["action_mask"].shape == (5, 10)
    assert np.all(env.roles == ACCESS)
    assert out["episode_meta"]["battery_cycle"] is True


def test_v8_preserves_residual_battery_initialization():
    env = make_env(randomize=False)
    env.reset(seed=1101)
    assert env.residual_uav_idx == 0
    assert env.battery[0] == env.cfg.residual_initial_battery
    assert np.allclose(env.battery[1:], env.cfg.max_battery)
    assert np.all(env.status == ACTIVE)


def test_nonactive_uav_exposes_only_policy_stay():
    env = make_env()
    env.reset(seed=1101)
    env.status[2] = RETURNING
    mask = env.get_action_mask()
    assert mask[2].sum() == 1
    assert mask[2, 4] == 1


def test_relay_target_must_be_active_and_request_open():
    env = make_env()
    env.reset(seed=1101)
    actor = 0
    target = 1
    env.request_open[:] = False
    env.relay_needed[:] = False
    env.request_open[target] = True
    env.relay_needed[target] = True
    env.status[target] = RETURNING
    mask = env.get_action_mask()
    assert mask[actor, 5 + target] == 0


def test_battery_lifecycle_return_charge_rejoin_occurs_within_horizon():
    env = make_env(horizon=150, randomize=False)
    out = env.reset(seed=1101)
    seen_return = seen_charge = seen_rejoin = False
    had_unavailable = False

    while True:
        out = env.step([4] * 5)
        if np.any(env.status == RETURNING):
            seen_return = True
            had_unavailable = True
        if np.any(env.status == CHARGING):
            seen_charge = True
            had_unavailable = True
        if had_unavailable and np.all(env.status == ACTIVE):
            seen_rejoin = True
        if out["truncated"]:
            break

    assert seen_return
    assert seen_charge
    assert seen_rejoin
    assert env.return_start_time[0] > 0
    assert env.station_arrival_time[0] > env.return_start_time[0]
    assert env.rejoin_time[0] > env.station_arrival_time[0]


def test_returning_uav_moves_automatically_even_policy_action_is_stay():
    env = make_env(horizon=150, randomize=False)
    env.reset(seed=1101)
    # Force a deterministic returning state away from the station.
    i = 0
    env.status[i] = RETURNING
    env.target_station_idx[i], _ = env._nearest_station(env.uav_positions[i])
    before = env.uav_positions[i].copy()
    env.step([4] * 5)
    after = env.uav_positions[i].copy()
    assert not np.array_equal(before, after)


def test_spatial_qmix_checkpoint_shape_compatible_between_v7_and_v8():
    from src.rl.v5_relay_request_spatial_qmix import V5RelayRequestSpatialQMIXLearner

    env = make_env()
    out = env.reset(seed=1101)
    learner = V5RelayRequestSpatialQMIXLearner(
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
    assert all(0 <= a < 10 for a in actions)
