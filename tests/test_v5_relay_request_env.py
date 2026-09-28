import numpy as np

from src.envs.v0_env import ACCESS, RELAY
from src.envs.v4_fixed_mobile_battery_env import RETURNING
from src.envs.v5_relay_request_env import (
    ACCEPT_BASE,
    RELEASE_RELAY,
    STAY,
    V5RelayRequestA1Config,
    V5RelayRequestA1Env,
)


def make_env():
    env = V5RelayRequestA1Env(V5RelayRequestA1Config(seed=1101, scenario_seed=81))
    return env


def test_v5_shapes_and_action_count():
    env = make_env()
    out = env.reset_with_scenario(0, 70000, layout_seed=1101)
    assert env.n_actions == 10
    assert out["state"].shape == (105,)
    assert out["obs"].shape == (5, 110)
    assert out["action_mask"].shape == (5, 10)


def test_relative_accept_target_mapping_is_invertible():
    env = make_env()
    for i in range(5):
        others = [j for j in range(5) if j != i]
        for slot, target in enumerate(others):
            action = ACCEPT_BASE + slot
            assert env.target_for_accept_action(i, action) == target
            assert env.accept_action_for_target(i, target) == action


def test_request_hysteresis_without_reward_shaping(monkeypatch):
    env = make_env()
    env.reset_with_scenario(0, 70000, layout_seed=1101)
    env.roles[:] = ACCESS
    env.relay_target[:] = -1

    offer = np.asarray([40.0, 40.0, 0.0, 0.0, 0.0])
    direct_low = np.asarray([20.0, 32.0, 0.0, 0.0, 0.0])  # ratios .5 and .8
    deficit = np.asarray([0.5, 0.2, 0.0, 0.0, 0.0])
    monkeypatch.setattr(env, "_compute_request_metrics", lambda: (offer, direct_low, deficit))
    env.relay_needed[:] = False
    env._refresh_request_state()
    assert env.relay_needed[0]
    assert not env.relay_needed[1]  # .8 is above ON=.75

    direct_mid = direct_low.copy()
    direct_mid[0] = 34.0  # ratio .85: still ON because OFF=.90
    monkeypatch.setattr(env, "_compute_request_metrics", lambda: (offer, direct_mid, deficit))
    env._refresh_request_state()
    assert env.relay_needed[0]

    direct_high = direct_low.copy()
    direct_high[0] = 38.0  # ratio .95: turns OFF
    monkeypatch.setattr(env, "_compute_request_metrics", lambda: (offer, direct_high, deficit))
    env._refresh_request_state()
    assert not env.relay_needed[0]


def test_access_can_accept_open_request_persistently():
    env = make_env()
    env.reset_with_scenario(0, 70000, layout_seed=1101)

    # Force a clean request/action condition independent of geometry.
    env.roles[:] = ACCESS
    env.relay_target[:] = -1
    env.relay_assignment_age[:] = 0
    env.relay_needed[:] = False
    env.request_open[:] = False
    target = 1
    actor = 2
    env.relay_needed[target] = True
    env.request_open[target] = True
    action = env.accept_action_for_target(actor, target)
    mask = env.get_action_mask()
    assert mask[actor, action] == 1

    before = env.uav_positions.copy()
    actions = [STAY] * 5
    actions[actor] = action
    env.step(actions)
    assert env.roles[actor] == RELAY
    assert env.relay_target[actor] == target
    np.testing.assert_array_equal(env.uav_positions[actor], before[actor])


def test_relay_movement_preserves_assignment_and_release_is_explicit(monkeypatch):
    env = make_env()
    env.reset_with_scenario(0, 70000, layout_seed=1101)
    rid = 0
    target = int(env.relay_target[rid])
    assert target >= 0
    env.relay_assignment_age[rid] = env.cfg.relay_min_dwell_slots

    # Keep target structurally valid/requesting during this unit test.
    old_refresh = env._refresh_request_state
    monkeypatch.setattr(env, "_refresh_request_state", lambda: None)
    actions = [STAY] * 5
    env.step(actions)
    assert env.roles[rid] == RELAY
    assert env.relay_target[rid] == target

    mask = env.get_action_mask()
    assert mask[rid, RELEASE_RELAY] == 1
    actions[rid] = RELEASE_RELAY
    env.step(actions)
    assert env.roles[rid] == ACCESS
    assert env.relay_target[rid] == -1
    monkeypatch.setattr(env, "_refresh_request_state", old_refresh)


def test_unavailable_uav_only_has_stay():
    env = make_env()
    env.reset_with_scenario(0, 70000, layout_seed=1101)
    env.status[3] = RETURNING
    mask = env.get_action_mask()
    assert mask[3].sum() == 1
    assert mask[3, STAY] == 1


def test_a1_initial_low_battery_relay_has_primary_target_when_request_exists():
    env = make_env()
    out = env.reset_with_scenario(2, 70002, layout_seed=1101)
    assert env.roles[2] == RELAY
    assert env.uav_positions[2].tolist() == [11, 6]
    assert env.battery[2] == 30.0
    assert int(np.sum(env.roles == RELAY)) == 1
    # Locked R4 A1 geometry should contain at least one direct-BH-deficient Access.
    assert np.any(env.relay_needed)
    assert env.relay_target[2] >= 0
    assert out["episode_meta"]["initial_relay_target"] == int(env.relay_target[2])


def test_v5_spatial_qmix_end_to_end():
    from src.rl.v5_relay_request_spatial_qmix import V5RelayRequestSpatialQMIXLearner

    env = make_env()
    out = env.reset_with_scenario(0, 70000, layout_seed=1101)
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
    nxt = env.step(actions)
    assert nxt["obs"].shape == (5, 110)
    assert nxt["action_mask"].shape == (5, 10)
