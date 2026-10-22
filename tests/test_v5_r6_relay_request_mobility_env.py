import numpy as np

from src.envs.v0_env import ACCESS, RELAY
from src.envs.v5_relay_request_env import ACCEPT_BASE, RELEASE_RELAY, STAY
from src.envs.v5_r6_relay_request_mobility_env import (
    V5R6RelayRequestMobilityConfig,
    V5R6RelayRequestMobilityEnv,
)


def make_env(seed=11):
    return V5R6RelayRequestMobilityEnv(V5R6RelayRequestMobilityConfig(seed=seed, horizon=30))


def test_r6_profile_keeps_canonical_map_and_starts_all_access():
    env = make_env()
    out = env.reset(seed=11)
    assert env.cfg.backhaul_radius_cells == 6.0
    assert env.cfg.enable_ue_mobility is True
    assert env.cfg.enable_battery_cycle is False
    assert np.all(env.roles == ACCESS)
    assert np.all(env.relay_target == -1)
    assert out["episode_meta"]["waypoint_tradeoff_map"] is False


def test_r6_profile_has_open_relay_requests_without_handcrafted_waypoints():
    env = make_env()
    env.reset(seed=11)
    seen = False
    for _ in range(20):
        seen |= bool(np.any(env.request_open))
        env.step([STAY] * env.cfg.num_uavs)
    assert seen


def test_accept_and_release_work_on_r6_profile():
    env = make_env()
    _out = env.reset(seed=11)
    target = int(np.flatnonzero(env.request_open)[0])
    actor = next(i for i in range(env.cfg.num_uavs) if i != target)
    action = env.accept_action_for_target(actor, target)
    assert ACCEPT_BASE <= action < RELEASE_RELAY
    actions = [STAY] * env.cfg.num_uavs
    actions[actor] = action
    _out = env.step(actions)
    assert env.roles[actor] == RELAY
    assert env.relay_target[actor] == target

    # Dwell for the minimum assignment age, then release explicitly.
    for _ in range(env.cfg.relay_min_dwell_slots):
        env.step([STAY] * env.cfg.num_uavs)
    mask = env.get_action_mask()
    if env.roles[actor] == RELAY:
        assert mask[actor, RELEASE_RELAY] == 1
        actions = [STAY] * env.cfg.num_uavs
        actions[actor] = RELEASE_RELAY
        env.step(actions)
        assert env.roles[actor] == ACCESS
        assert env.relay_target[actor] == -1
