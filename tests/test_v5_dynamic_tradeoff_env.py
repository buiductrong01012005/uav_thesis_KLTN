import numpy as np

from src.envs.v0_env import ACCESS
from src.envs.v5_dynamic_tradeoff_env import V5DynamicTradeoffConfig, V5DynamicTradeoffEnv


def make_cfg(**kw):
    base = dict(
        horizon=300,
        enable_ue_mobility=True,
        enable_battery_cycle=True,
        backhaul_radius_cells=4.0,
        initial_battery_min=35.0,
        initial_battery_max=100.0,
    )
    base.update(kw)
    return V5DynamicTradeoffConfig(**base)


def test_dynamic_profile_starts_all_access_and_randomizes_battery_phase():
    a = V5DynamicTradeoffEnv(make_cfg())
    oa = a.reset(seed=123)
    b = V5DynamicTradeoffEnv(make_cfg())
    ob = b.reset(seed=123)
    c = V5DynamicTradeoffEnv(make_cfg())
    oc = c.reset(seed=124)

    assert np.all(a.roles == ACCESS)
    assert np.all(a.status == 0)
    assert np.allclose(a.battery, b.battery)
    assert oa["episode_meta"]["initial_identity_permutation"] == ob["episode_meta"]["initial_identity_permutation"]
    assert (
        oa["episode_meta"]["initial_battery"] != oc["episode_meta"]["initial_battery"]
        or oa["episode_meta"]["initial_identity_permutation"] != oc["episode_meta"]["initial_identity_permutation"]
    )
    assert min(a.battery) >= 35.0 and max(a.battery) <= 100.0


def test_dynamic_profile_exposes_battery_return_and_rejoin_in_long_rollout():
    env = V5DynamicTradeoffEnv(make_cfg(horizon=300, initial_battery_min=20.0))
    out = env.reset(seed=321)
    saw_return = False
    saw_rejoin = False
    was_unavailable = np.zeros(env.cfg.num_uavs, dtype=bool)

    while not out.get("truncated", False):
        # Movement action 4 = Stay and preserves the current role.
        out = env.step([4] * env.cfg.num_uavs)
        status = np.asarray(out["info"]["status"], dtype=int)
        unavailable = status != 0
        saw_return |= bool(np.any(unavailable))
        saw_rejoin |= bool(np.any(was_unavailable & ~unavailable))
        was_unavailable |= unavailable

    assert saw_return
    assert saw_rejoin


def test_waypoint_mobility_reaches_both_tradeoff_endpoints():
    cfg = make_cfg(
        horizon=170,
        enable_battery_cycle=False,
        randomize_start_identity=False,
        randomize_tradeoff_phase=False,
        tradeoff_segment_slots=40,
    )
    env = V5DynamicTradeoffEnv(cfg)
    out = env.reset(seed=99)
    assert np.allclose(env.hotspot_centers, np.asarray(cfg.direct_friendly_centers, dtype=float))
    assert env.tradeoff_regime() == "direct_friendly"
    for _ in range(40):
        out = env.step([4] * env.cfg.num_uavs)
    assert np.allclose(env.hotspot_centers, np.asarray(cfg.relay_stressed_centers, dtype=float))
    assert out["info"]["tradeoff_regime"] == "relay_stressed"
    for _ in range(40):
        out = env.step([4] * env.cfg.num_uavs)
    assert np.allclose(env.hotspot_centers, np.asarray(cfg.direct_friendly_centers, dtype=float))
    assert out["info"]["tradeoff_regime"] == "direct_friendly"
