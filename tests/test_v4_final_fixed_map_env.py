import numpy as np

from src.envs.v0_env import ACCESS, RELAY
from src.envs.v4_final_fixed_map_env import (
    ACTIVE,
    CHARGING,
    RETURNING,
    V4FinalFixedMapConfig,
    V4FinalFixedMapEnv,
)


def test_compact_observation_dimensions():
    env = V4FinalFixedMapEnv(V4FinalFixedMapConfig(seed=1))
    out = env.reset(seed=1)
    assert out["state"].shape == (60,)
    assert out["obs"].shape == (5, 65)


def test_round_role_energy_ratio_access_twice_relay_comm_cost():
    cfg = V4FinalFixedMapConfig(seed=2, randomize_residual_uav=False)
    env = V4FinalFixedMapEnv(cfg)
    env.reset(seed=2)
    # Prevent return and compare two otherwise identical stationary UAVs.
    env.battery[:] = 100.0
    env.roles[:] = ACCESS
    actions = [env.encode_action(4, ACCESS)] * 5
    before = env.battery.copy()
    env.step(actions)
    access_drain = before[0] - env.battery[0]

    env.reset(seed=2)
    env.battery[:] = 100.0
    env.roles[:] = RELAY
    actions = [env.encode_action(4, RELAY)] * 5
    before = env.battery.copy()
    env.step(actions)
    relay_drain = before[0] - env.battery[0]

    assert np.isclose(access_drain, 0.3, atol=1e-5)
    assert np.isclose(relay_drain, 0.2, atol=1e-5)
    assert np.isclose(cfg.access_radio_drain, 2 * cfg.relay_radio_drain)


def test_return_trigger_uses_distance_plus_reserve_not_fixed_threshold():
    cfg = V4FinalFixedMapConfig(seed=3, randomize_residual_uav=False)
    env = V4FinalFixedMapEnv(cfg)
    env.reset(seed=3)
    i = 0
    station_idx, dist = env._nearest_station(env.uav_positions[i])
    threshold = dist * cfg.returning_step_drain + cfg.return_safety_margin
    env.battery[:] = 100.0
    env.battery[i] = threshold + 0.31  # after stationary Access drain 0.3 it remains above
    actions = [env.encode_action(4, ACCESS)] * 5
    env.step(actions)
    assert env.status[i] == ACTIVE

    env.reset(seed=3)
    station_idx, dist = env._nearest_station(env.uav_positions[i])
    threshold = dist * cfg.returning_step_drain + cfg.return_safety_margin
    env.battery[:] = 100.0
    env.battery[i] = threshold + 0.29  # after 0.3 drain it crosses the dynamic threshold
    env.step(actions)
    assert env.status[i] == RETURNING
    assert env.target_station_idx[i] == station_idx


def test_residual_uav_identity_is_seeded_and_not_a_return_threshold():
    cfg = V4FinalFixedMapConfig(seed=4, residual_initial_battery=30.0)
    env = V4FinalFixedMapEnv(cfg)
    env.reset(seed=123)
    idx1 = env.residual_uav_idx
    batt1 = env.battery.copy()
    env.reset(seed=123)
    assert env.residual_uav_idx == idx1
    assert np.allclose(env.battery, batt1)
    assert np.sum(np.isclose(env.battery, 30.0)) == 1
    assert np.sum(np.isclose(env.battery, 100.0)) == 4
    # Starting at 30 does not itself force Returning.
    assert env.status[idx1] == ACTIVE


def test_full_return_charge_rejoin_cycle_occurs_within_horizon():
    cfg = V4FinalFixedMapConfig(seed=5, residual_initial_battery=30.0, horizon=150)
    env = V4FinalFixedMapEnv(cfg)
    env.reset(seed=5)
    target = env.residual_uav_idx
    seen_return = seen_charge = seen_rejoin = False
    for _ in range(cfg.horizon):
        actions = [env.encode_action(4, ACCESS)] * cfg.num_uavs
        out = env.step(actions)
        seen_return |= bool(env.status[target] == RETURNING)
        seen_charge |= bool(env.status[target] == CHARGING)
        if seen_charge and env.status[target] == ACTIVE:
            seen_rejoin = True
            break
    assert seen_return and seen_charge and seen_rejoin
    assert env.rejoin_time[target] > env.return_start_time[target] >= 0


def test_access_radius_rejects_far_ue_even_with_good_shannon_link():
    cfg = V4FinalFixedMapConfig(seed=6, num_ues=1, num_hotspots=1)
    env = V4FinalFixedMapEnv(cfg)
    env.reset(seed=6)
    env.status[:] = ACTIVE
    env.roles[:] = ACCESS
    # Put all UAVs in one corner and UE farther than R_A=4 cells.
    env.uav_positions[:] = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1], [0, 2]])
    env.ue_positions[:] = np.asarray([[10.0, 10.0]])
    net = env.compute_network()
    assert net["utility"] == 0.0
    assert all(len(v) == 0 for v in net["association"].values())


def test_final_mobility_uses_correlated_groups_with_individual_local_motion():
    cfg = V4FinalFixedMapConfig(seed=7, num_ues=30)
    env = V4FinalFixedMapEnv(cfg)
    env.reset(seed=7)
    local_speeds = env.ue_local_speed_mps()
    group_speeds = env.group_center_speed_mps()
    assert np.all(local_speeds >= cfg.ue_local_speed_min_mps - 1e-6)
    assert np.all(local_speeds <= cfg.ue_local_speed_max_mps + 1e-6)
    assert np.allclose(group_speeds, cfg.group_center_speed_mps, atol=1e-6)
    # Individual local headings/speeds differ even though UEs share group motion.
    unique = np.unique(np.round(env.ue_local_velocity_cells_per_slot, 5), axis=0)
    assert len(unique) > 10


def test_group_mobility_stays_coherent_and_inside_affected_region():
    cfg = V4FinalFixedMapConfig(seed=8, num_ues=50)
    env = V4FinalFixedMapEnv(cfg)
    env.reset(seed=8)
    p0 = env.ue_positions.copy()
    c0 = env.hotspot_centers.copy()
    for _ in range(80):
        env._advance_mobile_ues()
        cells = np.floor(env.ue_positions).astype(int)
        assert np.all(cells >= 0)
        assert np.all(cells < cfg.grid_size)
        assert all(env.affected_mask[tuple(c)] for c in cells)
        # No UE is allowed to diffuse arbitrarily far from its group reference.
        assert np.max(np.linalg.norm(env.ue_offsets, axis=1)) <= cfg.max_ue_offset_cells + 1e-6
    displacement = np.linalg.norm(env.ue_positions - p0, axis=1)
    center_disp = np.linalg.norm(env.hotspot_centers - c0, axis=1)
    assert np.mean(displacement) > 0.25
    assert np.mean(center_disp) > 0.1


def test_group_centers_drive_correlated_motion_but_are_not_observed_directly():
    cfg = V4FinalFixedMapConfig(seed=9, num_ues=30)
    env = V4FinalFixedMapEnv(cfg)
    out = env.reset(seed=9)
    centers_before = env.hotspot_centers.copy()
    positions_before = env.ue_positions.copy()
    obs_before = out["obs"].copy()
    env._advance_mobile_ues()
    assert not np.allclose(env.hotspot_centers, centers_before)
    assert not np.allclose(env.ue_positions, positions_before)
    # Observation remains compact: coarse demand + fleet state + agent id;
    # group centers are only an environment-side mobility generator.
    assert obs_before.shape[1] == 65
