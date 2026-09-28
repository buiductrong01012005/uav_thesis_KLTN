import numpy as np

from src.envs.v5_curriculum import DEFAULT_CURRICULUM, make_curriculum_env


def test_curriculum_keeps_same_action_and_observation_layout():
    envs = [make_curriculum_env(stage, 123, training=True) for stage in DEFAULT_CURRICULUM]
    dims = {(e.n_actions, e.obs_dim, e.state_dim, e.cfg.num_uavs) for e in envs}
    assert len(dims) == 1
    assert next(iter(dims))[0] == 10


def test_stage1_has_mobility_and_role_but_no_battery_events():
    stage = DEFAULT_CURRICULUM[0]
    env = make_curriculum_env(stage, 101, training=True)
    out = env.reset(seed=101)
    assert env.cfg.enable_ue_mobility
    assert not env.cfg.enable_battery_cycle
    assert env.cfg.horizon == 200
    assert np.allclose(env.battery, env.cfg.max_battery)

    start_centers = env.hotspot_centers.copy()
    saw_unavailable = False
    while True:
        out = env.step([4] * env.cfg.num_uavs)
        saw_unavailable |= bool(np.any(np.asarray(out["info"]["status"]) != 0))
        if out["truncated"]:
            break

    assert not saw_unavailable
    assert not np.allclose(start_centers, env.hotspot_centers)


def test_stage2_introduces_battery_lifecycle():
    stage = DEFAULT_CURRICULUM[1]
    env = make_curriculum_env(stage, 202, training=False)
    out = env.reset(seed=202)
    assert env.cfg.enable_battery_cycle
    assert env.cfg.horizon == 300
    assert min(env.battery) >= 70.0

    prev_status = env.status.copy()
    returns = 0
    rejoins = 0
    while True:
        out = env.step([4] * env.cfg.num_uavs)
        status = np.asarray(out["info"]["status"], dtype=int)
        returns += int(np.sum((prev_status == 0) & (status == 1)))
        rejoins += int(np.sum((prev_status != 0) & (status == 0)))
        prev_status = status.copy()
        if out["truncated"]:
            break

    assert returns >= 1
    assert rejoins >= 1


def test_stage3_is_full_long_horizon_profile():
    stage = DEFAULT_CURRICULUM[2]
    env = make_curriculum_env(stage, 303, training=True)
    out = env.reset(seed=303)
    assert env.cfg.enable_ue_mobility
    assert env.cfg.enable_battery_cycle
    assert env.cfg.horizon == 600
    assert min(env.battery) >= 55.0
    assert max(env.battery) <= 100.0
    assert env.cfg.randomize_tradeoff_phase
    assert out["episode_meta"]["profile"] == "v5_dynamic_tradeoff_single_topology"


def test_validation_phase_is_locked_for_comparable_stage_evaluation():
    stage2 = DEFAULT_CURRICULUM[1]
    a = make_curriculum_env(stage2, 77, training=False)
    b = make_curriculum_env(stage2, 88, training=False)
    a.reset(seed=77)
    b.reset(seed=88)
    assert a.tradeoff_phase_offset == 0
    assert b.tradeoff_phase_offset == 0
