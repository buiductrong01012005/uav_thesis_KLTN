import numpy as np

from src.envs.v4_fixed_mobile_battery_env import ACTIVE
from src.envs.v9_r6_macro_relay_battery_focus_env import (
    V9R6MacroRelayBatteryFocusConfig,
    V9R6MacroRelayBatteryFocusEnv,
)
from scripts.run_v9_r6_battery_transition_curriculum import (
    _return_focus_state,
    sample_transition_focused_batch,
)
from src.rl.qmix import ReplayBuffer, Transition


def test_training_residual_battery_randomization_can_be_fixed_by_range():
    env = V9R6MacroRelayBatteryFocusEnv(
        V9R6MacroRelayBatteryFocusConfig(
            seed=11,
            randomize_residual_battery=True,
            residual_battery_min=27.0,
            residual_battery_max=27.0,
        )
    )
    out = env.reset(seed=11)
    rid = env.residual_uav_idx
    assert env.battery[rid] == 27.0
    assert out["episode_meta"]["sampled_residual_initial_battery"] == 27.0
    assert np.isclose(out["obs"][:, 25 + rid * env.UAV_FEAT + 3].max(), 0.27)


def test_standard_evaluation_condition_remains_battery_30():
    env = V9R6MacroRelayBatteryFocusEnv(
        V9R6MacroRelayBatteryFocusConfig(
            seed=12,
            randomize_residual_battery=False,
            residual_initial_battery=30.0,
        )
    )
    env.reset(seed=12)
    assert env.battery[env.residual_uav_idx] == 30.0


def test_return_margin_matches_environment_threshold():
    env = V9R6MacroRelayBatteryFocusEnv(
        V9R6MacroRelayBatteryFocusConfig(seed=13, randomize_residual_battery=False)
    )
    env.reset(seed=13)
    i = int(env.residual_uav_idx)
    assert env.status[i] == ACTIVE
    _, dist = env._nearest_station(env.uav_positions[i])
    threshold = dist * env.cfg.returning_step_drain + env.cfg.return_safety_margin
    assert np.isclose(env.return_energy_margin(i), env.battery[i] - threshold)


def test_focus_predicate_does_not_modify_action_mask():
    env = V9R6MacroRelayBatteryFocusEnv(
        V9R6MacroRelayBatteryFocusConfig(seed=14, randomize_residual_battery=False)
    )
    out = env.reset(seed=14)
    before = out["action_mask"].copy()
    _ = _return_focus_state(env, focus_margin=1000.0)
    after = env.get_action_mask()
    assert np.array_equal(before, after)
    assert after.shape == (5, 10)


def _dummy_transition(v: float):
    obs = np.full((5, 3), v, dtype=np.float32)
    state = np.full((4,), v, dtype=np.float32)
    actions = np.zeros(5, dtype=np.int64)
    mask = np.ones((5, 10), dtype=np.int8)
    return Transition(obs, state, actions, v, obs, state, mask, 0.0)


def test_transition_focused_sampler_returns_requested_batch_size():
    main = ReplayBuffer(100, 1)
    focus = ReplayBuffer(100, 2)
    for i in range(40):
        main.add(_dummy_transition(float(i)))
    for i in range(20):
        focus.add(_dummy_transition(float(100 + i)))
    batch, frac = sample_transition_focused_batch(main, focus, 32, 0.5)
    assert batch["reward"].shape[0] == 32
    assert np.isclose(frac, 0.5)


def test_training_battery_range_keeps_full_lifecycle_inside_horizon():
    from src.envs.v4_fixed_mobile_battery_env import RETURNING, CHARGING
    for b in (24.0, 36.0):
        env = V9R6MacroRelayBatteryFocusEnv(
            V9R6MacroRelayBatteryFocusConfig(
                seed=11,
                horizon=150,
                randomize_residual_battery=True,
                residual_battery_min=b,
                residual_battery_max=b,
            )
        )
        env.reset(seed=11)
        rid = int(env.residual_uav_idx)
        seen_return = seen_charge = seen_rejoin = False
        prev = env.status.copy()
        for _ in range(150):
            out = env.step([4] * env.cfg.num_uavs)
            now = env.status.copy()
            seen_return |= bool(prev[rid] == ACTIVE and now[rid] == RETURNING)
            seen_charge |= bool(prev[rid] == RETURNING and now[rid] == CHARGING)
            seen_rejoin |= bool(prev[rid] == CHARGING and now[rid] == ACTIVE)
            prev = now
            if out["truncated"]:
                break
        assert seen_return and seen_charge and seen_rejoin
