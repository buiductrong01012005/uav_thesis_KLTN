import numpy as np

from src.envs.v4_fixed_mobile_battery_env import ACTIVE
from src.envs.thesis_cartesian_r6_curriculum_env import (
    CartesianR6Config,
    CartesianR6Env,
    CartesianR6BatteryConfig,
    CartesianR6BatteryEnv,
    CartesianR6BatteryFocusConfig,
    CartesianR6BatteryFocusEnv,
)
from src.envs.v7_r6_macro_relay_stable_env import (
    V7R6MacroRelayStableConfig,
    V7R6MacroRelayStableEnv,
)
from src.envs.v8_r6_macro_relay_battery_env import (
    V8R6MacroRelayBatteryConfig,
    V8R6MacroRelayBatteryEnv,
)
from src.envs.v9_r6_macro_relay_battery_focus_env import (
    V9R6MacroRelayBatteryFocusConfig,
    V9R6MacroRelayBatteryFocusEnv,
)
from src.baselines.thesis_exhaustive_role_target_greedy import (
    ExhaustiveRoleTargetGreedy,
)
from src.baselines.thesis_exhaustive_cartesian_role_greedy import (
    ExhaustiveCartesianRoleGreedy,
)


def test_cartesian_architecture_constant_across_stages():
    envs = [
        CartesianR6Env(CartesianR6Config(seed=1)),
        CartesianR6BatteryEnv(CartesianR6BatteryConfig(seed=1)),
        CartesianR6BatteryFocusEnv(CartesianR6BatteryFocusConfig(seed=1)),
    ]
    dims = []
    for e in envs:
        o = e.reset(seed=1)
        dims.append((o["obs"].shape, o["state"].shape, o["action_mask"].shape, e.n_actions))
    assert dims[0] == dims[1] == dims[2]
    assert envs[0].n_actions == 10


def test_target_aware_architecture_constant_across_stages():
    envs = [
        V7R6MacroRelayStableEnv(V7R6MacroRelayStableConfig(seed=1)),
        V8R6MacroRelayBatteryEnv(V8R6MacroRelayBatteryConfig(seed=1)),
        V9R6MacroRelayBatteryFocusEnv(V9R6MacroRelayBatteryFocusConfig(seed=1)),
    ]
    dims = []
    for e in envs:
        o = e.reset(seed=1)
        dims.append((o["obs"].shape, o["state"].shape, o["action_mask"].shape, e.n_actions))
    assert dims[0] == dims[1] == dims[2]
    assert envs[0].n_actions == 10


def test_cartesian_stage1_battery_off_stays_active():
    e = CartesianR6Env(CartesianR6Config(seed=3, horizon=10))
    _o = e.reset(seed=3)
    for _ in range(10):
        acts = [e.encode_action(4, 0)] * e.cfg.num_uavs
        _o = e.step(acts)
    assert np.all(e.status == ACTIVE)
    assert np.allclose(e.battery, e.cfg.max_battery)


def test_cartesian_focus_randomizes_residual_battery_in_range():
    e = CartesianR6BatteryFocusEnv(
        CartesianR6BatteryFocusConfig(
            seed=4,
            randomize_residual_battery=True,
            residual_battery_min=24,
            residual_battery_max=36,
        )
    )
    e.reset(seed=4)
    assert 24 <= e.sampled_residual_initial_battery <= 36
    assert np.isfinite(e.min_active_return_margin())


def test_exhaustive_policy_checks_all_role_vectors_for_full_active_fleet():
    e = V8R6MacroRelayBatteryEnv(V8R6MacroRelayBatteryConfig(seed=2101, horizon=2))
    o = e.reset(seed=2101)
    p = ExhaustiveRoleTargetGreedy(e)
    acts = p.act(o["action_mask"])
    assert len(acts) == e.cfg.num_uavs
    assert p.last_stats["role_vectors_evaluated"] == 32
    for i, a in enumerate(acts):
        assert o["action_mask"][i, a]


def test_exhaustive_cartesian_policy_checks_every_role_vector_and_valid_actions():
    e = CartesianR6BatteryEnv(CartesianR6BatteryConfig(seed=2101, horizon=2))
    o = e.reset(seed=2101)
    p = ExhaustiveCartesianRoleGreedy(e)
    acts = p.act(o["action_mask"])
    assert len(acts) == e.cfg.num_uavs
    assert p.last_stats["role_vectors_evaluated"] == 32
    assert p.last_stats["expected_role_vectors"] == 32
    # For five active UAVs the exhaustive family counts are C(5,r).
    assert p.last_stats["role_family_vector_counts"] == {
        "0": 1,
        "1": 5,
        "2": 10,
        "3": 10,
        "4": 5,
        "5": 1,
    }
    for i, a in enumerate(acts):
        assert o["action_mask"][i, a]


def test_action_formulations_use_same_observation_and_network_dimensions():
    c = CartesianR6Env(CartesianR6Config(seed=5))
    t = V7R6MacroRelayStableEnv(V7R6MacroRelayStableConfig(seed=5))
    co = c.reset(seed=5)
    to = t.reset(seed=5)
    assert co["obs"].shape == to["obs"].shape
    assert co["state"].shape == to["state"].shape
    assert c.n_actions == t.n_actions == 10
