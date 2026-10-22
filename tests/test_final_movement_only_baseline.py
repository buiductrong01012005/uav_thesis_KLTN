import numpy as np

from src.envs.v0_env import ACCESS
from src.envs.final_movement_only_env import (
    FinalMovementOnlyNoBatteryEnv,
    FinalMovementOnlyBatteryFocusEnv,
)
from src.envs.v7_r6_macro_relay_stable_env import V7R6MacroRelayStableConfig
from src.envs.v9_r6_macro_relay_battery_focus_env import (
    V9R6MacroRelayBatteryFocusConfig,
)
from src.baselines.final_fast_greedy import FinalFastGreedyPolicy
from src.envs.v9_r6_macro_relay_battery_focus_env import V9R6MacroRelayBatteryFocusEnv


def test_movement_only_action_space_is_genuinely_five_actions():
    env = FinalMovementOnlyNoBatteryEnv(V7R6MacroRelayStableConfig(seed=1, horizon=10))
    out = env.reset(seed=1)
    assert env.n_actions == 5
    assert out["action_mask"].shape == (5, 5)
    assert np.all(out["action_mask"].sum(axis=1) >= 1)


def test_movement_only_active_uavs_never_become_relay():
    env = FinalMovementOnlyNoBatteryEnv(V7R6MacroRelayStableConfig(seed=2, horizon=10))
    out = env.reset(seed=2)
    for _ in range(5):
        actions = [int(np.flatnonzero(out["action_mask"][i])[0]) for i in range(5)]
        out = env.step(actions)
        assert np.all(env.roles[env.status == 0] == ACCESS)


def test_battery_focus_wrapper_keeps_five_action_masks_during_lifecycle():
    env = FinalMovementOnlyBatteryFocusEnv(V9R6MacroRelayBatteryFocusConfig(seed=3, horizon=150))
    out = env.reset(seed=3)
    saw_unavailable = False
    saw_rejoin = False
    for _ in range(150):
        mask = out["action_mask"]
        assert mask.shape == (5, 5)
        actions = [4 if mask[i, 4] else int(np.flatnonzero(mask[i])[0]) for i in range(5)]
        _before = env.status.copy()
        out = env.step(actions)
        if np.any(env.status != 0):
            saw_unavailable = True
        if saw_unavailable and np.all(env.status == 0):
            saw_rejoin = True
        if out["truncated"]:
            break
    assert saw_unavailable
    assert saw_rejoin


def test_fast_greedy_emits_only_valid_v9_actions():
    env = V9R6MacroRelayBatteryFocusEnv(V9R6MacroRelayBatteryFocusConfig(seed=4, horizon=20))
    out = env.reset(seed=4)
    policy = FinalFastGreedyPolicy(env, allow_relay=True)
    for _ in range(20):
        actions = policy.act(out["action_mask"])
        assert len(actions) == 5
        for i, a in enumerate(actions):
            assert out["action_mask"][i, a]
        out = env.step(actions)
        if out["truncated"]:
            break
