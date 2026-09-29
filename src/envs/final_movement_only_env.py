"""Independent 5-action movement-only environments for final thesis baselines.

These wrappers keep the exact V7/V8/V9 observation/state and physical dynamics,
but remove RelayTo from the policy action space entirely.  This is a genuinely
separate 5-action Spatial-QMIX baseline, not a 10-action V9 policy with relay
actions masked at evaluation time.
"""
from __future__ import annotations

import numpy as np

from src.envs.v0_env import MOVE_DELTAS
from src.envs.v4_fixed_mobile_battery_env import ACTIVE
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


class _MovementOnlyMixin:
    n_actions = 5

    def get_action_mask(self) -> np.ndarray:
        masks = np.zeros((self.cfg.num_uavs, 5), dtype=np.int8)
        for i in range(self.cfg.num_uavs):
            if self.status[i] != ACTIVE:
                masks[i, 4] = 1  # policy-level Stay placeholder
                continue
            for move_idx in range(5):
                cand = self.uav_positions[i] + MOVE_DELTAS[move_idx]
                if self._inside_grid(cand):
                    masks[i, move_idx] = 1
        return masks

    def _episode_meta(self) -> dict:
        meta = super()._episode_meta()
        meta.update({
            "final_baseline": "independent_movement_only_spatial_qmix",
            "policy_actions": "N/S/E/W/Stay only",
            "relay_actions_available": False,
        })
        return meta


class FinalMovementOnlyNoBatteryEnv(_MovementOnlyMixin, V7R6MacroRelayStableEnv):
    def __init__(self, config: V7R6MacroRelayStableConfig | None = None):
        cfg = config or V7R6MacroRelayStableConfig(enable_battery_cycle=False)
        if cfg.enable_battery_cycle:
            raise ValueError("No-battery baseline stage requires enable_battery_cycle=False")
        super().__init__(cfg)


class FinalMovementOnlyBatteryEnv(_MovementOnlyMixin, V8R6MacroRelayBatteryEnv):
    def __init__(self, config: V8R6MacroRelayBatteryConfig | None = None):
        super().__init__(config or V8R6MacroRelayBatteryConfig())


class FinalMovementOnlyBatteryFocusEnv(_MovementOnlyMixin, V9R6MacroRelayBatteryFocusEnv):
    def __init__(self, config: V9R6MacroRelayBatteryFocusConfig | None = None):
        super().__init__(config or V9R6MacroRelayBatteryFocusConfig())
