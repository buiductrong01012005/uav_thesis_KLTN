"""V9 battery-transition curriculum environment.

V9 does *not* change the deployed control semantics from V8.  It only adds
training-time residual-battery randomization so the learner cannot key on a
nearly fixed departure time.  Evaluation can keep the canonical residual
battery at 30 exactly as in V8.

Policy actions remain:
    0..4 : Access movement N/S/E/W/Stay
    5+j  : RelayTo(UAV j), RelayTo(self) masked

Battery lifecycle remains environment-controlled:
    Active -> Returning -> Charging -> Active
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from src.envs.v4_fixed_mobile_battery_env import ACTIVE
from src.envs.v8_r6_macro_relay_battery_env import (
    V8R6MacroRelayBatteryConfig,
    V8R6MacroRelayBatteryEnv,
)


@dataclass
class V9R6MacroRelayBatteryFocusConfig(V8R6MacroRelayBatteryConfig):
    # Training-only domain randomization.  Final validation/test leaves this
    # disabled and uses residual_initial_battery=30, matching V8 exactly.
    randomize_residual_battery: bool = False
    residual_battery_min: float = 24.0
    residual_battery_max: float = 36.0


class V9R6MacroRelayBatteryFocusEnv(V8R6MacroRelayBatteryEnv):
    """V8 dynamics plus optional training-time residual-battery randomization."""

    def __init__(self, config: V9R6MacroRelayBatteryFocusConfig | None = None):
        super().__init__(config or V9R6MacroRelayBatteryFocusConfig())
        self.cfg: V9R6MacroRelayBatteryFocusConfig
        if self.cfg.residual_battery_min <= 0.0:
            raise ValueError("residual_battery_min must be positive")
        if self.cfg.residual_battery_max < self.cfg.residual_battery_min:
            raise ValueError("residual_battery_max must be >= residual_battery_min")
        self.sampled_residual_initial_battery = float(self.cfg.residual_initial_battery)

    def reset(self, seed: int | None = None):
        out = super().reset(seed=seed)
        if self.cfg.randomize_residual_battery:
            sampled = float(
                self.rng.uniform(
                    self.cfg.residual_battery_min,
                    self.cfg.residual_battery_max,
                )
            )
            self.battery[int(self.residual_uav_idx)] = sampled
            self.sampled_residual_initial_battery = sampled
            # Battery is part of obs/state, so refresh after the training-only
            # randomized initial condition.
            out = self._refresh_output(out)
        else:
            self.sampled_residual_initial_battery = float(self.cfg.residual_initial_battery)

        out["episode_meta"].update(
            {
                "profile": "v9_r6_macro_relay_battery_focus",
                "residual_battery_randomized_for_training": bool(
                    self.cfg.randomize_residual_battery
                ),
                "sampled_residual_initial_battery": float(self.sampled_residual_initial_battery),
                "residual_battery_training_range": [
                    float(self.cfg.residual_battery_min),
                    float(self.cfg.residual_battery_max),
                ],
            }
        )
        return out

    def return_energy_margin(self, agent_id: int) -> float:
        """Battery above the environment's automatic return trigger threshold.

        This is a diagnostic/training-sampling quantity only; it does not mask
        or force any policy action.
        """
        i = int(agent_id)
        if self.status[i] != ACTIVE:
            return float("nan")
        _, dist_steps = self._nearest_station(self.uav_positions[i])
        threshold = float(dist_steps) * float(self.cfg.returning_step_drain) + float(
            self.cfg.return_safety_margin
        )
        return float(self.battery[i] - threshold)

    def min_active_return_margin(self) -> float:
        vals = [
            self.return_energy_margin(i)
            for i in range(self.cfg.num_uavs)
            if self.status[i] == ACTIVE
        ]
        return float(np.min(vals)) if vals else float("nan")

    def _episode_meta(self) -> dict:
        meta = super()._episode_meta()
        meta.update(
            {
                "control_version": "V9_R6_battery_transition_curriculum",
                "policy_action_changed_from_v8": False,
                "reward_changed_from_v8": False,
                "training_only_residual_battery_randomization": bool(
                    self.cfg.randomize_residual_battery
                ),
            }
        )
        return meta
