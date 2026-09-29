"""V8 R6 battery-aware macro-relay environment.

This stage keeps the V7 control semantics unchanged and only enables the
existing automatic battery lifecycle:

    Active -> Returning -> Charging -> Active

Policy action semantics remain:
    0..4 : Access movement N/S/E/W/Stay
    5+j  : RelayTo(UAV j), with RelayTo(self) masked

Returning/Charging UAVs are controlled by the environment and expose only the
policy-level Stay placeholder in the action mask.  A Relay macro lasts one slot;
there is no persistent assignment and no ReleaseRelay action.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.envs.v7_r6_macro_relay_stable_env import (
    V7R6MacroRelayStableConfig,
    V7R6MacroRelayStableEnv,
)


@dataclass
class V8R6MacroRelayBatteryConfig(V7R6MacroRelayStableConfig):
    # Keep the core V7 scenario and enable only the lifecycle extension.
    enable_battery_cycle: bool = True
    horizon: int = 150

    # Reuse the previously audited V4 lifecycle parameters.  Exactly one UAV
    # starts with residual energy, but its identity is randomized per episode.
    residual_initial_battery: float = 30.0
    randomize_residual_uav: bool = True
    max_battery: float = 100.0
    hover_drain: float = 0.1
    move_extra_drain: float = 0.1
    relay_radio_drain: float = 0.1
    access_radio_drain: float = 0.2
    returning_step_drain: float = 0.2
    return_safety_margin: float = 5.0
    charge_rate_per_slot: float = 5.0
    rejoin_threshold: float = 80.0


class V8R6MacroRelayBatteryEnv(V7R6MacroRelayStableEnv):
    """V7 macro-relay control with automatic battery cycling enabled."""

    def __init__(self, config: V8R6MacroRelayBatteryConfig | None = None):
        super().__init__(config or V8R6MacroRelayBatteryConfig())
        self.cfg: V8R6MacroRelayBatteryConfig
        if not self.cfg.enable_battery_cycle:
            raise ValueError("V8 battery environment requires enable_battery_cycle=True")

    def reset(self, seed: int | None = None):
        out = super().reset(seed=seed)
        out["episode_meta"].update({
            "profile": "v8_r6_macro_relay_battery",
            "battery_cycle": True,
            "battery_control": "environment-controlled Active->Returning->Charging->Active",
            "residual_initial_battery": float(self.cfg.residual_initial_battery),
            "randomize_residual_uav": bool(self.cfg.randomize_residual_uav),
            "rejoin_threshold": float(self.cfg.rejoin_threshold),
        })
        return out

    def _episode_meta(self) -> dict:
        meta = super()._episode_meta()
        meta.update({
            "control_version": "V8_R6_macro_relay_battery",
            "battery_cycle": True,
            "battery_control": "environment-controlled return/charge/rejoin",
            "battery_role": "robustness/availability extension; no proactive return action",
        })
        return meta
