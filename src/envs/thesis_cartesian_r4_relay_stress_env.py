"""Relay-stress variant of the thesis Cartesian environment.

Only the backhaul communication radius changes relative to the locked R6
thesis environment: 6 grid cells (300 m) -> 4 grid cells (200 m).  The map,
UE mobility, Access radius, battery lifecycle, action space, Shannon/pathloss
model, and opportunistic one-hop routing are inherited unchanged.

This module is intentionally separate so existing R6 experiments/checkpoints
cannot be silently reinterpreted as R4 results.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.envs.thesis_cartesian_r6_curriculum_env import (
    CartesianR6BatteryConfig,
    CartesianR6BatteryEnv,
)


@dataclass
class CartesianR4RelayStressBatteryConfig(CartesianR6BatteryConfig):
    """Battery+mobility thesis task with 200 m one-hop backhaul radius."""

    backhaul_radius_cells: float = 4.0


class CartesianR4RelayStressBatteryEnv(CartesianR6BatteryEnv):
    """Named environment for the 200 m relay-stress topology."""

    def __init__(self, config: CartesianR4RelayStressBatteryConfig | None = None):
        super().__init__(config or CartesianR4RelayStressBatteryConfig())
        self.cfg: CartesianR4RelayStressBatteryConfig
        if abs(float(self.cfg.backhaul_radius_cells) - 4.0) > 1e-12:
            raise ValueError("Relay-stress R4 environment requires backhaul_radius_cells=4.0")

    def reset(self, seed: int | None = None):
        out = super().reset(seed=seed)
        out["episode_meta"].update({
            "profile": "thesis_cartesian_r4_relay_stress",
            "relay_stress": True,
            "backhaul_radius_m": float(self.cfg.backhaul_radius_cells * self.cfg.cell_size_m),
            "reference_r6_backhaul_radius_m": 300.0,
        })
        return out
