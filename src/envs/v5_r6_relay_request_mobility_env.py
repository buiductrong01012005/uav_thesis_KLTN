"""R6 + V5 relay-request control on the canonical V4 map.

Purpose
-------
Keep the original physical topology and the canonical correlated UE-mobility
model.  Only the control abstraction is replaced by V5 persistent
Relay(target) assignments.

Physical changes from locked V4 core:
- backhaul radius: 6 cells (300 m)
- UE mobility: ON
- battery lifecycle: OFF

Control actions stay at 10 actions/agent:
- 0..4  : N/S/E/W/Stay, preserving current persistent assignment
- 5..8  : AcceptRelay(other UAV target)
- 9     : ReleaseRelay

All UAVs start as Access.  A Relay can only be created by an RL AcceptRelay
choice on an open request.  This class deliberately does *not* use the V5
waypoint trade-off map; hotspot motion is the canonical stochastic V4 model.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.envs.v0_env import ACCESS
from src.envs.v4_fixed_mobile_battery_env import ACTIVE
from src.envs.v5_relay_request_env import V5RelayRequestConfig, V5RelayRequestEnv


@dataclass
class V5R6RelayRequestMobilityConfig(V5RelayRequestConfig):
    horizon: int = 150
    enable_ue_mobility: bool = True
    enable_battery_cycle: bool = False
    backhaul_radius_cells: float = 6.0

    # Keep the request semantics from the supplied V5 implementation.
    relay_request_on_ratio: float = 0.75
    relay_request_off_ratio: float = 0.90
    relay_request_min_load_mbps: float = 1.0
    relay_min_dwell_slots: int = 2


class V5R6RelayRequestMobilityEnv(V5RelayRequestEnv):
    """Canonical map + natural mobility + persistent target-aware Relay control."""

    def __init__(self, config: V5R6RelayRequestMobilityConfig | None = None):
        super().__init__(config or V5R6RelayRequestMobilityConfig())
        self.cfg: V5R6RelayRequestMobilityConfig

    def reset(self, seed: int | None = None):
        out = super().reset(seed=seed)

        # The policy starts from a neutral 5-Access state.  This avoids baking a
        # particular Relay identity into initialization and makes every Relay an
        # explicit AcceptRelay(target) decision.
        self.roles[:] = ACCESS
        self.status[:] = ACTIVE
        self.relay_target[:] = -1
        self.relay_assignment_age[:] = 0

        # Battery is intentionally removed from this learnability experiment.
        self.battery[:] = self.cfg.max_battery
        self.target_station_idx[:] = -1
        self.return_start_time[:] = -1
        self.station_arrival_time[:] = -1
        self.rejoin_time[:] = -1

        self._refresh_request_state()
        out = self._refresh_output(out)
        out["episode_meta"].update({
            "profile": "v5_r6_relay_request_natural_mobility",
            "canonical_v4_topology": True,
            "waypoint_tradeoff_map": False,
            "all_access_initially": True,
            "backhaul_radius_cells": float(self.cfg.backhaul_radius_cells),
            "ue_mobility": True,
            "battery_cycle": False,
        })
        return out
