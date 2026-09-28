"""Single-topology V5 benchmark for joint dynamic-role + movement learning.

The terrestrial topology stays exactly the canonical V4/V5 map.  Instead of
building several maps whose static optimum is preselected, this profile makes
mobile UE groups traverse two valid regions of the *same* failed-BS Voronoi
cell:

* direct-friendly regime: five Access UAVs are generally preferable;
* backhaul-stressed regime: sacrificing one Access to create a Relay is
  generally preferable.

The group centers move smoothly back and forth between the two regimes while
individual UEs keep the canonical stochastic local motion around their center.
This makes the role optimum state-dependent and simultaneously makes continued
spatial repositioning useful.

Battery cycling is active in the training profile.  All UAVs start as Access;
initial battery phases and UAV identities over the same five start cells are
shuffled every episode.  Relay creation/release is therefore entirely a V5 RL
decision rather than an initialization rule.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from src.envs.v0_env import ACCESS
from src.envs.v4_fixed_mobile_battery_env import ACTIVE
from src.envs.v5_relay_request_env import V5RelayRequestConfig, V5RelayRequestEnv


@dataclass
class V5DynamicTradeoffConfig(V5RelayRequestConfig):
    horizon: int = 600
    enable_ue_mobility: bool = True
    enable_battery_cycle: bool = True
    backhaul_radius_cells: float = 4.0

    # Disable the old A1 "one residual UAV" initialization.  This profile uses
    # a shuffled staggered fleet battery vector instead.
    randomize_residual_uav: bool = False
    residual_initial_battery: float = 100.0
    initial_battery_min: float = 55.0
    initial_battery_max: float = 100.0
    randomize_start_identity: bool = True

    # Same-map traffic regimes found by environment-side necessity audit.
    # Every straight segment between corresponding group centers stays inside
    # the canonical failed-BS Voronoi region.
    direct_friendly_centers: tuple[tuple[float, float], ...] = (
        (11.0, 6.0), (12.0, 5.0), (10.0, 5.0)
    )
    relay_stressed_centers: tuple[tuple[float, float], ...] = (
        (8.0, 0.0), (10.0, 0.0), (12.0, 1.0)
    )
    tradeoff_segment_slots: int = 160
    randomize_tradeoff_phase: bool = True
    use_tradeoff_waypoint_mobility: bool = True


class V5DynamicTradeoffEnv(V5RelayRequestEnv):
    """One fixed topology whose traffic state changes the useful role mix."""

    def __init__(self, config: V5DynamicTradeoffConfig | None = None):
        super().__init__(config or V5DynamicTradeoffConfig())
        self.cfg: V5DynamicTradeoffConfig
        if not 0.0 < self.cfg.initial_battery_min <= self.cfg.initial_battery_max <= self.cfg.max_battery:
            raise ValueError("initial battery range must satisfy 0 < min <= max <= max_battery")
        if self.cfg.tradeoff_segment_slots < 1:
            raise ValueError("tradeoff_segment_slots must be >= 1")
        direct = np.asarray(self.cfg.direct_friendly_centers, dtype=np.float64)
        relay = np.asarray(self.cfg.relay_stressed_centers, dtype=np.float64)
        if direct.shape != (self.cfg.num_hotspots, 2) or relay.shape != (self.cfg.num_hotspots, 2):
            raise ValueError("tradeoff center sets must have shape (num_hotspots, 2)")
        for p in np.vstack([direct, relay]):
            if not self._center_is_valid_floor(p):
                raise ValueError(f"tradeoff center outside failed region: {p.tolist()}")
        # Validate the full straight-line corridor, not just endpoints.
        for h in range(self.cfg.num_hotspots):
            for alpha in np.linspace(0.0, 1.0, 101):
                p = (1.0 - alpha) * direct[h] + alpha * relay[h]
                if not self._center_is_valid_floor(p):
                    raise ValueError("tradeoff waypoint segment leaves failed-BS region")

        self.initial_identity_permutation = np.arange(self.cfg.num_uavs, dtype=np.int64)
        self.initial_battery_permutation = np.arange(self.cfg.num_uavs, dtype=np.int64)
        self.tradeoff_phase_offset = 0
        self.tradeoff_phase_step = 0

    @property
    def _direct_centers(self) -> np.ndarray:
        return np.asarray(self.cfg.direct_friendly_centers, dtype=np.float64)

    @property
    def _relay_centers(self) -> np.ndarray:
        return np.asarray(self.cfg.relay_stressed_centers, dtype=np.float64)

    def _centers_for_phase_step(self, phase_step: int) -> np.ndarray:
        seg = int(self.cfg.tradeoff_segment_slots)
        cycle = 2 * seg
        q = int(phase_step) % cycle
        alpha = (q / seg) if q <= seg else ((cycle - q) / seg)
        return (1.0 - alpha) * self._direct_centers + alpha * self._relay_centers

    def _set_tradeoff_centers(self) -> None:
        self.hotspot_centers[:] = self._centers_for_phase_step(
            self.tradeoff_phase_offset + self.tradeoff_phase_step
        )

    def reset(self, seed: int | None = None):
        out = super().reset(seed=seed)

        # All agents start as Access.  Relay creation is solely AcceptRelay().
        self.roles[:] = ACCESS
        self.status[:] = ACTIVE
        self.relay_target[:] = -1
        self.relay_assignment_age[:] = 0

        if self.cfg.randomize_start_identity:
            perm = self.rng.permutation(self.cfg.num_uavs)
        else:
            perm = np.arange(self.cfg.num_uavs, dtype=np.int64)
        self.initial_identity_permutation = np.asarray(perm, dtype=np.int64)
        self.uav_positions[:] = self.start_positions[self.initial_identity_permutation]

        levels = np.linspace(
            self.cfg.initial_battery_min,
            self.cfg.initial_battery_max,
            self.cfg.num_uavs,
            dtype=np.float64,
        )
        bperm = self.rng.permutation(self.cfg.num_uavs)
        self.initial_battery_permutation = np.asarray(bperm, dtype=np.int64)
        if self.cfg.enable_battery_cycle:
            self.battery[:] = levels[self.initial_battery_permutation]
        else:
            self.battery[:] = self.cfg.max_battery

        self.target_station_idx[:] = -1
        self.return_start_time[:] = -1
        self.station_arrival_time[:] = -1
        self.rejoin_time[:] = -1

        self.tradeoff_phase_step = 0
        cycle = 2 * int(self.cfg.tradeoff_segment_slots)
        self.tradeoff_phase_offset = (
            int(self.rng.integers(0, cycle))
            if self.cfg.use_tradeoff_waypoint_mobility and self.cfg.randomize_tradeoff_phase
            else 0
        )
        if self.cfg.use_tradeoff_waypoint_mobility:
            self._set_tradeoff_centers()
            self._project_group_positions_inside_region()

        self._refresh_request_state()
        out = self._refresh_output(out)
        out["episode_meta"].update({
            "profile": "v5_dynamic_tradeoff_single_topology",
            "all_access_initially": True,
            "ue_mobility": bool(self.cfg.enable_ue_mobility),
            "battery_cycle": bool(self.cfg.enable_battery_cycle),
            "initial_identity_permutation": self.initial_identity_permutation.astype(int).tolist(),
            "initial_battery": self.battery.astype(float).tolist(),
            "initial_battery_range": [
                float(self.cfg.initial_battery_min),
                float(self.cfg.initial_battery_max),
            ],
            "tradeoff_waypoints_enabled": bool(self.cfg.use_tradeoff_waypoint_mobility),
            "tradeoff_phase_offset": int(self.tradeoff_phase_offset),
            "tradeoff_segment_slots": int(self.cfg.tradeoff_segment_slots),
            "direct_friendly_centers": self._direct_centers.astype(float).tolist(),
            "relay_stressed_centers": self._relay_centers.astype(float).tolist(),
        })
        return out


    def tradeoff_alpha(self) -> float:
        """0=direct-friendly endpoint, 1=relay-stressed endpoint."""
        seg = int(self.cfg.tradeoff_segment_slots)
        cycle = 2 * seg
        q = int(self.tradeoff_phase_offset + self.tradeoff_phase_step) % cycle
        return float((q / seg) if q <= seg else ((cycle - q) / seg))

    def tradeoff_regime(self) -> str:
        a = self.tradeoff_alpha()
        if a <= 0.25:
            return "direct_friendly"
        if a >= 0.75:
            return "relay_stressed"
        return "transition"

    def step(self, actions):
        out = super().step(actions)
        out["info"]["tradeoff_alpha"] = self.tradeoff_alpha()
        out["info"]["tradeoff_regime"] = self.tradeoff_regime()
        out["info"]["tradeoff_phase_step"] = int(self.tradeoff_phase_step)
        return out

    def _advance_mobile_ues(self):
        if not self.cfg.enable_ue_mobility:
            return
        if not self.cfg.use_tradeoff_waypoint_mobility:
            return super()._advance_mobile_ues()

        # Smooth benchmark-level reference motion.  Exact UE locations are still
        # stochastic because every UE keeps its own local velocity and turning.
        self.tradeoff_phase_step += 1
        self._set_tradeoff_centers()

        local_turn = self.rng.normal(
            0.0, self.cfg.ue_local_turn_noise_std_rad, size=self.cfg.num_ues
        )
        self.ue_local_velocity_cells_per_slot = self._rotate_vectors(
            self.ue_local_velocity_cells_per_slot, local_turn
        )

        for u in range(self.cfg.num_ues):
            h = int(self.ue_hotspot[u])
            offset = self.ue_offsets[u].copy()
            vel = self.ue_local_velocity_cells_per_slot[u].copy()
            proposal_offset = offset + vel

            if np.linalg.norm(proposal_offset) > self.cfg.max_ue_offset_cells:
                norm = max(float(np.linalg.norm(offset)), 1e-9)
                radial = offset / norm
                vel = vel - 2.0 * np.dot(vel, radial) * radial
                proposal_offset = offset + vel
                if np.linalg.norm(proposal_offset) > self.cfg.max_ue_offset_cells:
                    nrm = max(float(np.linalg.norm(proposal_offset)), 1e-9)
                    proposal_offset *= self.cfg.max_ue_offset_cells / nrm

            absolute = self.hotspot_centers[h] + proposal_offset
            if not self._ue_position_is_valid(absolute):
                candidates = (
                    np.asarray([-vel[0], vel[1]]),
                    np.asarray([vel[0], -vel[1]]),
                    -vel,
                )
                accepted = False
                for cand_vel in candidates:
                    cand_offset = offset + cand_vel
                    if np.linalg.norm(cand_offset) > self.cfg.max_ue_offset_cells:
                        continue
                    cand_abs = self.hotspot_centers[h] + cand_offset
                    if self._ue_position_is_valid(cand_abs):
                        vel = cand_vel
                        proposal_offset = cand_offset
                        absolute = cand_abs
                        accepted = True
                        break
                if not accepted:
                    proposal_offset = np.zeros(2, dtype=np.float64)
                    absolute = self.hotspot_centers[h].copy()
                    vel = -vel

            self.ue_offsets[u] = proposal_offset
            self.ue_local_velocity_cells_per_slot[u] = vel
            self.ue_positions[u] = absolute
