"""Final fixed-map V4 environment candidate for the UAV thesis.

Design choices intentionally kept simple/defendable:
- physical world: 20x20 grid (50 m/cell), one fixed failed BS, fixed charging stations
- policy observation: 5x5 coarse traffic map + exact fleet state + agent id
- 5 UAVs, action = movement x {Access, Relay} (10 actions)
- one-hop relay only
- link feasibility by horizontal coverage radius; feasible links use distance/path-loss/Shannon rate
- finite Access association capacity and shared Relay backhaul capacity
- normalized battery model with role-dependent communication drain
- automatic return-to-charge when remaining energy is just sufficient to reach a station plus reserve

This module is separate from earlier V4 audit environments so old results remain reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from src.envs.v0_env import ACCESS, RELAY
from src.envs.v4_fixed_mobile_battery_env import (
    ACTIVE,
    CHARGING,
    RETURNING,
    V4FixedMobileBatteryConfig,
    V4FixedMobileBatteryEnv,
)


@dataclass
class V4FinalFixedMapConfig(V4FixedMobileBatteryConfig):
    # Long enough for one complete return -> charge -> rejoin cycle.
    horizon: int = 150

    # Hybrid coverage + Shannon model. Radii are horizontal distances.
    access_radius_cells: float = 4.0  # 200 m
    backhaul_radius_cells: float = 5.0  # 250 m for A->BS, A->R, R->BS

    # Battery: normalized units with round, ratio-based parameters.
    max_battery: float = 100.0
    hover_drain: float = 0.1
    move_extra_drain: float = 0.1
    relay_radio_drain: float = 0.1
    access_radio_drain: float = 0.2  # Access communication = 2 x Relay
    returning_step_drain: float = 0.2  # hover + movement, no communication
    return_safety_margin: float = 5.0
    charge_rate_per_slot: float = 5.0
    rejoin_threshold: float = 80.0

    # Experimental initial condition, NOT a return threshold.
    # One randomly selected UAV starts with residual energy so that a full
    # battery event appears within a 150-slot training episode.
    residual_initial_battery: float = 30.0
    randomize_residual_uav: bool = True

    # Observation aggregation only; physical world stays 20x20.
    coarse_obs_size: int = 5

    # Correlated group mobility (Reference-Point-Group style): each traffic
    # group has a slowly moving reference center, while every UE also moves
    # independently around that center. All UEs remain inside the failed-BS
    # service region. Speeds are physical and converted to cells/slot.
    group_center_speed_mps: float = 0.5
    group_turn_noise_std_rad: float = 0.04
    ue_local_speed_min_mps: float = 0.5
    ue_local_speed_max_mps: float = 1.0
    ue_local_turn_noise_std_rad: float = 0.12
    group_spread_cells: float = 1.5  # 75 m typical initial spread
    max_ue_offset_cells: float = 2.0  # keep a group spatially coherent

    # Ablation toggles. These do not change the system model; they let us add
    # dynamics one factor at a time and justify each component experimentally.
    enable_ue_mobility: bool = True
    enable_battery_cycle: bool = True


class V4FinalFixedMapEnv(V4FixedMobileBatteryEnv):
    """Fixed-map thesis environment with compact observation and hybrid radio model."""

    def __init__(self, config: V4FinalFixedMapConfig | None = None):
        super().__init__(config or V4FinalFixedMapConfig())
        self.cfg: V4FinalFixedMapConfig
        if self.cfg.grid_size % self.cfg.coarse_obs_size != 0:
            raise ValueError("grid_size must be divisible by coarse_obs_size")
        self.residual_uav_idx = 0
        self.ue_local_velocity_cells_per_slot = np.zeros((self.cfg.num_ues, 2), dtype=np.float64)

    # ------------------------------ reset/battery -----------------------------

    def reset(self, seed: int | None = None):
        out = super().reset(seed=seed)
        # Do not use a fixed low-battery UAV identity: otherwise a policy can
        # memorize which agent will leave rather than observing battery state.
        if self.cfg.randomize_residual_uav:
            self.residual_uav_idx = int(self.rng.integers(0, self.cfg.num_uavs))
        else:
            self.residual_uav_idx = 0
        self.battery[:] = self.cfg.max_battery
        if self.cfg.enable_battery_cycle:
            self.battery[self.residual_uav_idx] = self.cfg.residual_initial_battery
        # Refresh returned tensors because super().reset built them before the
        # experimental initial battery assignment above.
        out["obs"] = self.get_obs()
        out["state"] = self.get_state()
        out["episode_meta"] = self._episode_meta()
        return out

    # ------------------------------- UE mobility ------------------------------

    def _init_mobile_ues(self):
        """Initialize correlated moving groups with per-UE local motion.

        Group centers are only a mobility generator; they are never exposed to
        the policy. The environment keeps exact continuous UE positions, while
        the policy observes only the aggregated 5x5 demand map.
        """
        # Reuse the base reset-time spatial clustering and fixed group labels.
        super()._init_mobile_ues()

        # Use the final-core spread rather than the older audit default.
        self.ue_offsets = self.rng.normal(
            0.0, self.cfg.group_spread_cells, size=(self.cfg.num_ues, 2)
        )
        self._clip_group_offsets()

        # Slow reference-center motion: 0.5 m/s by default.
        n_groups = self.cfg.num_hotspots
        center_angles = self.rng.uniform(0.0, 2.0 * math.pi, size=n_groups)
        center_speed_cells = (
            self.cfg.group_center_speed_mps * self.cfg.slot_s / self.cfg.cell_size_m
        )
        self.hotspot_vel = (
            np.stack([np.sin(center_angles), np.cos(center_angles)], axis=1) * center_speed_cells
        )

        # Every UE has its own local heading/speed around the moving reference.
        local_speed_mps = self.rng.uniform(
            self.cfg.ue_local_speed_min_mps,
            self.cfg.ue_local_speed_max_mps,
            size=self.cfg.num_ues,
        )
        local_angles = self.rng.uniform(0.0, 2.0 * math.pi, size=self.cfg.num_ues)
        local_speed_cells = local_speed_mps * self.cfg.slot_s / self.cfg.cell_size_m
        self.ue_local_velocity_cells_per_slot = (
            np.stack([np.sin(local_angles), np.cos(local_angles)], axis=1)
            * local_speed_cells[:, None]
        )

        self._project_group_positions_inside_region()

    def _advance_mobile_ues(self):
        """Advance reference centers and individual UE motion by one slot.

        Centers move slowly and persistently. Each UE keeps an independent
        local velocity relative to its center. A hard relative-distance bound
        prevents the group from diffusing away, and all absolute UE positions
        are kept inside the failed-BS Voronoi service region.
        """
        if not self.cfg.enable_ue_mobility:
            return

        # 1) Persistent, slowly turning reference centers.
        turns = self.rng.normal(0.0, self.cfg.group_turn_noise_std_rad, size=self.cfg.num_hotspots)
        self.hotspot_vel = self._rotate_vectors(self.hotspot_vel, turns)
        for h in range(self.cfg.num_hotspots):
            cur = self.hotspot_centers[h].copy()
            vel = self.hotspot_vel[h].copy()
            proposal = cur + vel
            if not self._center_is_valid_floor(proposal):
                candidates = (
                    np.asarray([-vel[0], vel[1]]),
                    np.asarray([vel[0], -vel[1]]),
                    -vel,
                )
                accepted = False
                for cand_vel in candidates:
                    cand = cur + cand_vel
                    if self._center_is_valid_floor(cand):
                        vel = cand_vel
                        proposal = cand
                        accepted = True
                        break
                if not accepted:
                    proposal = cur
                    vel = -vel
            self.hotspot_vel[h] = vel
            self.hotspot_centers[h] = proposal

        # 2) Independent local motion around the reference center.
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

            # Reflect the local velocity at the relative group boundary.
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
                # Try simple local reflections first.
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
                    # Fall back to the center; center itself is guaranteed valid.
                    proposal_offset = np.zeros(2, dtype=np.float64)
                    absolute = self.hotspot_centers[h].copy()
                    vel = -vel

            self.ue_offsets[u] = proposal_offset
            self.ue_local_velocity_cells_per_slot[u] = vel
            self.ue_positions[u] = absolute

    @staticmethod
    def _rotate_vectors(vectors: np.ndarray, angles: np.ndarray) -> np.ndarray:
        c = np.cos(angles)
        s = np.sin(angles)
        x = vectors[:, 0].copy()
        y = vectors[:, 1].copy()
        out = vectors.copy()
        out[:, 0] = c * x - s * y
        out[:, 1] = s * x + c * y
        return out

    def _clip_group_offsets(self):
        norms = np.linalg.norm(self.ue_offsets, axis=1)
        bad = norms > self.cfg.max_ue_offset_cells
        if np.any(bad):
            self.ue_offsets[bad] *= (self.cfg.max_ue_offset_cells / norms[bad])[:, None]

    def _project_group_positions_inside_region(self):
        for u in range(self.cfg.num_ues):
            h = int(self.ue_hotspot[u])
            pos = self.hotspot_centers[h] + self.ue_offsets[u]
            if not self._ue_position_is_valid(pos):
                self.ue_offsets[u] = 0.0
                pos = self.hotspot_centers[h].copy()
            self.ue_positions[u] = pos

    def _ue_position_is_valid(self, pos: np.ndarray) -> bool:
        if np.any(pos < 0.0) or np.any(pos >= self.cfg.grid_size):
            return False
        cell = np.floor(pos).astype(int)
        cell = np.clip(cell, 0, self.cfg.grid_size - 1)
        return bool(self.affected_mask[tuple(cell)])

    def _center_is_valid_floor(self, center: np.ndarray) -> bool:
        return self._ue_position_is_valid(center)

    def ue_local_speed_mps(self) -> np.ndarray:
        return (
            np.linalg.norm(self.ue_local_velocity_cells_per_slot, axis=1)
            * self.cfg.cell_size_m
            / self.cfg.slot_s
        )

    def group_center_speed_mps(self) -> np.ndarray:
        return np.linalg.norm(self.hotspot_vel, axis=1) * self.cfg.cell_size_m / self.cfg.slot_s

    # ------------------------------ compact state -----------------------------

    def _coarse_demand_map(self) -> np.ndarray:
        """Aggregate UE demand from physical 20x20 cells into a 5x5 observation map.

        Values represent normalized UE count per coarse region. The environment
        still retains every UE's detailed position for association/rate logic.
        """
        g = self.cfg.grid_size
        c = self.cfg.coarse_obs_size
        block = g // c
        counts = np.zeros((c, c), dtype=np.float32)
        for p in self.ue_positions:
            fine = np.floor(p).astype(int)
            fine = np.clip(fine, 0, g - 1)
            cr = min(c - 1, int(fine[0] // block))
            cc = min(c - 1, int(fine[1] // block))
            counts[cr, cc] += 1.0
        # Fixed denominator preserves absolute-load information and does not
        # renormalize each snapshot by its own maximum.
        return np.clip(counts / float(self.cfg.max_ues_per_access), 0.0, 4.0)

    def get_state(self) -> np.ndarray:
        """60-D global state: 5x5 traffic + exact states of five UAVs.

        Per UAV: x, y, role, battery, Active, Returning, Charging = 7 values.
        Static BS/station topology is known a priori and intentionally omitted.
        """
        gden = max(self.cfg.grid_size - 1, 1)
        feats: list[float] = self._coarse_demand_map().ravel().astype(np.float32).tolist()
        for i in range(self.cfg.num_uavs):
            feats.extend((self.uav_positions[i] / gden).astype(np.float32).tolist())
            # Role is meaningful only while Active; unavailable UAVs use 0.
            if self.status[i] == ACTIVE:
                feats.append(float(self.roles[i] == RELAY))  # Access=0, Relay=1
            else:
                feats.append(0.0)
            feats.append(float(self.battery[i] / self.cfg.max_battery))
            feats.extend(
                [
                    float(self.status[i] == ACTIVE),
                    float(self.status[i] == RETURNING),
                    float(self.status[i] == CHARGING),
                ]
            )
        return np.asarray(feats, dtype=np.float32)

    def get_obs(self) -> np.ndarray:
        state = self.get_state()
        out = []
        for i in range(self.cfg.num_uavs):
            aid = np.zeros(self.cfg.num_uavs, dtype=np.float32)
            aid[i] = 1.0
            out.append(np.concatenate([state, aid]))
        return np.stack(out, axis=0)

    def step(self, actions):
        out = super().step(actions)
        if not self.cfg.enable_battery_cycle:
            # Battery-free ablation: no UAV availability change and no hidden timer.
            self.battery[:] = self.cfg.max_battery
            self.status[:] = ACTIVE
            self.target_station_idx[:] = -1
            self.return_start_time[:] = -1
            self.station_arrival_time[:] = -1
            self.rejoin_time[:] = -1
            out["obs"] = self.get_obs()
            out["state"] = self.get_state()
            out["action_mask"] = self.get_action_mask()
            out["info"]["status"] = self.status.astype(int).tolist()
            out["info"]["status_names"] = ["Active"] * self.cfg.num_uavs
            out["info"]["battery"] = self.battery.astype(float).tolist()
            out["info"]["active_count"] = self.cfg.num_uavs
        return out

    # --------------------------- hybrid radio model ---------------------------

    def _within_radius_cells(self, p1_grid, p2_grid, radius_cells: float) -> bool:
        p1 = np.asarray(p1_grid, dtype=np.float64)
        p2 = np.asarray(p2_grid, dtype=np.float64)
        return bool(np.linalg.norm(p1 - p2) <= float(radius_cells) + 1e-9)

    def _xy_within_radius_m(self, p1_xy, p2_xy, radius_cells: float) -> bool:
        return bool(
            np.linalg.norm(
                np.asarray(p1_xy, dtype=np.float64) - np.asarray(p2_xy, dtype=np.float64)
            )
            <= radius_cells * self.cfg.cell_size_m + 1e-9
        )

    def compute_network(self) -> dict:
        """One-hop deterministic allocation with radius feasibility + Shannon rates."""
        M = self.cfg.num_ues
        active = self.status == ACTIVE
        access_ids = np.flatnonzero(active & (self.roles == ACCESS))
        relay_ids = np.flatnonzero(active & (self.roles == RELAY))
        ue_rate = np.zeros(M, dtype=np.float64)
        if len(access_ids) == 0:
            return self._network_result(ue_rate, {}, {}, {})

        # UE->Access feasibility uses horizontal access radius. Within the
        # radius, association score and achieved rate still depend on Shannon.
        scores = np.full((M, len(access_ids)), -np.inf, dtype=np.float64)
        for jj, aid in enumerate(access_ids):
            uav_grid = self.uav_positions[aid].astype(np.float64)
            uav_xy = self._cell_to_xy(uav_grid)
            for u in range(M):
                ue_grid = self.ue_positions[u]
                if not self._within_radius_cells(ue_grid, uav_grid, self.cfg.access_radius_cells):
                    continue
                ue_xy = self._cell_to_xy(ue_grid)
                pl = self._a2g_path_loss_db(
                    ue_xy,
                    self.cfg.ue_height_m,
                    uav_xy,
                    self.cfg.uav_altitude_m,
                    self.cfg.access_fc_hz,
                )
                scores[u, jj] = self.cfg.access_tx_dbm - pl

        # Prioritize UEs with fewer/genuinely ambiguous feasible choices.
        feasible_count = np.isfinite(scores).sum(axis=1)
        if len(access_ids) > 1:
            safe_scores = np.where(np.isfinite(scores), scores, -1e9)
            top2 = np.partition(safe_scores, -2, axis=1)[:, -2:]
            ambiguity = np.abs(top2[:, 1] - top2[:, 0])
            ue_order = np.lexsort((ambiguity, feasible_count))
        else:
            ue_order = np.argsort(feasible_count)

        assoc: dict[int, list[int]] = {int(a): [] for a in access_ids}
        for u in ue_order:
            if feasible_count[u] == 0:
                continue
            pref = np.argsort(scores[u])[::-1]
            for jj in pref:
                if not np.isfinite(scores[u, jj]):
                    continue
                aid = int(access_ids[jj])
                if len(assoc[aid]) < self.cfg.max_ues_per_access:
                    assoc[aid].append(int(u))
                    break

        # Access rates: equal bandwidth and power sharing among associated UEs.
        access_offer: dict[int, float] = {}
        access_ue_rates: dict[int, dict[int, float]] = {}
        for aid in access_ids:
            users = assoc[int(aid)]
            if not users:
                access_offer[int(aid)] = 0.0
                access_ue_rates[int(aid)] = {}
                continue
            n_u = len(users)
            bw = self.cfg.access_bw_hz / n_u
            tx_dbm_per_ue = self.cfg.access_tx_dbm - 10.0 * math.log10(n_u)
            rates: dict[int, float] = {}
            uav_xy = self._cell_to_xy(self.uav_positions[aid])
            for u in users:
                ue_xy = self._cell_to_xy(self.ue_positions[u])
                pl = self._a2g_path_loss_db(
                    ue_xy,
                    self.cfg.ue_height_m,
                    uav_xy,
                    self.cfg.uav_altitude_m,
                    self.cfg.access_fc_hz,
                )
                rates[u] = self._shannon_rate_mbps(tx_dbm_per_ue - pl, bw)
            access_ue_rates[int(aid)] = rates
            access_offer[int(aid)] = float(
                sum(min(r, self.cfg.ue_demand_mbps) for r in rates.values())
            )

        # Relay->BS feasible links and Shannon capacities.
        relay_bs_cap: dict[int, float] = {}
        for rid in relay_ids:
            rxy = self._cell_to_xy(self.uav_positions[rid])
            caps = []
            for bsxy in self.surviving_bs_xy:
                if self._xy_within_radius_m(rxy, bsxy, self.cfg.backhaul_radius_cells):
                    caps.append(
                        self._bh_a2g_rate(rxy, self.cfg.uav_altitude_m, bsxy, self.cfg.bs_height_m)
                    )
            relay_bs_cap[int(rid)] = max(caps) if caps else 0.0

        # Each Access chooses the best one-hop option: direct or via one Relay.
        direct_cap: dict[int, float] = {}
        relay_first_cap: dict[tuple[int, int], float] = {}
        routes: dict[int, int | None] = {}
        for aid in access_ids:
            axy = self._cell_to_xy(self.uav_positions[aid])
            direct_caps = []
            for bsxy in self.surviving_bs_xy:
                if self._xy_within_radius_m(axy, bsxy, self.cfg.backhaul_radius_cells):
                    direct_caps.append(
                        self._bh_a2g_rate(axy, self.cfg.uav_altitude_m, bsxy, self.cfg.bs_height_m)
                    )
            dcap = max(direct_caps) if direct_caps else 0.0
            direct_cap[int(aid)] = dcap
            best_cap = dcap
            best_r: int | None = None
            for rid in relay_ids:
                if relay_bs_cap[int(rid)] <= 0.0:
                    continue
                rxy = self._cell_to_xy(self.uav_positions[rid])
                if not self._xy_within_radius_m(axy, rxy, self.cfg.backhaul_radius_cells):
                    continue
                c_ar = self._bh_a2a_rate(axy, rxy)
                relay_first_cap[(int(aid), int(rid))] = c_ar
                candidate = min(c_ar, relay_bs_cap[int(rid)])
                if candidate > best_cap + 1e-9:
                    best_cap = candidate
                    best_r = int(rid)
            routes[int(aid)] = best_r

        # Direct capacities plus shared Relay second hop.
        bh_cap: dict[int, float] = {}
        for aid in access_ids:
            if routes[int(aid)] is None:
                bh_cap[int(aid)] = direct_cap[int(aid)]

        for rid in relay_ids:
            clients = [int(a) for a in access_ids if routes[int(a)] == int(rid)]
            if not clients:
                continue
            weights = np.asarray(
                [max(1e-6, min(access_offer[a], relay_first_cap[(a, int(rid))])) for a in clients],
                dtype=np.float64,
            )
            shares = relay_bs_cap[int(rid)] * weights / weights.sum()
            for a, share in zip(clients, shares):
                bh_cap[a] = min(relay_first_cap[(a, int(rid))], float(share))

        # Backhaul-limited per-UE service.
        for aid in access_ids:
            offer = access_offer[int(aid)]
            if offer <= 1e-12:
                continue
            scale = min(1.0, bh_cap.get(int(aid), 0.0) / offer)
            for u, access_rate in access_ue_rates[int(aid)].items():
                ue_rate[u] = min(self.cfg.ue_demand_mbps, access_rate) * scale

        return self._network_result(ue_rate, assoc, routes, bh_cap)

    def _episode_meta(self) -> dict:
        meta = super()._episode_meta()
        meta.update(
            {
                "hybrid_radius_shannon": True,
                "access_radius_cells": float(self.cfg.access_radius_cells),
                "backhaul_radius_cells": float(self.cfg.backhaul_radius_cells),
                "coarse_obs_size": int(self.cfg.coarse_obs_size),
                "residual_uav_idx": int(getattr(self, "residual_uav_idx", -1)),
                "residual_initial_battery": float(self.cfg.residual_initial_battery),
                "mobility_model": {
                    "type": "correlated_moving_groups_with_individual_local_motion",
                    "num_groups": int(self.cfg.num_hotspots),
                    "group_center_speed_mps": float(self.cfg.group_center_speed_mps),
                    "ue_local_speed_mps": [
                        float(self.cfg.ue_local_speed_min_mps),
                        float(self.cfg.ue_local_speed_max_mps),
                    ],
                    "group_spread_cells": float(self.cfg.group_spread_cells),
                    "max_ue_offset_cells": float(self.cfg.max_ue_offset_cells),
                    "confined_to_failed_bs_region": True,
                },
                "battery_model": {
                    "hover": float(self.cfg.hover_drain),
                    "move_extra": float(self.cfg.move_extra_drain),
                    "access_radio": float(self.cfg.access_radio_drain),
                    "relay_radio": float(self.cfg.relay_radio_drain),
                    "return_step": float(self.cfg.returning_step_drain),
                    "reserve": float(self.cfg.return_safety_margin),
                    "charge_per_slot": float(self.cfg.charge_rate_per_slot),
                    "rejoin_threshold": float(self.cfg.rejoin_threshold),
                },
            }
        )
        return meta
