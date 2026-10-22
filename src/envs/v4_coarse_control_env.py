"""Coarse-control variant of the final fixed-map V4 environment.

The physical/network map remains 20x20 at 50 m/cell.  The MARL controller,
however, acts on a 5x5 control lattice: one movement action advances one
control cell = 4 fine cells = 200 m.  With a 10 m/s UAV speed this corresponds
to a 20 s control interval.

The purpose is to make action resolution consistent with the 5x5 aggregated
traffic observation, while keeping exact UE positions and physical link/rate
calculation inside the environment.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from src.envs.v0_env import ACCESS, RELAY, MOVE_DELTAS
from src.envs.v4_fixed_mobile_battery_env import ACTIVE, RETURNING, CHARGING
from src.envs.v4_final_fixed_map_env import V4FinalFixedMapConfig, V4FinalFixedMapEnv


@dataclass
class V4CoarseControlConfig(V4FinalFixedMapConfig):
    # One control move = 4 x 50 m = 200 m.  At 10 m/s this is 20 s.
    control_stride_cells: int = 4
    slot_s: float = 20.0
    horizon: int = 40  # 800 s, close to the previous 150 x 5 s = 750 s

    # Distinct starting points on the 5x5 control lattice.  Coordinates remain
    # fine-grid indices, chosen near the affected BS3 region.
    uav_start_cells: tuple[tuple[int, int], ...] = (
        (9, 1),
        (9, 5),
        (13, 1),
        (5, 5),
        (13, 5),
    )

    # Energy parameters below are still defined per 5 s base interval in the
    # thesis model.  The environment scales them by slot_s / 5 at runtime.
    energy_base_slot_s: float = 5.0


class V4CoarseControlEnv(V4FinalFixedMapEnv):
    """Final V4 physics with a 5x5 MARL control lattice."""

    def __init__(self, config: V4CoarseControlConfig | None = None):
        super().__init__(config or V4CoarseControlConfig())
        self.cfg: V4CoarseControlConfig
        if self.cfg.grid_size // self.cfg.control_stride_cells != self.cfg.coarse_obs_size:
            raise ValueError("control stride must match coarse observation size")
        if self.cfg.slot_s <= 0 or self.cfg.energy_base_slot_s <= 0:
            raise ValueError("slot durations must be positive")

    @property
    def _energy_scale(self) -> float:
        return float(self.cfg.slot_s / self.cfg.energy_base_slot_s)

    def _move_delta(self, move_idx: int) -> np.ndarray:
        return MOVE_DELTAS[int(move_idx)] * int(self.cfg.control_stride_cells)

    def _coarse_step_toward(self, pos: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Move up to one coarse-control step toward a charging station."""
        p = pos.copy()
        stride = int(self.cfg.control_stride_cells)
        dr = int(target[0] - p[0])
        dc = int(target[1] - p[1])
        if abs(dc) >= abs(dr) and dc != 0:
            p[1] += int(np.sign(dc)) * min(stride, abs(dc))
        elif dr != 0:
            p[0] += int(np.sign(dr)) * min(stride, abs(dr))
        return p

    def _nearest_station(self, cell: np.ndarray) -> tuple[int, int]:
        # Return distance is expressed in coarse control steps, not fine cells.
        d_fine = np.abs(self.charging_stations - cell).sum(axis=1)
        idx = int(np.argmin(d_fine))
        steps = int(math.ceil(float(d_fine[idx]) / float(self.cfg.control_stride_cells)))
        return idx, steps

    def get_action_mask(self) -> np.ndarray:
        masks = np.ones((self.cfg.num_uavs, self.n_actions), dtype=np.int8)
        noop = self.encode_action(4, ACCESS)
        for i in range(self.cfg.num_uavs):
            if self.status[i] != ACTIVE:
                masks[i, :] = 0
                masks[i, noop] = 1
                continue
            for a in range(self.n_actions):
                move_idx, _ = self.decode_action(a)
                cand = self.uav_positions[i] + self._move_delta(move_idx)
                if not self._inside_grid(cand):
                    masks[i, a] = 0
        return masks

    def step(self, actions):
        if len(actions) != self.cfg.num_uavs:
            raise ValueError(f"Expected {self.cfg.num_uavs} actions")

        status_before = self.status.copy()
        positions_before = self.uav_positions.copy()
        roles_before = self.roles.copy()

        proposed = self.uav_positions.copy()
        proposed_roles = self.roles.copy()
        moved_active = np.zeros(self.cfg.num_uavs, dtype=bool)

        for i in range(self.cfg.num_uavs):
            if status_before[i] == ACTIVE:
                move_idx, role = self.decode_action(int(actions[i]))
                cand = self.uav_positions[i] + self._move_delta(move_idx)
                if self._inside_grid(cand):
                    proposed[i] = cand
                    moved_active[i] = not np.array_equal(cand, self.uav_positions[i])
                proposed_roles[i] = role
            elif status_before[i] == RETURNING:
                station = self.charging_stations[self.target_station_idx[i]]
                proposed[i] = self._coarse_step_toward(self.uav_positions[i], station)

        self._resolve_collisions(proposed, positions_before, status_before)
        self.uav_positions = proposed
        self.roles = proposed_roles

        for i in np.flatnonzero(status_before == RETURNING):
            station = self.charging_stations[self.target_station_idx[i]]
            if np.array_equal(self.uav_positions[i], station):
                self.status[i] = CHARGING
                if self.station_arrival_time[i] < 0:
                    self.station_arrival_time[i] = self.t + 1

        net = self.compute_network()
        utility = float(net["utility"])

        # Preserve the original per-5-s energy model in physical time by scaling
        # all drains/charge by the control interval ratio (20 / 5 = 4 by default).
        es = self._energy_scale
        for i in range(self.cfg.num_uavs):
            if status_before[i] == ACTIVE:
                drain = self.cfg.hover_drain
                if moved_active[i]:
                    drain += self.cfg.move_extra_drain
                drain += (
                    self.cfg.access_radio_drain
                    if self.roles[i] == ACCESS
                    else self.cfg.relay_radio_drain
                )
                self.battery[i] = max(0.0, float(self.battery[i] - es * drain))
            elif status_before[i] == RETURNING:
                self.battery[i] = max(
                    0.0, float(self.battery[i] - es * self.cfg.returning_step_drain)
                )
            elif status_before[i] == CHARGING:
                self.battery[i] = min(
                    self.cfg.max_battery,
                    float(self.battery[i] + es * self.cfg.charge_rate_per_slot),
                )

        for i in np.flatnonzero(status_before == CHARGING):
            if self.battery[i] >= self.cfg.rejoin_threshold:
                self.status[i] = ACTIVE
                self.roles[i] = ACCESS
                self.target_station_idx[i] = -1
                if self.rejoin_time[i] < 0:
                    self.rejoin_time[i] = self.t + 1

        for i in range(self.cfg.num_uavs):
            if self.status[i] != ACTIVE:
                continue
            station_idx, dist_steps = self._nearest_station(self.uav_positions[i])
            energy_to_station = dist_steps * es * self.cfg.returning_step_drain
            threshold = energy_to_station + self.cfg.return_safety_margin
            if self.battery[i] <= threshold:
                self.status[i] = RETURNING
                self.target_station_idx[i] = station_idx
                if self.return_start_time[i] < 0:
                    self.return_start_time[i] = self.t + 1

        self._advance_mobile_ues()
        self.t += 1
        truncated = self.t >= self.cfg.horizon

        info = {
            "t": self.t,
            "utility": utility,
            "mean_rate_mbps": float(np.mean(net["ue_rate_mbps"])),
            "sum_served_mbps": float(
                np.sum(np.minimum(net["ue_rate_mbps"], self.cfg.ue_demand_mbps))
            ),
            "satisfied_fraction": float(
                np.mean(net["ue_rate_mbps"] >= self.cfg.ue_demand_mbps - 1e-9)
            ),
            "num_access": int(np.sum((self.roles == ACCESS) & (self.status == ACTIVE))),
            "num_relay": int(np.sum((self.roles == RELAY) & (self.status == ACTIVE))),
            "active_count": int(np.sum(self.status == ACTIVE)),
            "status": self.status.astype(int).tolist(),
            "battery": self.battery.astype(float).tolist(),
            "roles": self.roles.astype(int).tolist(),
            "uav_positions": self.uav_positions.astype(int).tolist(),
            "return_start_time": self.return_start_time.astype(int).tolist(),
            "station_arrival_time": self.station_arrival_time.astype(int).tolist(),
            "rejoin_time": self.rejoin_time.astype(int).tolist(),
            "routes": net["routes"],
            "access_loads": net["access_loads"],
            "backhaul_caps_mbps": net["backhaul_caps_mbps"],
            "role_switches_this_slot": int(
                np.sum((roles_before != self.roles) & (status_before == ACTIVE))
            ),
            "control_stride_cells": int(self.cfg.control_stride_cells),
            "control_interval_s": float(self.cfg.slot_s),
        }
        out = {
            "obs": self.get_obs(),
            "state": self.get_state(),
            "action_mask": self.get_action_mask(),
            "reward": utility,
            "terminated": False,
            "truncated": truncated,
            "info": info,
        }

        if not self.cfg.enable_battery_cycle:
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
            out["info"]["battery"] = self.battery.astype(float).tolist()
            out["info"]["active_count"] = self.cfg.num_uavs

        return out
