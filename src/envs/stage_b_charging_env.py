"""Stage-B recovery with an automatic return/charge/rejoin cycle (V3).

V3 extends the V2 battery-drop environment while deliberately keeping the
learned action space unchanged:
    movement + {Access, Relay}

The environment, not the policy, handles:
    Active -> Returning -> Charging -> Active

This version isolates *reactive reintegration* rather than anticipatory energy
management. Battery level, charging countdown, and charging-station location
are not exposed to the learned policy. A non-active UAV is encoded exactly as
in V2 (role one-hot [0, 0]), preserving the Stage-B/V2 observation dimension
and allowing direct checkpoint transfer.

The return threshold is interpreted as a reserved-energy trigger: once the
threshold is reached, mission control is removed and the UAV automatically
flies a shortest Manhattan path to the charging station. Return propulsion is
not optimized in V3.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.envs.stage_b_recovery_env import StageBConfig, StageBRecoveryEnv
from src.envs.v0_env import ACCESS, RELAY, MOVE_DELTAS

ACTIVE = 0
RETURNING = 1
CHARGING = 2

STATUS_NAMES = {
    ACTIVE: "Active",
    RETURNING: "Returning",
    CHARGING: "Charging",
}


@dataclass
class StageBChargingConfig(StageBConfig):
    # UAV 1 reaches the return threshold after seven active slots by default.
    initial_battery: tuple[float, ...] = (100.0, 10.0, 100.0)
    max_battery: float = 100.0
    base_drain_per_slot: float = 1.0
    return_threshold: float = 3.0

    # A single shared station is sufficient for the thesis sanity model.
    charging_station: tuple[int, int] = (4, 4)
    charge_duration: int = 4


class StageBChargingEnv(StageBRecoveryEnv):
    """Recovery environment with deterministic battery cycling.

    Only ACTIVE UAVs are controlled by the policy and can provide Access/Relay
    service. RETURNING and CHARGING UAVs have a single masked NoOp action.
    Returning UAVs move automatically one grid cell per slot toward the fixed
    charging station. Charging completes after `charge_duration` full slots;
    the UAV then rejoins the active fleet on the next decision step.
    """

    def __init__(self, config: StageBChargingConfig):
        if len(config.initial_battery) != config.num_uavs:
            raise ValueError("initial_battery must provide one value per UAV")
        if config.charge_duration < 1:
            raise ValueError("charge_duration must be >= 1")

        self.charging_cfg = config
        self.battery = np.asarray(config.initial_battery, dtype=np.float32)
        self.status = np.full(config.num_uavs, ACTIVE, dtype=np.int64)
        self.active = np.ones(config.num_uavs, dtype=bool)
        self.charge_remaining = np.zeros(config.num_uavs, dtype=np.int64)
        self.return_start_time = np.full(config.num_uavs, -1, dtype=np.int64)
        self.station_arrival_time = np.full(config.num_uavs, -1, dtype=np.int64)
        self.rejoin_time = np.full(config.num_uavs, -1, dtype=np.int64)
        self.charging_station = np.asarray(config.charging_station, dtype=np.int64)
        super().__init__(config)

    def reset(self):
        self.battery = np.asarray(self.charging_cfg.initial_battery, dtype=np.float32).copy()
        self.status = np.full(self.cfg.num_uavs, ACTIVE, dtype=np.int64)
        self.active = np.ones(self.cfg.num_uavs, dtype=bool)
        self.charge_remaining = np.zeros(self.cfg.num_uavs, dtype=np.int64)
        self.return_start_time = np.full(self.cfg.num_uavs, -1, dtype=np.int64)
        self.station_arrival_time = np.full(self.cfg.num_uavs, -1, dtype=np.int64)
        self.rejoin_time = np.full(self.cfg.num_uavs, -1, dtype=np.int64)
        return super().reset()

    def step(self, actions: Sequence[int]):
        if len(actions) != self.cfg.num_uavs:
            raise ValueError(f"Expected {self.cfg.num_uavs} actions")

        # Remember who was returning/charging at the beginning of the slot.
        # Newly triggered return/charging states do not consume a phase slot
        # until the next environment step.
        returning_before = self.status == RETURNING
        charging_before = self.status == CHARGING

        # 1) Policy controls ACTIVE UAVs only.
        new_positions = self.uav_positions.copy()
        new_roles = self.roles.copy()
        for i, action in enumerate(actions):
            if self.status[i] != ACTIVE:
                continue
            move_idx, role = self.decode_action(int(action))
            candidate = self.uav_positions[i] + MOVE_DELTAS[move_idx]
            if self._inside_grid(candidate):
                new_positions[i] = candidate
            new_roles[i] = role

        self.uav_positions = new_positions
        self.roles = new_roles

        # 2) Returning UAVs are moved automatically by the environment.
        for i in np.flatnonzero(returning_before):
            self.uav_positions[i] = self._one_step_to_station(self.uav_positions[i])
            if np.array_equal(self.uav_positions[i], self.charging_station):
                self.status[i] = CHARGING
                self.charge_remaining[i] = int(self.charging_cfg.charge_duration)
                self.station_arrival_time[i] = self.t + 1

        self._sync_active()

        # 3) Network service uses only ACTIVE UAVs.
        served_mask, valid_access, relay_chains = self.compute_served_users()
        served_count = int(served_mask.sum())
        reward = float(served_count / self.discovered_ue_count)

        self.t += 1

        # 4) Active-mission battery drain may trigger automatic return.
        for i in range(self.cfg.num_uavs):
            if self.status[i] != ACTIVE:
                continue
            self.battery[i] -= float(self.charging_cfg.base_drain_per_slot)
            if self.battery[i] <= float(self.charging_cfg.return_threshold):
                self.status[i] = RETURNING
                self.return_start_time[i] = self.t

        # 5) UAVs that were already charging at the start of this slot make
        # one unit of charging progress. Newly arrived UAVs start next slot.
        for i in np.flatnonzero(charging_before):
            if self.status[i] != CHARGING:
                continue
            self.charge_remaining[i] -= 1
            if self.charge_remaining[i] <= 0:
                self.charge_remaining[i] = 0
                self.status[i] = ACTIVE
                self.battery[i] = float(self.charging_cfg.max_battery)
                # Give a deterministic valid role until the policy acts on the
                # next slot. It is not rewarded in the current slot because
                # service was already computed above.
                self.roles[i] = ACCESS
                self.rejoin_time[i] = self.t

        self._sync_active()

        terminated = False
        truncated = self.t >= self.cfg.horizon

        info = {
            "t": self.t,
            "served_count": served_count,
            "served_ratio": reward,
            "served_discovered_ratio": reward,
            "served_total_ratio": float(served_count / self.stage_b_cfg.total_true_ues),
            "information_ceiling": float(self.information_ceiling),
            "valid_access_count": int(valid_access.sum()),
            "relay_chains": relay_chains,
            "num_access": int(np.sum((self.roles == ACCESS) & self.active)),
            "num_relay": int(np.sum((self.roles == RELAY) & self.active)),
            "active_mask": self.active.astype(np.int8).tolist(),
            "active_count": int(self.active.sum()),
            "status": self.status.astype(int).tolist(),
            "status_names": [STATUS_NAMES[int(s)] for s in self.status],
            "battery": self.battery.astype(float).tolist(),
            "charge_remaining": self.charge_remaining.astype(int).tolist(),
            "return_start_time": self.return_start_time.astype(int).tolist(),
            "station_arrival_time": self.station_arrival_time.astype(int).tolist(),
            "rejoin_time": self.rejoin_time.astype(int).tolist(),
            "charging_station": self.charging_station.astype(int).tolist(),
        }

        return {
            "obs": self.get_obs(),
            "state": self.get_state(),
            "action_mask": self.get_action_mask(),
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": info,
        }

    def compute_backhaul(self):
        valid_access = np.zeros(self.cfg.num_uavs, dtype=bool)
        relay_chains: list[tuple[int, int | None]] = []

        access_ids = np.flatnonzero((self.roles == ACCESS) & self.active)
        relay_ids = np.flatnonzero((self.roles == RELAY) & self.active)

        for access_id in access_ids:
            access_pos = self.uav_positions[access_id]
            if self._distance(access_pos, self.bs_position) <= self.cfg.backhaul_range:
                valid_access[access_id] = True
                relay_chains.append((int(access_id), None))
                continue

            for relay_id in relay_ids:
                relay_pos = self.uav_positions[relay_id]
                if (
                    self._distance(access_pos, relay_pos) <= self.cfg.backhaul_range
                    and self._distance(relay_pos, self.bs_position) <= self.cfg.backhaul_range
                ):
                    valid_access[access_id] = True
                    relay_chains.append((int(access_id), int(relay_id)))
                    break

        return valid_access, relay_chains

    def get_state(self) -> np.ndarray:
        """Keep Stage-B/V2 dimensionality for direct transfer learning.

        Each UAV contributes normalized position + two role bits. A UAV that is
        Returning or Charging uses role bits [0, 0]. The exact battery and
        countdown are intentionally hidden in V3.
        """
        features: list[float] = []
        denom = max(self.cfg.grid_size - 1, 1)

        for i in range(self.cfg.num_uavs):
            features.extend((self.uav_positions[i] / denom).tolist())
            if not self.active[i]:
                features.extend([0.0, 0.0])
            elif self.roles[i] == ACCESS:
                features.extend([1.0, 0.0])
            else:
                features.extend([0.0, 1.0])

        ue_map = np.zeros((self.cfg.grid_size, self.cfg.grid_size), dtype=np.float32)
        for pos in self.ue_positions:
            ue_map[tuple(pos)] += 1.0
        if ue_map.max() > 0:
            ue_map /= ue_map.max()

        bs_map = np.zeros_like(ue_map)
        bs_map[tuple(self.bs_position)] = 1.0

        features.extend(ue_map.ravel().tolist())
        features.extend(bs_map.ravel().tolist())
        return np.asarray(features, dtype=np.float32)

    def get_action_mask(self) -> np.ndarray:
        masks = super().get_action_mask()
        noop = self.encode_action(4, ACCESS)
        for i in range(self.cfg.num_uavs):
            if not self.active[i]:
                masks[i, :] = 0
                masks[i, noop] = 1
        return masks

    def _sync_active(self):
        self.active = self.status == ACTIVE

    def _one_step_to_station(self, pos: np.ndarray) -> np.ndarray:
        """Deterministic shortest Manhattan step, row first then column."""
        pos = pos.copy()
        dr = int(self.charging_station[0] - pos[0])
        dc = int(self.charging_station[1] - pos[1])
        if dr != 0:
            pos[0] += 1 if dr > 0 else -1
        elif dc != 0:
            pos[1] += 1 if dc > 0 else -1
        return pos


class StageBChargingObservableEnv(StageBChargingEnv):
    """V3.1: expose only a normalized ETA-to-active feature per UAV.

    This is the minimal extra coordination signal proposed in the system model:
    active UAV -> ETA 0;
    returning UAV -> Manhattan return distance + fixed charge duration;
    charging UAV -> remaining charge slots.

    Raw battery level remains hidden, so this still does not optimize proactive
    energy scheduling. It only lets the fleet anticipate a known teammate
    return and prepare for reintegration.
    """

    def get_state(self) -> np.ndarray:
        base = super().get_state()
        max_return = 2 * (self.cfg.grid_size - 1)
        norm = max(max_return + int(self.charging_cfg.charge_duration), 1)
        eta = []
        for i in range(self.cfg.num_uavs):
            if self.status[i] == ACTIVE:
                value = 0.0
            elif self.status[i] == RETURNING:
                dist = int(np.abs(self.uav_positions[i] - self.charging_station).sum())
                value = float(dist + self.charging_cfg.charge_duration) / norm
            else:
                value = float(self.charge_remaining[i]) / norm
            eta.append(value)
        return np.concatenate([base, np.asarray(eta, dtype=np.float32)])
