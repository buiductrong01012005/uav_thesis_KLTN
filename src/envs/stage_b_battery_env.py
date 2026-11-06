"""Stage-B recovery with a minimal battery-driven UAV unavailability event.

V2 deliberately isolates *rebalancing* from energy-management planning:
- policy still controls only movement + {Access, Relay};
- battery drains internally in the environment;
- a UAV whose battery reaches the threshold becomes unavailable for the rest
  of the episode;
- the policy is informed that a teammate is unavailable via the existing
  role slots in the shared state: inactive UAV role one-hot = [0, 0].

Crucially, state/observation dimensionality is unchanged from StageBRecoveryEnv,
so the learned V0/Stage-B recovery skill can be transferred directly.
A later V3 can expose battery level and add return/charge/rejoin explicitly.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.envs.stage_b_recovery_env import StageBConfig, StageBRecoveryEnv
from src.envs.v0_env import ACCESS, RELAY


@dataclass
class StageBBatteryConfig(StageBConfig):
    # One deterministic low-battery UAV is enough for the V2 learnability test.
    # Default UAV 1 is intentionally the first UAV to become unavailable.
    initial_battery: tuple[float, ...] = (100.0, 7.0, 100.0)
    base_drain_per_slot: float = 1.0
    return_threshold: float = 0.0


class StageBBatteryEnv(StageBRecoveryEnv):
    """Stage-B environment with permanent within-episode battery dropout.

    This version does *not* ask the policy to anticipate the failure. Battery
    level stays internal; once dropout occurs, all agents observe the changed
    fleet availability and must reconfigure online. This keeps V2 focused on
    one question only: can the fleet rebalance after losing a UAV?
    """

    def __init__(self, config: StageBBatteryConfig):
        if len(config.initial_battery) != config.num_uavs:
            raise ValueError("initial_battery must provide one value per UAV")
        self.battery_cfg = config
        self.battery = np.asarray(config.initial_battery, dtype=np.float32)
        self.active = np.ones(config.num_uavs, dtype=bool)
        self.dropout_time = np.full(config.num_uavs, -1, dtype=np.int64)
        super().__init__(config)

    def reset(self):
        self.battery = np.asarray(self.battery_cfg.initial_battery, dtype=np.float32).copy()
        self.active = np.ones(self.cfg.num_uavs, dtype=bool)
        self.dropout_time = np.full(self.cfg.num_uavs, -1, dtype=np.int64)
        return super().reset()

    def step(self, actions: Sequence[int]):
        # Inactive UAV actions are ignored. Replace them with a valid no-op so
        # the inherited transition can be reused safely.
        sanitized = list(actions)
        noop = self.encode_action(4, ACCESS)  # Stay/Access; ignored if inactive.
        for i in range(self.cfg.num_uavs):
            if not self.active[i]:
                sanitized[i] = noop

        # Apply active UAV moves/roles manually so an inactive UAV truly stays
        # frozen and does not accidentally change role/position.
        if len(sanitized) != self.cfg.num_uavs:
            raise ValueError(f"Expected {self.cfg.num_uavs} actions")

        new_positions = self.uav_positions.copy()
        new_roles = self.roles.copy()
        for i, action in enumerate(sanitized):
            if not self.active[i]:
                continue
            move_idx, role = self.decode_action(int(action))
            candidate = self.uav_positions[i] + self._move_delta(move_idx)
            if self._inside_grid(candidate):
                new_positions[i] = candidate
            new_roles[i] = role

        self.uav_positions = new_positions
        self.roles = new_roles

        # Service for this slot is computed using the fleet that was active at
        # the beginning of the slot. Battery dropout affects the next decision.
        served_mask, valid_access, relay_chains = self.compute_served_users()
        served_count = int(served_mask.sum())
        reward = float(served_count / self.discovered_ue_count)

        self.t += 1

        # Internal battery dynamics. V2 uses a constant drain so the experiment
        # isolates rebalancing instead of energy-aware action planning.
        for i in range(self.cfg.num_uavs):
            if not self.active[i]:
                continue
            self.battery[i] -= float(self.battery_cfg.base_drain_per_slot)
            if self.battery[i] <= float(self.battery_cfg.return_threshold):
                self.active[i] = False
                self.dropout_time[i] = self.t

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
            "battery": self.battery.astype(float).tolist(),
            "dropout_time": self.dropout_time.astype(int).tolist(),
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
        """Same dimensionality as Stage B/V0, with [0,0] role for inactive UAVs."""
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

    @staticmethod
    def _move_delta(move_idx: int) -> np.ndarray:
        # Local import avoided only to keep this subclass dependency explicit.
        from src.envs.v0_env import MOVE_DELTAS

        return MOVE_DELTAS[move_idx]
