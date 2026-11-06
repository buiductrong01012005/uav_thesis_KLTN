"""V1 environment: V0 Access/Relay coordination + active Scout role.

V1 adds exactly one source of partial information:
- the failed region (entire 5x5 grid in the first experiment) is known;
- true UE locations are hidden at reset;
- a UAV in Scout role reveals exact UE counts in a small footprint around its post-move cell;
- scouting results are shared instantly with the whole fleet;
- only discovered UEs can be served by Access UAVs.

Everything else remains deliberately simple: no battery, no charging, no
continuous trajectory, no power/bandwidth optimization, and at most one relay hop.
"""

from dataclasses import dataclass
from typing import Iterable, Sequence
import numpy as np

SCOUT = 0
ACCESS = 1
RELAY = 2

ROLE_NAMES = {
    SCOUT: "Scout",
    ACCESS: "Access",
    RELAY: "Relay",
}

MOVE_NAMES = ["North", "South", "East", "West", "Stay"]
MOVE_DELTAS = np.asarray(
    [
        [-1, 0],
        [1, 0],
        [0, 1],
        [0, -1],
        [0, 0],
    ],
    dtype=np.int64,
)


@dataclass
class V1Config:
    grid_size: int = 5
    num_uavs: int = 3
    horizon: int = 50
    access_range: float = 1.5
    backhaul_range: float = 2.5
    scout_radius: float = 1.0

    # Discovery shaping is intentionally small relative to service reward.
    discovery_cell_weight: float = 0.10
    discovery_ue_weight: float = 0.20

    surviving_bs: tuple[int, int] = (0, 4)

    # Same fixed hidden map as V0 for the first overfit experiment.
    ue_positions: tuple[tuple[int, int], ...] = (
        (3, 0),
        (3, 1),
        (3, 2),
        (3, 3),
        (4, 0),
        (4, 1),
        (4, 2),
        (4, 3),
    )

    uav_start_positions: tuple[tuple[int, int], ...] = (
        (2, 0),
        (2, 2),
        (1, 3),
    )


class UAVRecoveryEnvV1:
    """Minimal V1 Dec-POMDP-style simulator.

    Local action encoding:
        action = move_index * 3 + role

    5 movements x 3 roles = 15 actions per UAV.
    """

    def __init__(self, config: V1Config | None = None):
        self.cfg = config or V1Config()
        if self.cfg.num_uavs != 3:
            raise ValueError("The first V1 experiment is intentionally fixed at 3 UAVs.")

        self.ue_positions = np.asarray(self.cfg.ue_positions, dtype=np.int64)
        self.bs_position = np.asarray(self.cfg.surviving_bs, dtype=np.int64)
        self.start_positions = np.asarray(self.cfg.uav_start_positions, dtype=np.int64)

        self.true_ue_map = np.zeros((self.cfg.grid_size, self.cfg.grid_size), dtype=np.int64)
        for p in self.ue_positions:
            self.true_ue_map[tuple(p)] += 1

        self.t = 0
        self.uav_positions = self.start_positions.copy()
        self.roles = np.full(self.cfg.num_uavs, SCOUT, dtype=np.int64)
        self.known_mask = np.zeros_like(self.true_ue_map, dtype=bool)
        self.discovered_ue_map = np.zeros_like(self.true_ue_map, dtype=np.int64)

    def reset(self):
        self.t = 0
        self.uav_positions = self.start_positions.copy()
        self.roles = np.full(self.cfg.num_uavs, SCOUT, dtype=np.int64)
        self.known_mask.fill(False)
        self.discovered_ue_map.fill(0)
        return {
            "obs": self.get_obs(),
            "state": self.get_state(),
            "action_mask": self.get_action_mask(),
        }

    def step(self, actions: Sequence[int]):
        if len(actions) != self.cfg.num_uavs:
            raise ValueError(f"Expected {self.cfg.num_uavs} actions.")

        new_positions = self.uav_positions.copy()
        new_roles = self.roles.copy()
        for i, action in enumerate(actions):
            move_idx, role = self.decode_action(int(action))
            candidate = self.uav_positions[i] + MOVE_DELTAS[move_idx]
            if self._inside_grid(candidate):
                new_positions[i] = candidate
            new_roles[i] = role

        self.uav_positions = new_positions
        self.roles = new_roles

        new_cells = 0
        new_ues = 0
        for i in range(self.cfg.num_uavs):
            if self.roles[i] != SCOUT:
                continue
            for cell in self._scout_footprint(self.uav_positions[i]):
                if not self.known_mask[cell]:
                    self.known_mask[cell] = True
                    count = int(self.true_ue_map[cell])
                    self.discovered_ue_map[cell] = count
                    new_cells += 1
                    new_ues += count

        served_mask, valid_access, relay_chains = self.compute_served_users()
        served_count = int(served_mask.sum())
        served_ratio = served_count / len(self.ue_positions)
        discovery_ratio = float(self.known_mask.mean())
        discovered_ue_count = int(self.discovered_ue_map.sum())
        discovered_ue_ratio = discovered_ue_count / len(self.ue_positions)

        discovery_reward = self.cfg.discovery_cell_weight * (
            new_cells / self.known_mask.size
        ) + self.cfg.discovery_ue_weight * (new_ues / len(self.ue_positions))
        reward = float(served_ratio + discovery_reward)

        self.t += 1
        terminated = False
        truncated = self.t >= self.cfg.horizon

        info = {
            "t": self.t,
            "served_count": served_count,
            "served_ratio": float(served_ratio),
            "new_cells": int(new_cells),
            "new_ues": int(new_ues),
            "discovery_ratio": discovery_ratio,
            "discovered_ue_count": discovered_ue_count,
            "discovered_ue_ratio": float(discovered_ue_ratio),
            "valid_access_count": int(valid_access.sum()),
            "relay_chains": relay_chains,
            "num_scout": int((self.roles == SCOUT).sum()),
            "num_access": int((self.roles == ACCESS).sum()),
            "num_relay": int((self.roles == RELAY).sum()),
            "service_reward": float(served_ratio),
            "discovery_reward": float(discovery_reward),
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
        access_ids = np.flatnonzero(self.roles == ACCESS)
        relay_ids = np.flatnonzero(self.roles == RELAY)

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

    def compute_served_users(self):
        """Only UEs in cells already revealed by Scout are service candidates."""
        valid_access, relay_chains = self.compute_backhaul()
        served = np.zeros(len(self.ue_positions), dtype=bool)
        valid_access_ids = np.flatnonzero(valid_access)

        for ue_id, ue_pos in enumerate(self.ue_positions):
            cell = tuple(ue_pos)
            if not self.known_mask[cell]:
                continue
            for access_id in valid_access_ids:
                if self._distance(ue_pos, self.uav_positions[access_id]) <= self.cfg.access_range:
                    served[ue_id] = True
                    break
        return served, valid_access, relay_chains

    def get_state(self) -> np.ndarray:
        """Training state uses fleet-known information only; hidden UE truth is excluded."""
        features: list[float] = []
        denom = max(self.cfg.grid_size - 1, 1)

        for i in range(self.cfg.num_uavs):
            features.extend((self.uav_positions[i] / denom).tolist())
            role_onehot = [0.0, 0.0, 0.0]
            role_onehot[int(self.roles[i])] = 1.0
            features.extend(role_onehot)

        # Normalize discovered counts by total UE count; unknown cells remain 0
        # but are distinguishable via known_mask.
        discovered = self.discovered_ue_map.astype(np.float32) / max(len(self.ue_positions), 1)
        known = self.known_mask.astype(np.float32)
        bs_map = np.zeros_like(discovered, dtype=np.float32)
        bs_map[tuple(self.bs_position)] = 1.0

        features.extend(discovered.ravel().tolist())
        features.extend(known.ravel().tolist())
        features.extend(bs_map.ravel().tolist())
        return np.asarray(features, dtype=np.float32)

    def get_obs(self) -> np.ndarray:
        state = self.get_state()
        obs = []
        for i in range(self.cfg.num_uavs):
            agent_id = np.zeros(self.cfg.num_uavs, dtype=np.float32)
            agent_id[i] = 1.0
            obs.append(np.concatenate([state, agent_id]))
        return np.stack(obs, axis=0)

    def get_action_mask(self) -> np.ndarray:
        masks = np.ones((self.cfg.num_uavs, 15), dtype=np.int8)
        for i, pos in enumerate(self.uav_positions):
            for action in range(15):
                move_idx, _ = self.decode_action(action)
                candidate = pos + MOVE_DELTAS[move_idx]
                if not self._inside_grid(candidate):
                    masks[i, action] = 0
        return masks

    @staticmethod
    def decode_action(action: int) -> tuple[int, int]:
        if not 0 <= action < 15:
            raise ValueError("V1 action must be in [0, 14].")
        return action // 3, action % 3

    @staticmethod
    def encode_action(move_idx: int, role: int) -> int:
        if move_idx not in range(5):
            raise ValueError("move_idx must be in [0, 4].")
        if role not in (SCOUT, ACCESS, RELAY):
            raise ValueError("Invalid V1 role.")
        return move_idx * 3 + role

    def _scout_footprint(self, center: np.ndarray) -> list[tuple[int, int]]:
        """Cells revealed by a Scout at ``center``.

        V1 uses perfect sensing inside a small Euclidean footprint. With the
        default radius 1.0 this is the current cell plus its four cardinal
        neighbours (clipped at the grid boundary).
        """
        cells = []
        for r in range(self.cfg.grid_size):
            for c in range(self.cfg.grid_size):
                p = np.asarray([r, c], dtype=np.int64)
                if self._distance(center, p) <= self.cfg.scout_radius:
                    cells.append((r, c))
        return cells

    def _inside_grid(self, pos: np.ndarray) -> bool:
        return bool(np.all(pos >= 0) and np.all(pos < self.cfg.grid_size))

    @staticmethod
    def _distance(a: Iterable[int], b: Iterable[int]) -> float:
        return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))
