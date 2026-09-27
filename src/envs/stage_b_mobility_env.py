"""Slow-UE-mobility robustness environment for Stage-B recovery.

This module is evaluation-only by design.  It keeps the trained Stage-B
movement+{Access,Relay} policy unchanged and introduces slow ground-user
motion after each service slot.

Two observation modes are supported:

``stale`` (default)
    The policy continues to see the UE map that was discovered at the start
    of Stage B.  Ground-truth UEs move underneath that fixed map.  This is the
    intended two-stage robustness test: how much does service degrade when the
    scouting result becomes stale?

``tracked``
    The policy sees the current ground-truth UE positions after every move.
    This is *not* the thesis system assumption.  It is only a diagnostic upper
    bound that separates stale-information loss from control/coordination loss.

UE mobility is deliberately simple: independently for every UE and every slot,
with probability ``ue_move_prob`` the UE moves to one uniformly selected valid
4-neighbour cell.  Multiple UEs may occupy the same cell after moving.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from src.envs.stage_b_random_charging_env import (
    RandomStageBChargingConfig,
    RandomizedStageBChargingEnv,
)
from src.envs.v0_env import ACCESS


@dataclass
class MobilityStageBChargingConfig(RandomStageBChargingConfig):
    ue_move_prob: float = 0.0
    observation_mode: str = "stale"  # {"stale", "tracked"}


class MobilityRandomizedStageBChargingEnv(RandomizedStageBChargingEnv):
    """Random dynamic-fleet Stage B plus slow UE mobility.

    Network service is always computed from ``actual_ue_positions``.  The map
    exposed in the state/observation is selected by ``observation_mode``.
    """

    def __init__(self, config: MobilityStageBChargingConfig):
        if not 0.0 <= float(config.ue_move_prob) <= 1.0:
            raise ValueError("ue_move_prob must be in [0, 1]")
        if config.observation_mode not in {"stale", "tracked"}:
            raise ValueError("observation_mode must be 'stale' or 'tracked'")
        self.mobility_cfg = config
        self.observed_ue_positions = np.empty((0, 2), dtype=np.int64)
        self.actual_ue_positions = np.empty((0, 2), dtype=np.int64)
        self.total_ue_moves = 0
        super().__init__(config)

    def _sample_episode(self):
        super()._sample_episode()
        # `self.ue_positions` is the discovered Stage-A map.  Freeze a copy for
        # stale observations, and maintain a separate ground-truth copy that
        # will move during Stage B.
        self.observed_ue_positions = self.ue_positions.copy()
        self.actual_ue_positions = self.ue_positions.copy()
        self.total_ue_moves = 0

    def reset(self):
        out = super().reset()
        # Parent reset already calls our overridden _sample_episode().
        out["obs"] = self.get_obs()
        out["state"] = self.get_state()
        out["episode_meta"].update(self.mobility_meta())
        return out

    def step(self, actions):
        # Service for this slot is evaluated at the current ground-truth UE
        # locations.  Then UEs move and the resulting positions belong to the
        # next decision state.
        out = super().step(actions)
        moved = self._move_ground_ues()
        self.total_ue_moves += moved

        # Parent constructed next obs before mobility.  Rebuild it after UE
        # motion so tracked mode sees the new positions.  In stale mode this is
        # intentionally unchanged because observed_ue_positions are frozen.
        out["obs"] = self.get_obs()
        out["state"] = self.get_state()
        out["info"].update(self.mobility_meta())
        out["info"]["ue_moves_this_slot"] = int(moved)
        out["info"]["total_ue_moves"] = int(self.total_ue_moves)
        return out

    def mobility_meta(self) -> dict:
        return {
            "ue_move_prob": float(self.mobility_cfg.ue_move_prob),
            "observation_mode": self.mobility_cfg.observation_mode,
        }

    def compute_served_users(self):
        """Serve the *actual moving* UEs, never the stale observation map."""
        valid_access, relay_chains = self.compute_backhaul()
        served = np.zeros(len(self.actual_ue_positions), dtype=bool)
        valid_access_ids = np.flatnonzero(valid_access)

        for ue_id, ue_pos in enumerate(self.actual_ue_positions):
            for access_id in valid_access_ids:
                if (
                    self._distance(ue_pos, self.uav_positions[access_id])
                    <= self.cfg.access_range
                ):
                    served[ue_id] = True
                    break
        return served, valid_access, relay_chains

    def get_state(self) -> np.ndarray:
        """Stage-B state with either stale or perfectly tracked UE map."""
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

        if self.mobility_cfg.observation_mode == "tracked":
            map_positions = self.actual_ue_positions
        else:
            map_positions = self.observed_ue_positions

        ue_map = np.zeros((self.cfg.grid_size, self.cfg.grid_size), dtype=np.float32)
        for pos in map_positions:
            ue_map[tuple(pos)] += 1.0
        if ue_map.max() > 0:
            ue_map /= ue_map.max()

        bs_map = np.zeros_like(ue_map)
        bs_map[tuple(self.bs_position)] = 1.0

        features.extend(ue_map.ravel().tolist())
        features.extend(bs_map.ravel().tolist())
        return np.asarray(features, dtype=np.float32)

    def _move_ground_ues(self) -> int:
        p = float(self.mobility_cfg.ue_move_prob)
        if p <= 0.0:
            return 0

        g = self.cfg.grid_size
        moved = 0
        for i in range(len(self.actual_ue_positions)):
            if float(self.rng.random()) >= p:
                continue
            r, c = map(int, self.actual_ue_positions[i])
            neighbours: list[tuple[int, int]] = []
            if r > 0:
                neighbours.append((r - 1, c))
            if r + 1 < g:
                neighbours.append((r + 1, c))
            if c > 0:
                neighbours.append((r, c - 1))
            if c + 1 < g:
                neighbours.append((r, c + 1))
            if not neighbours:
                continue
            j = int(self.rng.integers(0, len(neighbours)))
            self.actual_ue_positions[i] = np.asarray(neighbours[j], dtype=np.int64)
            moved += 1
        return moved
