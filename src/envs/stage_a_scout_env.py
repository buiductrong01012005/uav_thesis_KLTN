"""Stage A: Scout-only multi-UAV exploration environment.

This stage deliberately separates information acquisition from communication
recovery. All UAVs are Scouts and choose movement only. The hidden UE map is
never exposed to the policy or centralized training state.

The primary learning objective is fast spatial coverage of the failed region.
UE-discovery metrics are logged for evaluation, but the default training reward
uses newly revealed cells only, avoiding leakage of hidden UE locations through
reward on the fixed-map sanity task.
"""

from dataclasses import dataclass
from typing import Iterable, Sequence
import numpy as np

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
class StageAScoutConfig:
    grid_size: int = 5
    num_uavs: int = 3
    scout_budget: int = 8
    scout_radius: float = 1.0

    # Fixed hidden map for the first learnability/overfit experiment.
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


class StageAScoutEnv:
    """Scout-only Stage A.

    Each UAV has five actions: N/S/E/W/Stay.
    After simultaneous movement, every UAV scans a perfect radius-1 footprint.

    Reward:
        new_cells / total_cells

    This is a dense potential-like coverage signal and intentionally does not
    reward hidden UE counts. Thus the learner is encouraged to map the region,
    not memorize the fixed UE hotspot through the reward channel.
    """

    def __init__(self, config: StageAScoutConfig | None = None):
        self.cfg = config or StageAScoutConfig()
        if self.cfg.num_uavs != len(self.cfg.uav_start_positions):
            raise ValueError("num_uavs must match uav_start_positions length")

        self.ue_positions = np.asarray(self.cfg.ue_positions, dtype=np.int64)
        self.start_positions = np.asarray(self.cfg.uav_start_positions, dtype=np.int64)

        self.true_ue_map = np.zeros((self.cfg.grid_size, self.cfg.grid_size), dtype=np.int64)
        for p in self.ue_positions:
            self.true_ue_map[tuple(p)] += 1

        self.t = 0
        self.uav_positions = self.start_positions.copy()
        self.known_mask = np.zeros_like(self.true_ue_map, dtype=bool)
        self.discovered_ue_map = np.zeros_like(self.true_ue_map, dtype=np.int64)

    def reset(self):
        self.t = 0
        self.uav_positions = self.start_positions.copy()
        self.known_mask.fill(False)
        self.discovered_ue_map.fill(0)
        return self._output()

    def step(self, actions: Sequence[int]):
        if len(actions) != self.cfg.num_uavs:
            raise ValueError(f"Expected {self.cfg.num_uavs} actions")

        # Simultaneous movement.
        new_positions = self.uav_positions.copy()
        for i, action in enumerate(actions):
            action = int(action)
            if not 0 <= action < 5:
                raise ValueError("Stage A action must be in [0, 4]")
            candidate = self.uav_positions[i] + MOVE_DELTAS[action]
            if self._inside_grid(candidate):
                new_positions[i] = candidate
        self.uav_positions = new_positions

        new_cells = 0
        new_ues = 0
        for i in range(self.cfg.num_uavs):
            for cell in self._scout_footprint(self.uav_positions[i]):
                if not self.known_mask[cell]:
                    self.known_mask[cell] = True
                    count = int(self.true_ue_map[cell])
                    self.discovered_ue_map[cell] = count
                    new_cells += 1
                    new_ues += count

        reward = float(new_cells / self.known_mask.size)
        self.t += 1
        terminated = bool(self.known_mask.all())
        truncated = self.t >= self.cfg.scout_budget and not terminated

        info = {
            "t": self.t,
            "new_cells": int(new_cells),
            "new_ues": int(new_ues),
            "discovery_ratio": float(self.known_mask.mean()),
            "discovered_ue_count": int(self.discovered_ue_map.sum()),
            "discovered_ue_ratio": float(self.discovered_ue_map.sum() / max(len(self.ue_positions), 1)),
        }
        out = self._output()
        out.update({
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": info,
        })
        return out

    def get_state(self) -> np.ndarray:
        """Fleet-known centralized state; hidden UE truth is excluded."""
        denom = max(self.cfg.grid_size - 1, 1)
        features: list[float] = []
        for i in range(self.cfg.num_uavs):
            features.extend((self.uav_positions[i] / denom).tolist())

        # The discovered demand map is available because it has been sensed.
        discovered = self.discovered_ue_map.astype(np.float32) / max(len(self.ue_positions), 1)
        known = self.known_mask.astype(np.float32)
        features.extend(discovered.ravel().tolist())
        features.extend(known.ravel().tolist())
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
        masks = np.ones((self.cfg.num_uavs, 5), dtype=np.int8)
        for i, pos in enumerate(self.uav_positions):
            for action in range(5):
                candidate = pos + MOVE_DELTAS[action]
                if not self._inside_grid(candidate):
                    masks[i, action] = 0
        return masks

    def _output(self):
        return {
            "obs": self.get_obs(),
            "state": self.get_state(),
            "action_mask": self.get_action_mask(),
        }

    def _scout_footprint(self, center: np.ndarray) -> list[tuple[int, int]]:
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
