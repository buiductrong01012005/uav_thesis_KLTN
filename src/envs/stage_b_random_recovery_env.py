"""Randomized Stage-B Access/Relay recovery environment.

The action/observation dimensions are identical to StageBRecoveryEnv. Only the
UE hotspot is resampled at every reset, allowing the same QMIX architecture to
learn recovery across different demand geometries.
"""

from dataclasses import dataclass
import numpy as np

from src.envs.stage_b_recovery_env import StageBConfig, StageBRecoveryEnv


@dataclass
class RandomStageBConfig:
    uav_start_positions: tuple[tuple[int, int], ...]
    grid_size: int = 5
    num_uavs: int = 3
    horizon: int = 20
    num_ues: int = 8
    hotspot_sigma: float = 1.15
    access_range: float = 1.5
    backhaul_range: float = 2.5
    surviving_bs: tuple[int, int] = (0, 4)
    seed: int = 0
    randomize_uav_starts: bool = False


class RandomizedStageBRecoveryEnv(StageBRecoveryEnv):
    def __init__(self, config: RandomStageBConfig):
        self.random_cfg = config
        if config.num_ues > config.grid_size ** 2:
            raise ValueError("num_ues cannot exceed grid cells")
        self.rng = np.random.default_rng(config.seed)
        self.hotspot_center = np.zeros(2, dtype=np.int64)
        initial = tuple((i // config.grid_size, i % config.grid_size) for i in range(config.num_ues))
        super().__init__(StageBConfig(
            discovered_ue_positions=initial,
            uav_start_positions=config.uav_start_positions,
            total_true_ues=config.num_ues,
            grid_size=config.grid_size,
            num_uavs=config.num_uavs,
            horizon=config.horizon,
            access_range=config.access_range,
            backhaul_range=config.backhaul_range,
            surviving_bs=config.surviving_bs,
        ))

    def reset(self):
        self._sample_episode()
        return super().reset()

    def seed(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def _sample_episode(self):
        g = self.random_cfg.grid_size
        cells = np.asarray([(r,c) for r in range(g) for c in range(g)], dtype=np.int64)
        self.hotspot_center = self.rng.integers(0, g, size=2, dtype=np.int64)
        dist2 = np.sum((cells - self.hotspot_center[None,:])**2, axis=1)
        sigma2 = max(self.random_cfg.hotspot_sigma**2, 1e-6)
        w = np.exp(-dist2/(2*sigma2)) + 1e-9
        w /= w.sum()
        idx = self.rng.choice(len(cells), size=self.random_cfg.num_ues, replace=False, p=w)
        self.ue_positions = cells[idx].copy()
        if self.random_cfg.randomize_uav_starts:
            sidx = self.rng.choice(len(cells), size=self.random_cfg.num_uavs, replace=False)
            self.start_positions = cells[sidx].copy()
        else:
            self.start_positions = np.asarray(self.random_cfg.uav_start_positions, dtype=np.int64).copy()
