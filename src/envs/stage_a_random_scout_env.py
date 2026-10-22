"""Randomized Stage-A Scout environment for generalization experiments.

This module keeps the same observation/action dimensions as ``StageAScoutEnv``
so fixed-map QMIX checkpoints can be evaluated zero-shot.

Randomization is deliberately staged:
- UE hotspot location is randomized every episode by default.
- UAV initial positions can optionally be randomized.

The hidden UE map remains hidden from the learner and the reward remains
coverage-only (newly revealed cells / total cells). Therefore randomizing UE
locations tests whether a learned exploration strategy generalizes rather than
memorizes the fixed hotspot.
"""

from dataclasses import dataclass
import numpy as np

from src.envs.stage_a_scout_env import StageAScoutConfig, StageAScoutEnv


@dataclass
class RandomStageAConfig:
    grid_size: int = 5
    num_uavs: int = 3
    scout_budget: int = 8
    scout_radius: float = 1.0
    num_ues: int = 8
    hotspot_sigma: float = 1.15
    randomize_uav_starts: bool = False
    seed: int = 0

    # Used when randomize_uav_starts=False. Matches the fixed-map V1 setup.
    fixed_uav_start_positions: tuple[tuple[int, int], ...] = (
        (2, 0),
        (2, 2),
        (1, 3),
    )


class RandomizedStageAScoutEnv(StageAScoutEnv):
    """Stage-A exploration with a new hidden hotspot every episode.

    UE locations are sampled without replacement from a Gaussian-like spatial
    distribution centered at a random grid cell. This creates a compact hotspot
    while guaranteeing exactly ``num_ues`` UEs in distinct cells.

    ``reset()`` resamples the hidden episode before clearing the Scout state.
    The policy never receives the hotspot center or unsensed UE locations.
    """

    def __init__(self, config: RandomStageAConfig | None = None):
        self.random_cfg = config or RandomStageAConfig()
        if self.random_cfg.num_ues > self.random_cfg.grid_size**2:
            raise ValueError("num_ues cannot exceed number of grid cells")
        if self.random_cfg.num_uavs != len(self.random_cfg.fixed_uav_start_positions):
            raise ValueError("num_uavs must match fixed_uav_start_positions length")

        self.rng = np.random.default_rng(self.random_cfg.seed)
        self.hotspot_center = np.zeros(2, dtype=np.int64)

        # Create an initial valid base environment; reset() will immediately
        # resample the hidden UE map before training/evaluation starts.
        init_ues = tuple(
            (i // self.random_cfg.grid_size, i % self.random_cfg.grid_size)
            for i in range(self.random_cfg.num_ues)
        )
        base_cfg = StageAScoutConfig(
            grid_size=self.random_cfg.grid_size,
            num_uavs=self.random_cfg.num_uavs,
            scout_budget=self.random_cfg.scout_budget,
            scout_radius=self.random_cfg.scout_radius,
            ue_positions=init_ues,
            uav_start_positions=self.random_cfg.fixed_uav_start_positions,
        )
        super().__init__(base_cfg)

    def seed(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self._sample_episode()
        return super().reset()

    def _sample_episode(self):
        g = self.random_cfg.grid_size
        self.hotspot_center = self.rng.integers(0, g, size=2, dtype=np.int64)

        cells = np.asarray([(r, c) for r in range(g) for c in range(g)], dtype=np.int64)
        dist2 = np.sum((cells - self.hotspot_center[None, :]) ** 2, axis=1)
        sigma2 = max(self.random_cfg.hotspot_sigma**2, 1e-6)
        weights = np.exp(-dist2 / (2.0 * sigma2)) + 1e-9
        weights /= weights.sum()

        chosen = self.rng.choice(
            len(cells),
            size=self.random_cfg.num_ues,
            replace=False,
            p=weights,
        )
        self.ue_positions = cells[chosen].copy()

        self.true_ue_map = np.zeros((g, g), dtype=np.int64)
        for p in self.ue_positions:
            self.true_ue_map[tuple(p)] += 1

        if self.random_cfg.randomize_uav_starts:
            starts_idx = self.rng.choice(len(cells), size=self.random_cfg.num_uavs, replace=False)
            self.start_positions = cells[starts_idx].copy()
        else:
            self.start_positions = np.asarray(
                self.random_cfg.fixed_uav_start_positions, dtype=np.int64
            ).copy()

        # StageAScoutEnv.reset() uses these arrays and clears them in-place.
        self.known_mask = np.zeros_like(self.true_ue_map, dtype=bool)
        self.discovered_ue_map = np.zeros_like(self.true_ue_map, dtype=np.int64)


class RandomizedStageACoverageOnlyEnv(RandomizedStageAScoutEnv):
    """Randomized Scout environment with a minimal exploration observation.

    The policy sees only:
    - all UAV positions;
    - the binary known/unknown mask.

    The discovered UE map is still maintained internally and returned through
    episode metrics/handoff, but it is *not* fed back into the Scout policy.
    This prevents the fixed-map learner from using demand values as accidental
    route landmarks when Stage A's objective is pure spatial exploration.
    """

    def get_state(self) -> np.ndarray:
        denom = max(self.cfg.grid_size - 1, 1)
        features: list[float] = []
        for i in range(self.cfg.num_uavs):
            features.extend((self.uav_positions[i] / denom).tolist())
        features.extend(self.known_mask.astype(np.float32).ravel().tolist())
        return np.asarray(features, dtype=np.float32)
