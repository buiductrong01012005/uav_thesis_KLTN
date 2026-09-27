"""Randomized dynamic-fleet Stage-B environment for the final curriculum.

This environment is the C3 task used after static and spatial recovery:
- UE hotspot is resampled every episode;
- UAV initial positions may be resampled every episode;
- exactly one UAV is selected uniformly to become low-battery;
- its automatic return trigger time is sampled from a fixed interval;
- charging duration is sampled from a fixed interval;
- policy action remains movement + {Access, Relay};
- battery/return/charging are environment dynamics, not policy actions.

State/observation dimensionality remains identical to the simpler Stage-B
recovery environment, so one QMIX learner can be trained by curriculum across
C1 -> C2 -> C3 without architecture surgery.
"""

from dataclasses import dataclass
import numpy as np

from src.envs.stage_b_charging_env import StageBChargingConfig, StageBChargingEnv


@dataclass
class RandomStageBChargingConfig:
    uav_start_positions: tuple[tuple[int, int], ...]
    grid_size: int = 5
    num_uavs: int = 3
    horizon: int = 24
    num_ues: int = 8
    hotspot_sigma: float = 1.15
    access_range: float = 1.5
    backhaul_range: float = 2.5
    surviving_bs: tuple[int, int] = (0, 4)
    charging_station: tuple[int, int] = (4, 4)

    max_battery: float = 100.0
    base_drain_per_slot: float = 1.0
    return_threshold: float = 3.0

    # Failure dynamics are sampled once per episode.
    dropout_slot_min: int = 5
    dropout_slot_max: int = 9
    charge_duration_min: int = 3
    charge_duration_max: int = 5

    seed: int = 0
    randomize_uav_starts: bool = True


class RandomizedStageBChargingEnv(StageBChargingEnv):
    """C3 curriculum task with randomized spatial and fleet dynamics."""

    def __init__(self, config: RandomStageBChargingConfig):
        if config.num_ues > config.grid_size ** 2:
            raise ValueError("num_ues cannot exceed grid cells")
        if config.dropout_slot_min < 1 or config.dropout_slot_max < config.dropout_slot_min:
            raise ValueError("invalid dropout slot range")
        if config.charge_duration_min < 1 or config.charge_duration_max < config.charge_duration_min:
            raise ValueError("invalid charge-duration range")

        self.random_cfg = config
        self.rng = np.random.default_rng(config.seed)
        self.hotspot_center = np.zeros(2, dtype=np.int64)
        self.sampled_dropout_uav = 0
        self.sampled_dropout_slot = config.dropout_slot_min
        self.sampled_charge_duration = config.charge_duration_min

        # Placeholder UE set; every reset replaces it before parent reset.
        initial_ues = tuple(
            (i // config.grid_size, i % config.grid_size)
            for i in range(config.num_ues)
        )
        initial_battery = tuple([config.max_battery] * config.num_uavs)

        super().__init__(StageBChargingConfig(
            discovered_ue_positions=initial_ues,
            uav_start_positions=config.uav_start_positions,
            total_true_ues=config.num_ues,
            grid_size=config.grid_size,
            num_uavs=config.num_uavs,
            horizon=config.horizon,
            access_range=config.access_range,
            backhaul_range=config.backhaul_range,
            surviving_bs=config.surviving_bs,
            initial_battery=initial_battery,
            max_battery=config.max_battery,
            base_drain_per_slot=config.base_drain_per_slot,
            return_threshold=config.return_threshold,
            charging_station=config.charging_station,
            charge_duration=config.charge_duration_min,
        ))

    def seed(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self._sample_episode()
        out = super().reset()
        out["episode_meta"] = self.episode_meta()
        return out

    def step(self, actions):
        out = super().step(actions)
        out["info"].update(self.episode_meta())
        return out

    def episode_meta(self) -> dict:
        return {
            "hotspot_center": self.hotspot_center.astype(int).tolist(),
            "dropout_uav": int(self.sampled_dropout_uav),
            "planned_dropout_slot": int(self.sampled_dropout_slot),
            "sampled_charge_duration": int(self.sampled_charge_duration),
        }

    def _sample_episode(self):
        cfg = self.random_cfg
        g = cfg.grid_size
        cells = np.asarray([(r, c) for r in range(g) for c in range(g)], dtype=np.int64)

        # Spatial randomization.
        self.hotspot_center = self.rng.integers(0, g, size=2, dtype=np.int64)
        dist2 = np.sum((cells - self.hotspot_center[None, :]) ** 2, axis=1)
        sigma2 = max(cfg.hotspot_sigma ** 2, 1e-6)
        weights = np.exp(-dist2 / (2.0 * sigma2)) + 1e-9
        weights /= weights.sum()
        idx = self.rng.choice(len(cells), size=cfg.num_ues, replace=False, p=weights)
        self.ue_positions = cells[idx].copy()

        if cfg.randomize_uav_starts:
            sidx = self.rng.choice(len(cells), size=cfg.num_uavs, replace=False)
            self.start_positions = cells[sidx].copy()
        else:
            self.start_positions = np.asarray(cfg.uav_start_positions, dtype=np.int64).copy()

        # Fleet-dynamics randomization. Only one UAV is made low-battery in an
        # episode so C3 tests rebalancing/rejoin rather than cascading failures.
        self.sampled_dropout_uav = int(self.rng.integers(0, cfg.num_uavs))
        self.sampled_dropout_slot = int(
            self.rng.integers(cfg.dropout_slot_min, cfg.dropout_slot_max + 1)
        )
        self.sampled_charge_duration = int(
            self.rng.integers(cfg.charge_duration_min, cfg.charge_duration_max + 1)
        )

        batteries = [float(cfg.max_battery)] * cfg.num_uavs
        # The parent triggers RETURNING after the active-slot drain when
        # battery <= threshold, therefore threshold + k*drain triggers at slot k.
        batteries[self.sampled_dropout_uav] = (
            float(cfg.return_threshold)
            + self.sampled_dropout_slot * float(cfg.base_drain_per_slot)
        )
        self.charging_cfg.initial_battery = tuple(batteries)
        self.charging_cfg.charge_duration = self.sampled_charge_duration
