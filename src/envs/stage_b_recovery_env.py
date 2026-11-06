"""Stage B: recovery environment fed by a completed Stage-A scout result.

Stage B intentionally contains only the communication-recovery skill:
movement + {Access, Relay}. The UE map supplied to this environment is the
*discovered* map from Stage A. Hidden/missed UEs are not exposed to Stage B.

This makes the information bottleneck explicit: Stage A determines which UEs
Stage B can possibly serve, while Stage B learns how to recover service for the
UEs it actually knows about.
"""

from dataclasses import dataclass
from typing import Sequence

from src.envs.v0_env import UAVRecoveryEnvV0, V0Config


@dataclass
class StageBConfig:
    discovered_ue_positions: tuple[tuple[int, int], ...]
    uav_start_positions: tuple[tuple[int, int], ...]
    total_true_ues: int
    grid_size: int = 5
    num_uavs: int = 3
    horizon: int = 30
    access_range: float = 1.5
    backhaul_range: float = 2.5
    surviving_bs: tuple[int, int] = (0, 4)


class StageBRecoveryEnv(UAVRecoveryEnvV0):
    """V0 recovery dynamics initialized from a Stage-A scout output.

    Training reward is the fraction of *discovered* UEs served because those
    are the only users Stage B is allowed to know. Evaluation additionally
    reports service as a fraction of all true UEs, exposing the ceiling imposed
    by missed discoveries in Stage A.
    """

    def __init__(self, config: StageBConfig):
        if len(config.discovered_ue_positions) == 0:
            raise ValueError("Stage B requires at least one discovered UE")
        if config.total_true_ues < len(config.discovered_ue_positions):
            raise ValueError("total_true_ues cannot be smaller than discovered UE count")

        self.stage_b_cfg = config
        super().__init__(
            V0Config(
                grid_size=config.grid_size,
                num_uavs=config.num_uavs,
                horizon=config.horizon,
                access_range=config.access_range,
                backhaul_range=config.backhaul_range,
                surviving_bs=config.surviving_bs,
                ue_positions=config.discovered_ue_positions,
                uav_start_positions=config.uav_start_positions,
            )
        )

    @property
    def discovered_ue_count(self) -> int:
        return len(self.ue_positions)

    @property
    def information_ceiling(self) -> float:
        return self.discovered_ue_count / self.stage_b_cfg.total_true_ues

    def step(self, actions: Sequence[int]):
        out = super().step(actions)
        served_count = out["info"]["served_count"]
        out["info"]["served_discovered_ratio"] = float(served_count / self.discovered_ue_count)
        out["info"]["served_total_ratio"] = float(served_count / self.stage_b_cfg.total_true_ues)
        out["info"]["information_ceiling"] = float(self.information_ceiling)
        # `reward` from V0 is already served_count / discovered_ue_count.
        return out
