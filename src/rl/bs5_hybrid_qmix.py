"""Spatial-QMIX Hybrid head; A/R-only inputs remain identical to old148D actor."""

from copy import deepcopy
import torch
from torch import nn
from src.rl.bs5_service_v2_qmix import ServiceV2Actor, ServiceV2QMIX
from src.rl.qmix import QMIXLearner


class FixedPool5To3(nn.Module):
    """AdaptiveAvgPool2d(3,3) for a5x5 map, without CUDA atomic backward.

    The fixed linear map is mathematically identical to PyTorch's adaptive
    average bins. It has no learned parameters or persistent state-dict keys.
    CUDA matmul is deterministic under the runner's CUBLAS workspace setting.
    """

    def __init__(self):
        super().__init__()
        matrix = torch.zeros((9, 25), dtype=torch.float64)
        for row in range(3):
            for col in range(3):
                r0, r1 = row * 5 // 3, ((row + 1) * 5 + 2) // 3
                c0, c1 = col * 5 // 3, ((col + 1) * 5 + 2) // 3
                for r in range(r0, r1):
                    for c in range(c0, c1):
                        matrix[row * 3 + col, r * 5 + c] = 1 / ((r1 - r0) * (c1 - c0))
        self.register_buffer("matrix", matrix, persistent=False)

    def forward(self, x):
        if x.shape[-2:] != (5, 5):
            raise ValueError("Fixed pooling expects5x5 convolution maps")
        pooled = torch.matmul(x.flatten(-2), self.matrix.to(dtype=x.dtype).t())
        return pooled.reshape(*x.shape[:-2], 3, 3)


class HybridActor(ServiceV2Actor):
    def __init__(self):
        super().__init__()
        self.encoder[-1] = FixedPool5To3()
        self.head[-1] = nn.Linear(128, 15)

    def forward(self, obs):
        if obs.shape[-1] != 148:
            raise ValueError("Hybrid actor requires148D causal observation")
        batch = len(obs)
        sensor = obs[:, :75].reshape(batch, 3, 5, 5)
        fleet = obs[:, 75:130].reshape(batch, 5, 11)
        coords = (fleet[:, :, :2] * 19 / 4).long().clamp(0, 4)
        index = coords[:, :, 0] * 5 + coords[:, :, 1]
        valid, active = fleet[:, :, 10] > 0.5, fleet[:, :, 4] > 0.5
        hybrid = (fleet[:, :, 2] > 0.25) & (fleet[:, :, 2] < 0.75)
        relay = fleet[:, :, 2] > 0.75
        maps = []
        for flags in [
            valid & active & (~relay | hybrid),
            valid & active & (relay | hybrid),
            valid & ~active,
        ]:
            flat = torch.zeros((batch, 25), dtype=obs.dtype, device=obs.device)
            flat.scatter_add_(1, index, flags.to(obs.dtype))
            maps.append(flat.clamp_max(1).reshape(batch, 1, 5, 5))
        own_coords = (obs[:, 130:132] * 19 / 4).long().clamp(0, 4)
        own = torch.zeros((batch, 25), dtype=obs.dtype, device=obs.device)
        own.scatter_(1, own_coords[:, :1] * 5 + own_coords[:, 1:2], 1)
        encoded = self.encoder(
            torch.cat([sensor, *maps, own.reshape(batch, 1, 5, 5)], dim=1)
        ).flatten(1)
        return self.head(torch.cat([encoded, obs[:, 75:]], dim=1))


class HybridQMIX(QMIXLearner):
    def __init__(self, *, lr=3e-4, gamma=0.99, device="cpu"):
        super().__init__(5, 148, 66, 15, lr=lr, gamma=gamma, device=device)
        self.agent = HybridActor().to(self.device)
        self.target_agent = deepcopy(self.agent)
        self.optimizer = torch.optim.Adam(
            list(self.agent.parameters()) + list(self.mixer.parameters()), lr=lr
        )

    def train_batch(self, batch):
        loss = super().train_batch(batch)
        params = list(self.agent.parameters()) + list(self.mixer.parameters())
        if not torch.isfinite(torch.tensor(loss)) or not all(
            torch.isfinite(p).all() for p in params
        ):
            raise FloatingPointError("Nonfinite Hybrid learner update")
        return loss


def make_learner(arm, **kwargs):
    if arm == "C-H":
        return HybridQMIX(**kwargs)
    learner = ServiceV2QMIX(**kwargs)
    learner.agent.encoder[-1] = FixedPool5To3().to(learner.device)
    learner.target_agent.encoder[-1] = FixedPool5To3().to(learner.device)
    return learner
