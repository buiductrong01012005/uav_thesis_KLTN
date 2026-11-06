"""Spatial CNN + vector actor for causal v2 observations, standard QMIX learner."""
from copy import deepcopy

import torch
from torch import nn

from src.rl.qmix import QMIXLearner


class ServiceV2Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(7, 32, 3, padding=1), nn.ReLU(),
                                     nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
                                     nn.AdaptiveAvgPool2d((3, 3)))
        self.head = nn.Sequential(nn.Linear(32*9 + 73, 128), nn.ReLU(), nn.Linear(128, 10))

    def forward(self, obs):
        if obs.shape[-1] != 148:
            raise ValueError('ServiceV2Actor requires 148-D causal observation')
        batch = len(obs)
        sensor = obs[:, :75].reshape(batch, 3, 5, 5)
        fleet = obs[:, 75:130].reshape(batch, 5, 11)
        coords = (fleet[:, :, :2] * 19 / 4).long().clamp(0, 4)
        index = coords[:, :, 0] * 5 + coords[:, :, 1]
        valid = fleet[:, :, 10] > .5
        active = fleet[:, :, 4] > .5
        relay = fleet[:, :, 2] > .5
        maps = []
        for flags in [valid & active & ~relay, valid & active & relay, valid & ~active]:
            flat = torch.zeros((batch, 25), dtype=obs.dtype, device=obs.device)
            flat.scatter_add_(1, index, flags.to(obs.dtype))
            maps.append(flat.clamp_max(1).reshape(batch, 1, 5, 5))
        self_coords = (obs[:, 130:132] * 19 / 4).long().clamp(0, 4)
        self_map = torch.zeros((batch, 25), dtype=obs.dtype, device=obs.device)
        self_map.scatter_(1, (self_coords[:, :1]*5 + self_coords[:, 1:2]), 1)
        encoded = self.encoder(torch.cat([sensor, *maps, self_map.reshape(batch, 1, 5, 5)], dim=1)).flatten(1)
        return self.head(torch.cat([encoded, obs[:, 75:]], dim=1))


class ServiceV2QMIX(QMIXLearner):
    def __init__(self, *, lr=3e-4, gamma=.99, device='cpu'):
        super().__init__(5, 148, 66, 10, lr=lr, gamma=gamma, device=device)
        self.agent = ServiceV2Actor().to(self.device)
        self.target_agent = deepcopy(self.agent)
        self.optimizer = torch.optim.Adam(list(self.agent.parameters()) + list(self.mixer.parameters()), lr=lr)

    def train_batch(self, batch):
        loss = super().train_batch(batch)
        params = list(self.agent.parameters()) + list(self.mixer.parameters())
        if not torch.isfinite(torch.tensor(loss)) or not all(torch.isfinite(p).all() for p in params):
            raise FloatingPointError('Nonfinite v2 learner update')
        return loss
