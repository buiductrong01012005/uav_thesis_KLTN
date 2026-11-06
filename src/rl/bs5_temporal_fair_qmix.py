"""Same CNN/QMIX in both fairness modes; one extra causal history-map channel."""
from copy import deepcopy
import torch
from torch import nn
from src.rl.qmix import QMIXLearner
from src.rl.bs5_hybrid_qmix import FixedPool5To3
from src.envs.bs5_temporal_fair import OBS_DIM, STATE_DIM


class TemporalFairActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(8, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), FixedPool5To3())
        self.head = nn.Sequential(nn.Linear(32*9+73, 128), nn.ReLU(), nn.Linear(128, 10))

    def forward(self, obs):
        if obs.shape[-1] != OBS_DIM:
            raise ValueError("Temporal fair actor requires173D observations")
        batch = len(obs)
        sensor = obs[:, :75].reshape(batch, 3, 5, 5)
        fleet = obs[:, 75:130].reshape(batch, 5, 11)
        coords = (fleet[:, :, :2]*19/4).long().clamp(0, 4)
        index = coords[:, :, 0]*5+coords[:, :, 1]
        valid, active, relay = fleet[:, :, 10] > .5, fleet[:, :, 4] > .5, fleet[:, :, 2] > .5
        maps = []
        for flags in [valid & active & ~relay, valid & active & relay, valid & ~active]:
            flat = torch.zeros((batch, 25), dtype=obs.dtype, device=obs.device)
            flat.scatter_add_(1, index, flags.to(obs.dtype))
            maps.append(flat.clamp_max(1).reshape(batch, 1, 5, 5))
        coords = (obs[:, 130:132]*19/4).long().clamp(0, 4)
        own = torch.zeros((batch, 25), dtype=obs.dtype, device=obs.device)
        own.scatter_(1, coords[:, :1]*5+coords[:, 1:2], 1)
        history = obs[:, 148:173].reshape(batch, 1, 5, 5)
        encoded = self.encoder(torch.cat([sensor, *maps, own.reshape(batch, 1, 5, 5), history], dim=1)).flatten(1)
        return self.head(torch.cat([encoded, obs[:, 75:148]], dim=1))


class TemporalFairQMIX(QMIXLearner):
    def __init__(self, *, lr=3e-4, gamma=.99, device="cpu"):
        super().__init__(5, OBS_DIM, STATE_DIM, 10, lr=lr, gamma=gamma, device=device)
        self.agent = TemporalFairActor().to(self.device)
        self.target_agent = deepcopy(self.agent)
        self.optimizer = torch.optim.Adam(list(self.agent.parameters())+list(self.mixer.parameters()), lr=lr)

    def train_batch(self, batch):
        loss = super().train_batch(batch)
        if not torch.isfinite(torch.tensor(loss)) or not all(torch.isfinite(p).all() for p in list(self.agent.parameters())+list(self.mixer.parameters())):
            raise FloatingPointError("Nonfinite temporal-fair update")
        return loss


def make_learner(arm, **kwargs):
    if arm != "C-R":
        raise ValueError("Matched fairness experiment uses C-R")
    return TemporalFairQMIX(**kwargs)
