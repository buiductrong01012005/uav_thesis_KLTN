"""Four-arm CNN/QMIX: common encoder, hidden body and mixer initialization; arm-specific output sizes."""

from copy import deepcopy
import torch
from torch import nn
from src.rl.qmix import QMIXLearner
from src.rl.pooling import FixedPool5To3
from src.envs.observation import OBS_DIM, STATE_DIM
from src.envs.config import ARM_ACTIONS
from src.configuration import architecture_spec


class FourArmActor(nn.Module):
    def __init__(self, n_actions=10, architecture=None):
        super().__init__()
        architecture = deepcopy(architecture if architecture is not None else architecture_spec())
        if architecture["name"] != "fourarm_cnn_qmix_v1":
            raise ValueError("Unsupported saved architecture")
        channels, hidden = architecture["conv_channels"], architecture["hidden_dim"]
        if not all(isinstance(n, int) and n > 0 for n in [channels, hidden]):
            raise ValueError("Architecture widths must be positive integers")
        self.architecture = architecture
        self.encoder = nn.Sequential(
            nn.Conv2d(8, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            FixedPool5To3(),
        )
        self.head = nn.Sequential(
            nn.Linear(channels * 9 + 73, hidden), nn.ReLU(), nn.Linear(hidden, n_actions)
        )

    def forward(self, obs):
        if obs.shape[-1] != OBS_DIM:
            raise ValueError("Temporal fair actor requires173D observations")
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
        coords = (obs[:, 130:132] * 19 / 4).long().clamp(0, 4)
        own = torch.zeros((batch, 25), dtype=obs.dtype, device=obs.device)
        own.scatter_(1, coords[:, :1] * 5 + coords[:, 1:2], 1)
        history = obs[:, 148:173].reshape(batch, 1, 5, 5)
        encoded = self.encoder(
            torch.cat([sensor, *maps, own.reshape(batch, 1, 5, 5), history], dim=1)
        ).flatten(1)
        return self.head(torch.cat([encoded, obs[:, 75:148]], dim=1))


class FourArmQMIX(QMIXLearner):
    def __init__(self, *, n_actions=10, lr=3e-4, gamma=0.99, device="cpu", architecture=None):
        super().__init__(5, OBS_DIM, STATE_DIM, 10, lr=lr, gamma=gamma, device=device)
        # Fixed parent width preserves RNG consumption before shared encoder/mixer construction.
        self.n_actions = n_actions
        self.agent = FourArmActor(n_actions, architecture).to(self.device)
        self.target_agent = deepcopy(self.agent)
        self.optimizer = torch.optim.Adam(
            list(self.agent.parameters()) + list(self.mixer.parameters()), lr=lr
        )

    def train_batch(self, batch):
        loss = super().train_batch(batch)
        if not torch.isfinite(torch.tensor(loss)) or not all(
            torch.isfinite(p).all()
            for p in list(self.agent.parameters()) + list(self.mixer.parameters())
        ):
            raise FloatingPointError("Nonfinite temporal-fair update")
        return loss


def make_learner(arm, **kwargs):
    sizes = ARM_ACTIONS
    if arm not in sizes:
        raise ValueError("Unknown four-arm learner")
    return FourArmQMIX(n_actions=sizes[arm], **kwargs)
