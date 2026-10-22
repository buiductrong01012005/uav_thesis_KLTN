"""Spatial QMIX with an explicit failed-BS context, separately versioned.

Uses the existing spatial feature encoder without changing its legacy parser.
Old checkpoints must not be silently loaded into the new observation layout.
"""

from __future__ import annotations

import torch
from torch import nn

from src.rl.qmix import QMixer, QMIXLearner
from src.rl.v5_relay_request_spatial_qmix import V5RelayRequestSpatialAgentQNetwork


class EpisodeFailureSpatialAgent(nn.Module):
    def __init__(self, n_agents, grid_size, coarse_size, n_actions, n_bs=6, hidden_dim=64):
        super().__init__()
        self.n_agents, self.n_bs = int(n_agents), int(n_bs)
        self.encoder = V5RelayRequestSpatialAgentQNetwork(
            n_agents, grid_size, coarse_size, n_actions, hidden_dim
        )
        self.legacy_state_dim = self.encoder.state_dim
        feature_dim = self.encoder.head[0].in_features
        self.encoder.head = nn.Identity()
        self.state_dim = self.legacy_state_dim + self.n_bs
        self.expected_obs_dim = self.state_dim + self.n_agents
        self.head = nn.Sequential(
            nn.Linear(feature_dim + self.n_bs, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, obs):
        if obs.ndim != 2 or obs.shape[-1] != self.expected_obs_dim:
            raise ValueError(f"Expected [B,{self.expected_obs_dim}], got {tuple(obs.shape)}")
        legacy = torch.cat((obs[:, : self.legacy_state_dim], obs[:, self.state_dim :]), dim=1)
        context = obs[:, self.legacy_state_dim : self.state_dim]
        return self.head(torch.cat((self.encoder(legacy), context), dim=1))


class EpisodeFailureSpatialQMIX(QMIXLearner):
    def __init__(self, env, lr=3e-4, gamma=0.99, device="cpu"):
        self.n_agents, self.n_actions = env.cfg.num_uavs, env.n_actions
        self.device, self.gamma = torch.device(device), float(gamma)
        self.agent = self._agent(env).to(self.device)
        if (self.agent.state_dim, self.agent.expected_obs_dim) != (env.state_dim, env.obs_dim):
            raise ValueError("failure-aware learner requires the failure-context observation")
        self.target_agent = self._agent(env).to(self.device)
        self.mixer = QMixer(self.n_agents, env.state_dim).to(self.device)
        self.target_mixer = QMixer(self.n_agents, env.state_dim).to(self.device)
        self.update_targets()
        self.optimizer = torch.optim.Adam(
            list(self.agent.parameters()) + list(self.mixer.parameters()), lr=lr
        )

    @staticmethod
    def _agent(env):
        return EpisodeFailureSpatialAgent(
            env.cfg.num_uavs,
            env.cfg.grid_size,
            env.cfg.coarse_obs_size,
            env.n_actions,
            len(env.bs_xy),
        )
