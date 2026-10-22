"""QMIX with a small spatial encoder for grid observations.

Only the per-agent Q network changes. The task, reward, mixer, optimizer and
training rule remain identical to the MLP-QMIX baseline. This makes the module
a clean architecture audit: does an inductive bias for 2-D geometry improve
spatial generalization without adding reward shaping or hand-coded policy logic?
"""

from __future__ import annotations

import torch
from torch import nn

from src.rl.qmix import QMIXLearner, QMixer


class SpatialAgentQNetwork(nn.Module):
    """Parse the existing flat Stage-B observation into spatial channels.

    Expected flat observation layout (from StageB/V0):
      [for each UAV: x, y, access_bit, relay_bit],
      [UE map], [BS map], [agent-id one-hot].

    Constructed channels per controlled agent:
      0 UE density
      1 surviving-BS location
      2 Access-UAV locations
      3 Relay-UAV locations
      4 unavailable-UAV locations
      5 controlled-agent location

    The network still accepts the exact same flat observation tensor, so no
    environment/interface changes are required.
    """

    def __init__(self, n_agents: int, grid_size: int, n_actions: int, hidden_dim: int = 64):
        super().__init__()
        self.n_agents = n_agents
        self.grid_size = grid_size
        self.n_actions = n_actions
        self.uav_feature_dim = 4
        self.uav_block = n_agents * self.uav_feature_dim
        self.map_cells = grid_size * grid_size
        self.expected_obs_dim = self.uav_block + 2 * self.map_cells + n_agents

        self.encoder = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 3)),
        )
        self.head = nn.Sequential(
            nn.Linear(32 * 3 * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim != 2:
            raise ValueError("SpatialAgentQNetwork expects [B, obs_dim]")
        if obs.shape[-1] != self.expected_obs_dim:
            raise ValueError(f"Expected obs_dim={self.expected_obs_dim}, got {obs.shape[-1]}")

        b = obs.shape[0]
        g = self.grid_size
        uav = obs[:, : self.uav_block].view(b, self.n_agents, self.uav_feature_dim)
        ue0 = self.uav_block
        ue = obs[:, ue0 : ue0 + self.map_cells].view(b, 1, g, g)
        bs0 = ue0 + self.map_cells
        bs = obs[:, bs0 : bs0 + self.map_cells].view(b, 1, g, g)
        agent_id = obs[:, bs0 + self.map_cells :]
        self_idx = agent_id.argmax(dim=-1)

        # Positions are normalized to [0,1]. Convert back to grid indices.
        coords = torch.round(uav[:, :, :2] * max(g - 1, 1)).long()
        coords = coords.clamp(0, g - 1)
        flat_idx = coords[:, :, 0] * g + coords[:, :, 1]  # [B,N]

        # Build role maps without a Python loop.  scatter_add is equivalent to
        # placing a 1 at every UAV cell; clamp keeps the maps binary if two
        # UAVs occupy the same cell.  This is purely an implementation-speed
        # change: the architecture and observations are unchanged.
        access_flat = torch.zeros((b, g * g), dtype=obs.dtype, device=obs.device)
        relay_flat = torch.zeros_like(access_flat)
        inactive_flat = torch.zeros_like(access_flat)

        is_access = (uav[:, :, 2] > 0.5).to(obs.dtype)
        is_relay = (uav[:, :, 3] > 0.5).to(obs.dtype)
        is_inactive = ((is_access == 0) & (is_relay == 0)).to(obs.dtype)

        access_flat.scatter_add_(1, flat_idx, is_access)
        relay_flat.scatter_add_(1, flat_idx, is_relay)
        inactive_flat.scatter_add_(1, flat_idx, is_inactive)
        access = access_flat.clamp_max_(1.0).view(b, 1, g, g)
        relay = relay_flat.clamp_max_(1.0).view(b, 1, g, g)
        inactive = inactive_flat.clamp_max_(1.0).view(b, 1, g, g)

        self_flat = torch.zeros((b, g * g), dtype=obs.dtype, device=obs.device)
        batch_idx = torch.arange(b, device=obs.device)
        self_cell = flat_idx[batch_idx, self_idx]
        self_flat.scatter_(1, self_cell.unsqueeze(1), 1.0)
        self_map = self_flat.view(b, 1, g, g)

        x = torch.cat([ue, bs, access, relay, inactive, self_map], dim=1)
        z = self.encoder(x).flatten(1)
        return self.head(z)


class SpatialQMIXLearner(QMIXLearner):
    """Drop-in QMIX learner using SpatialAgentQNetwork."""

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        state_dim: int,
        n_actions: int,
        grid_size: int,
        lr: float = 5e-4,
        gamma: float = 0.99,
        hidden_dim: int = 64,
        mixing_embed_dim: int = 32,
        device: str = "cpu",
    ):
        # Do not call QMIXLearner.__init__, because it constructs MLP agents.
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.gamma = gamma
        self.device = torch.device(device)

        self.agent = SpatialAgentQNetwork(
            n_agents=n_agents,
            grid_size=grid_size,
            n_actions=n_actions,
            hidden_dim=hidden_dim,
        ).to(self.device)
        if self.agent.expected_obs_dim != obs_dim:
            raise ValueError(
                f"Spatial parser expects obs_dim={self.agent.expected_obs_dim}, environment has {obs_dim}"
            )

        self.mixer = QMixer(n_agents, state_dim, mixing_embed_dim).to(self.device)
        self.target_agent = SpatialAgentQNetwork(
            n_agents=n_agents,
            grid_size=grid_size,
            n_actions=n_actions,
            hidden_dim=hidden_dim,
        ).to(self.device)
        self.target_mixer = QMixer(n_agents, state_dim, mixing_embed_dim).to(self.device)
        self.update_targets()

        params = list(self.agent.parameters()) + list(self.mixer.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr)
