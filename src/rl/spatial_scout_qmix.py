"""Spatial QMIX encoder for Stage-A cooperative scouting.

The Scout task is purely spatial: agents need to spread over a grid according
 to a shared known/unknown mask.  This module keeps the Stage-A environment,
 reward, centralized mixer, optimizer and QMIX update unchanged and replaces
 only the per-agent flat MLP with a small convolutional encoder.

It is intentionally tied to ``RandomizedStageACoverageOnlyEnv`` so hidden or
 discovered UE values cannot leak into the Scout policy.  Expected observation
 layout is:
    [all UAV (row,col) positions], [known-mask], [agent-id one-hot].
"""

from __future__ import annotations

import torch
from torch import nn

from src.rl.qmix import QMIXLearner, QMixer


class SpatialScoutAgentQNetwork(nn.Module):
    """CNN Q-network for the coverage-only Stage-A observation.

    Spatial channels for each controlled UAV:
      0 unknown-cell mask (1 = still needs scouting)
      1 occupancy of all UAVs
      2 controlled-UAV location

    No UE-demand channel is used: Stage A optimizes coverage only.
    """

    def __init__(self, n_agents: int, grid_size: int, n_actions: int = 5, hidden_dim: int = 64):
        super().__init__()
        self.n_agents = int(n_agents)
        self.grid_size = int(grid_size)
        self.n_actions = int(n_actions)
        self.position_block = 2 * self.n_agents
        self.map_cells = self.grid_size * self.grid_size
        self.expected_obs_dim = self.position_block + self.map_cells + self.n_agents

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 3)),
        )
        self.head = nn.Sequential(
            nn.Linear(32 * 3 * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim != 2:
            raise ValueError("SpatialScoutAgentQNetwork expects [B, obs_dim]")
        if obs.shape[-1] != self.expected_obs_dim:
            raise ValueError(f"Expected obs_dim={self.expected_obs_dim}, got {obs.shape[-1]}")

        b = obs.shape[0]
        g = self.grid_size
        positions = obs[:, : self.position_block].view(b, self.n_agents, 2)
        known0 = self.position_block
        known = obs[:, known0 : known0 + self.map_cells].view(b, 1, g, g)
        agent_id = obs[:, known0 + self.map_cells :]
        self_idx = agent_id.argmax(dim=-1)

        # Environment positions are normalized to [0,1].
        coords = torch.round(positions * max(g - 1, 1)).long().clamp(0, g - 1)
        flat_idx = coords[:, :, 0] * g + coords[:, :, 1]

        fleet_flat = torch.zeros((b, g * g), dtype=obs.dtype, device=obs.device)
        fleet_flat.scatter_add_(1, flat_idx, torch.ones_like(flat_idx, dtype=obs.dtype))
        fleet = fleet_flat.clamp_max_(1.0).view(b, 1, g, g)

        batch_idx = torch.arange(b, device=obs.device)
        self_cell = flat_idx[batch_idx, self_idx]
        self_flat = torch.zeros((b, g * g), dtype=obs.dtype, device=obs.device)
        self_flat.scatter_(1, self_cell.unsqueeze(1), 1.0)
        self_map = self_flat.view(b, 1, g, g)

        unknown = 1.0 - known
        x = torch.cat([unknown, fleet, self_map], dim=1)
        return self.head(self.encoder(x).flatten(1))


class SpatialScoutQMIXLearner(QMIXLearner):
    """Drop-in QMIX learner with the Stage-A spatial Scout encoder."""

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
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.gamma = gamma
        self.device = torch.device(device)

        self.agent = SpatialScoutAgentQNetwork(
            n_agents=n_agents,
            grid_size=grid_size,
            n_actions=n_actions,
            hidden_dim=hidden_dim,
        ).to(self.device)
        if self.agent.expected_obs_dim != obs_dim:
            raise ValueError(
                f"Spatial Scout parser expects obs_dim={self.agent.expected_obs_dim}, "
                f"environment has {obs_dim}. Use RandomizedStageACoverageOnlyEnv."
            )

        self.mixer = QMixer(n_agents, state_dim, mixing_embed_dim).to(self.device)
        self.target_agent = SpatialScoutAgentQNetwork(
            n_agents=n_agents,
            grid_size=grid_size,
            n_actions=n_actions,
            hidden_dim=hidden_dim,
        ).to(self.device)
        self.target_mixer = QMixer(n_agents, state_dim, mixing_embed_dim).to(self.device)
        self.update_targets()

        params = list(self.agent.parameters()) + list(self.mixer.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr)
