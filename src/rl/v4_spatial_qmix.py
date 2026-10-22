from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from src.rl.qmix import QMIXLearner, QMixer


class V4SpatialAgentQNetwork(nn.Module):
    """Spatial encoder for V4 fixed-map observations.

    Observation layout is exactly V4FixedMobileBatteryEnv.get_obs():
      [global state, agent-id one-hot]

    State layout:
      per UAV: row, col, access_bit, relay_bit, battery, active, returning, charging
      demand map, surviving-BS map, failed-BS map, charging-station map

    The network turns the flat state back into spatial maps and keeps only
    battery/status scalars in a small vector branch. No extra environment
    information or hand-coded role score is introduced.
    """

    UAV_FEAT = 8

    def __init__(
        self,
        n_agents: int,
        grid_size: int,
        n_actions: int,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.n_agents = int(n_agents)
        self.grid_size = int(grid_size)
        self.n_actions = int(n_actions)
        self.uav_block = self.n_agents * self.UAV_FEAT
        self.map_cells = self.grid_size * self.grid_size
        self.state_dim = self.uav_block + 4 * self.map_cells
        self.expected_obs_dim = self.state_dim + self.n_agents

        # 9 channels: demand, surviving BS, failed BS, stations,
        # Access UAV, Relay UAV, Returning UAV, Charging UAV, self UAV.
        # V4 uses a 20x20 map. Pool to 5x5 before the CNN so training remains
        # practical on CPU. Demand is sum-pooled (load preserving); binary
        # entity maps are max-pooled (presence preserving).
        self.encoder = nn.Sequential(
            nn.Conv2d(9, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 24, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Keep exact per-UAV coordinates/roles/battery/status in the vector
        # branch. The CNN receives a coarse 5x5 spatial summary, while these
        # scalars preserve one-cell movement resolution.
        scalar_dim = self.n_agents * self.UAV_FEAT + self.n_agents
        self.head = nn.Sequential(
            nn.Linear(24 * 5 * 5 + scalar_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim != 2:
            raise ValueError("V4SpatialAgentQNetwork expects [B, obs_dim]")
        if obs.shape[-1] != self.expected_obs_dim:
            raise ValueError(f"Expected obs_dim={self.expected_obs_dim}, got {obs.shape[-1]}")

        b = obs.shape[0]
        g = self.grid_size
        state = obs[:, : self.state_dim]
        agent_id = obs[:, self.state_dim :]
        uav = state[:, : self.uav_block].view(b, self.n_agents, self.UAV_FEAT)

        p = self.uav_block
        demand = state[:, p : p + self.map_cells].view(b, 1, g, g)
        p += self.map_cells
        surviving = state[:, p : p + self.map_cells].view(b, 1, g, g)
        p += self.map_cells
        failed = state[:, p : p + self.map_cells].view(b, 1, g, g)
        p += self.map_cells
        stations = state[:, p : p + self.map_cells].view(b, 1, g, g)

        coords = torch.round(uav[:, :, :2] * max(g - 1, 1)).long().clamp(0, g - 1)
        flat_idx = coords[:, :, 0] * g + coords[:, :, 1]

        access_w = (uav[:, :, 2] > 0.5).to(obs.dtype)
        relay_w = (uav[:, :, 3] > 0.5).to(obs.dtype)
        returning_w = (uav[:, :, 6] > 0.5).to(obs.dtype)
        charging_w = (uav[:, :, 7] > 0.5).to(obs.dtype)

        def scatter_map(weights: torch.Tensor) -> torch.Tensor:
            x = torch.zeros((b, g * g), dtype=obs.dtype, device=obs.device)
            x.scatter_add_(1, flat_idx, weights)
            return x.clamp_max_(1.0).view(b, 1, g, g)

        access = scatter_map(access_w)
        relay = scatter_map(relay_w)
        returning = scatter_map(returning_w)
        charging = scatter_map(charging_w)

        self_idx = agent_id.argmax(dim=-1)
        batch_idx = torch.arange(b, device=obs.device)
        self_cell = flat_idx[batch_idx, self_idx]
        self_flat = torch.zeros((b, g * g), dtype=obs.dtype, device=obs.device)
        self_flat.scatter_(1, self_cell.unsqueeze(1), 1.0)
        self_map = self_flat.view(b, 1, g, g)

        # 4x4 pooling because the core map is 20x20.
        demand_p = F.avg_pool2d(demand, kernel_size=4, stride=4) * 16.0
        entity_maps = [
            surviving,
            failed,
            stations,
            access,
            relay,
            returning,
            charging,
            self_map,
        ]
        entity_p = [F.max_pool2d(x, kernel_size=4, stride=4) for x in entity_maps]
        maps = torch.cat([demand_p, *entity_p], dim=1)
        z = self.encoder(maps).flatten(1)

        # Preserve exact UAV state after coarse spatial pooling.
        scalar = torch.cat([uav.reshape(b, -1), agent_id], dim=1)
        return self.head(torch.cat([z, scalar], dim=1))


class V4SpatialQMIXLearner(QMIXLearner):
    """QMIX with the V4 spatial per-agent Q network."""

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        state_dim: int,
        n_actions: int,
        grid_size: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        hidden_dim: int = 64,
        mixing_embed_dim: int = 32,
        device: str = "cpu",
    ):
        self.n_agents = int(n_agents)
        self.n_actions = int(n_actions)
        self.gamma = float(gamma)
        self.device = torch.device(device)

        self.agent = V4SpatialAgentQNetwork(
            n_agents=n_agents,
            grid_size=grid_size,
            n_actions=n_actions,
            hidden_dim=hidden_dim,
        ).to(self.device)
        if self.agent.expected_obs_dim != obs_dim:
            raise ValueError(
                f"Spatial parser expects obs_dim={self.agent.expected_obs_dim}, env has {obs_dim}"
            )

        self.target_agent = V4SpatialAgentQNetwork(
            n_agents=n_agents,
            grid_size=grid_size,
            n_actions=n_actions,
            hidden_dim=hidden_dim,
        ).to(self.device)
        self.mixer = QMixer(n_agents, state_dim, mixing_embed_dim).to(self.device)
        self.target_mixer = QMixer(n_agents, state_dim, mixing_embed_dim).to(self.device)
        self.update_targets()

        self.optimizer = torch.optim.Adam(
            list(self.agent.parameters()) + list(self.mixer.parameters()), lr=lr
        )
