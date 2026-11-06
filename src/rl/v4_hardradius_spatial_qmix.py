from __future__ import annotations

import torch
from torch import nn

from src.rl.qmix import QMIXLearner, QMixer


class V4HardRadiusSpatialAgentQNetwork(nn.Module):
    """Compact spatial Q-network for the fixed-map hard-radius V4 core.

    Observation layout (per agent), matching V4FinalFixedMapEnv:
      state = coarse traffic map + 5 UAV * [x,y,relay,battery,active,returning,charging]
      obs   = state + 5-dim agent id

    The 5x5 CNN sees aggregate traffic plus coarse UAV/status maps reconstructed
    from the exact UAV coordinates. Exact UAV states are also retained in a
    vector branch so one-cell movement resolution is not lost.
    """

    UAV_FEAT = 7

    def __init__(
        self,
        n_agents: int,
        grid_size: int,
        coarse_size: int,
        n_actions: int,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.n_agents = int(n_agents)
        self.grid_size = int(grid_size)
        self.coarse_size = int(coarse_size)
        self.n_actions = int(n_actions)
        self.uav_block = self.n_agents * self.UAV_FEAT
        self.coarse_cells = self.coarse_size * self.coarse_size
        self.state_dim = self.uav_block + self.coarse_cells
        self.expected_obs_dim = self.state_dim + self.n_agents
        if self.grid_size % self.coarse_size != 0:
            raise ValueError("grid_size must be divisible by coarse_size")
        self.block = self.grid_size // self.coarse_size

        # traffic, Access, Relay, Returning, Charging, self
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 24, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        scalar_dim = self.uav_block + self.n_agents
        self.head = nn.Sequential(
            nn.Linear(24 * self.coarse_cells + scalar_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim != 2 or obs.shape[-1] != self.expected_obs_dim:
            raise ValueError(f"Expected [B,{self.expected_obs_dim}], got {tuple(obs.shape)}")
        b = obs.shape[0]
        state = obs[:, : self.state_dim]
        agent_id = obs[:, self.state_dim :]
        # Environment layout is [coarse demand, exact fleet state].  Earlier
        # compact-V4 runs accidentally parsed these blocks in reverse order,
        # which fed traffic values into the UAV-coordinate branch and fleet
        # values into the CNN.  Keep this split explicit and unit-tested.
        demand = state[:, : self.coarse_cells].view(b, 1, self.coarse_size, self.coarse_size)
        uav = state[:, self.coarse_cells :].view(b, self.n_agents, self.UAV_FEAT)

        # Exact coordinates are normalized to [0,1] in the environment.
        fine = torch.round(uav[:, :, :2] * max(self.grid_size - 1, 1)).long()
        fine = fine.clamp(0, self.grid_size - 1)
        coarse = torch.div(fine, self.block, rounding_mode="floor").clamp(0, self.coarse_size - 1)
        flat_idx = coarse[:, :, 0] * self.coarse_size + coarse[:, :, 1]

        relay = uav[:, :, 2].clamp(0.0, 1.0)
        active = uav[:, :, 4].clamp(0.0, 1.0)
        returning = uav[:, :, 5].clamp(0.0, 1.0)
        charging = uav[:, :, 6].clamp(0.0, 1.0)
        access = active * (1.0 - relay)
        relay = active * relay

        def scatter(weights: torch.Tensor) -> torch.Tensor:
            x = torch.zeros((b, self.coarse_cells), dtype=obs.dtype, device=obs.device)
            x.scatter_add_(1, flat_idx, weights)
            return x.clamp_max_(1.0).view(b, 1, self.coarse_size, self.coarse_size)

        access_map = scatter(access)
        relay_map = scatter(relay)
        returning_map = scatter(returning)
        charging_map = scatter(charging)

        self_idx = agent_id.argmax(dim=-1)
        batch_idx = torch.arange(b, device=obs.device)
        self_cell = flat_idx[batch_idx, self_idx]
        self_flat = torch.zeros((b, self.coarse_cells), dtype=obs.dtype, device=obs.device)
        self_flat.scatter_(1, self_cell.unsqueeze(1), 1.0)
        self_map = self_flat.view(b, 1, self.coarse_size, self.coarse_size)

        maps = torch.cat(
            [demand, access_map, relay_map, returning_map, charging_map, self_map],
            dim=1,
        )
        z = self.encoder(maps).flatten(1)
        scalar = torch.cat([uav.reshape(b, -1), agent_id], dim=1)
        return self.head(torch.cat([z, scalar], dim=1))


class V4HardRadiusSpatialQMIXLearner(QMIXLearner):
    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        state_dim: int,
        n_actions: int,
        grid_size: int = 20,
        coarse_size: int = 5,
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

        self.agent = V4HardRadiusSpatialAgentQNetwork(
            n_agents, grid_size, coarse_size, n_actions, hidden_dim
        ).to(self.device)
        if self.agent.expected_obs_dim != obs_dim:
            raise ValueError(
                f"Compact spatial parser expects obs_dim={self.agent.expected_obs_dim}, env has {obs_dim}"
            )
        if self.agent.state_dim != state_dim:
            raise ValueError(
                f"Compact spatial parser expects state_dim={self.agent.state_dim}, env has {state_dim}"
            )
        self.target_agent = V4HardRadiusSpatialAgentQNetwork(
            n_agents, grid_size, coarse_size, n_actions, hidden_dim
        ).to(self.device)
        self.mixer = QMixer(n_agents, state_dim, mixing_embed_dim).to(self.device)
        self.target_mixer = QMixer(n_agents, state_dim, mixing_embed_dim).to(self.device)
        self.update_targets()
        self.optimizer = torch.optim.Adam(
            list(self.agent.parameters()) + list(self.mixer.parameters()), lr=lr
        )
