"""MAPPO with the same spatial actor encoder used by Spatial-QMIX.

This module is intentionally minimal for an algorithm-family audit. The task,
reward, observations, action space, and spatial inductive bias are unchanged;
only the MARL learning rule differs from QMIX.
"""

from __future__ import annotations

import torch

from src.rl.mappo import MAPPO
from src.rl.spatial_qmix import SpatialAgentQNetwork


class SpatialMAPPO(MAPPO):
    """Parameter-shared MAPPO with a CNN actor and centralized MLP critic.

    The actor is exactly the same spatial per-agent architecture used by
    Spatial-QMIX, except its outputs are interpreted as policy logits instead
    of Q-values. This keeps the representation audit controlled.
    """

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        state_dim: int,
        n_actions: int,
        grid_size: int,
        actor_lr: float = 3e-4,
        critic_lr: float = 5e-4,
        hidden_dim: int = 64,
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 10.0,
        device: str = "cpu",
    ):
        # Build the standard centralized critic and PPO machinery first.
        super().__init__(
            obs_dim=obs_dim,
            state_dim=state_dim,
            n_actions=n_actions,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            hidden_dim=hidden_dim,
            clip_ratio=clip_ratio,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
            max_grad_norm=max_grad_norm,
            device=device,
        )
        self.n_agents = n_agents
        self.grid_size = grid_size

        # Replace only the actor representation. Everything else in MAPPO is
        # unchanged, including masked categorical actions and PPO updates.
        self.actor = SpatialAgentQNetwork(
            n_agents=n_agents,
            grid_size=grid_size,
            n_actions=n_actions,
            hidden_dim=hidden_dim,
        ).to(self.device)
        if self.actor.expected_obs_dim != obs_dim:
            raise ValueError(
                f"Spatial parser expects obs_dim={self.actor.expected_obs_dim}, "
                f"environment has {obs_dim}"
            )
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
