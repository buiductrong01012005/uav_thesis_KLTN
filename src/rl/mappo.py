from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


class SharedActor(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class CentralCritic(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


class MAPPO:
    """Small parameter-shared MAPPO implementation for the thesis V0 probe.

    - One shared decentralized actor is applied independently to each UAV obs.
    - A centralized critic consumes the global training state.
    - Common team rewards produce a common GAE advantage which is replicated
      across agents for the shared actor update.
    """

    def __init__(
        self,
        obs_dim: int,
        state_dim: int,
        n_actions: int,
        actor_lr: float = 3e-4,
        critic_lr: float = 5e-4,
        hidden_dim: int = 128,
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 10.0,
        device: str = "cpu",
    ):
        self.n_actions = n_actions
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = torch.device(device)

        self.actor = SharedActor(obs_dim, n_actions, hidden_dim).to(self.device)
        self.critic = CentralCritic(state_dim, hidden_dim).to(self.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def _masked_dist(self, obs_t: torch.Tensor, mask_t: torch.Tensor) -> Categorical:
        logits = self.actor(obs_t)
        logits = logits.masked_fill(~mask_t, -1e9)
        return Categorical(logits=logits)

    @torch.no_grad()
    def act(self, obs: np.ndarray, state: np.ndarray, action_mask: np.ndarray, deterministic: bool = False):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        state_t = torch.as_tensor(state[None, :], dtype=torch.float32, device=self.device)
        mask_t = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device)
        dist = self._masked_dist(obs_t, mask_t)
        if deterministic:
            logits = self.actor(obs_t).masked_fill(~mask_t, -1e9)
            actions = logits.argmax(dim=-1)
        else:
            actions = dist.sample()
        logp = dist.log_prob(actions)
        value = self.critic(state_t)[0]
        return actions.cpu().numpy().astype(np.int64), logp.cpu().numpy().astype(np.float32), float(value.item())

    @torch.no_grad()
    def value(self, state: np.ndarray) -> float:
        s = torch.as_tensor(state[None, :], dtype=torch.float32, device=self.device)
        return float(self.critic(s)[0].item())

    def update(
        self,
        obs: np.ndarray,
        states: np.ndarray,
        masks: np.ndarray,
        actions: np.ndarray,
        old_logp: np.ndarray,
        advantages: np.ndarray,
        returns: np.ndarray,
        epochs: int = 4,
        minibatch_size: int = 256,
        rng: np.random.Generator | None = None,
    ) -> dict:
        rng = rng or np.random.default_rng()
        T, N, obs_dim = obs.shape

        # Actor samples are agent-time pairs.
        actor_obs = obs.reshape(T * N, obs_dim)
        actor_masks = masks.reshape(T * N, self.n_actions)
        actor_actions = actions.reshape(T * N)
        actor_old_logp = old_logp.reshape(T * N)
        actor_adv = np.repeat(advantages, N)

        # Advantage normalization is standard PPO practice.
        actor_adv = (actor_adv - actor_adv.mean()) / (actor_adv.std() + 1e-8)

        actor_losses, entropies, critic_losses = [], [], []

        actor_count = T * N
        critic_count = T

        for _ in range(epochs):
            actor_idx = rng.permutation(actor_count)
            for start in range(0, actor_count, minibatch_size):
                idx = actor_idx[start : start + minibatch_size]
                o = torch.as_tensor(actor_obs[idx], dtype=torch.float32, device=self.device)
                m = torch.as_tensor(actor_masks[idx], dtype=torch.bool, device=self.device)
                a = torch.as_tensor(actor_actions[idx], dtype=torch.int64, device=self.device)
                old_lp = torch.as_tensor(actor_old_logp[idx], dtype=torch.float32, device=self.device)
                adv = torch.as_tensor(actor_adv[idx], dtype=torch.float32, device=self.device)

                dist = self._masked_dist(o, m)
                lp = dist.log_prob(a)
                ratio = torch.exp(lp - old_lp)
                unclipped = ratio * adv
                clipped = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv
                entropy = dist.entropy().mean()
                actor_loss = -torch.min(unclipped, clipped).mean() - self.entropy_coef * entropy

                self.actor_opt.zero_grad(set_to_none=True)
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_opt.step()

                actor_losses.append(float(actor_loss.item()))
                entropies.append(float(entropy.item()))

            critic_idx = rng.permutation(critic_count)
            for start in range(0, critic_count, minibatch_size):
                idx = critic_idx[start : start + minibatch_size]
                s = torch.as_tensor(states[idx], dtype=torch.float32, device=self.device)
                ret = torch.as_tensor(returns[idx], dtype=torch.float32, device=self.device)
                pred = self.critic(s)
                critic_loss = self.value_coef * ((pred - ret) ** 2).mean()

                self.critic_opt.zero_grad(set_to_none=True)
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_opt.step()
                critic_losses.append(float(critic_loss.item()))

        return {
            "actor_loss": float(np.mean(actor_losses)) if actor_losses else 0.0,
            "critic_loss": float(np.mean(critic_losses)) if critic_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
        }

    def save(self, path: str):
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
