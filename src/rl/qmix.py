from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class AgentQNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class QMixer(nn.Module):
    """Standard monotonic QMIX-style mixing network."""

    def __init__(self, n_agents: int, state_dim: int, mixing_embed_dim: int = 32):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.embed_dim = mixing_embed_dim

        self.hyper_w1 = nn.Linear(state_dim, n_agents * mixing_embed_dim)
        self.hyper_b1 = nn.Linear(state_dim, mixing_embed_dim)

        self.hyper_w_final = nn.Linear(state_dim, mixing_embed_dim)
        self.v = nn.Sequential(
            nn.Linear(state_dim, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, 1),
        )

    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        # agent_qs: [B, n_agents]
        # states:    [B, state_dim]
        batch = agent_qs.shape[0]

        w1 = torch.abs(self.hyper_w1(states)).view(batch, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(states).view(batch, 1, self.embed_dim)

        hidden = F.elu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)

        w_final = torch.abs(self.hyper_w_final(states)).view(batch, self.embed_dim, 1)
        v = self.v(states).view(batch, 1, 1)

        q_tot = torch.bmm(hidden, w_final) + v
        return q_tot.view(batch)


@dataclass
class Transition:
    obs: np.ndarray
    state: np.ndarray
    actions: np.ndarray
    reward: float
    next_obs: np.ndarray
    next_state: np.ndarray
    next_action_mask: np.ndarray
    done: float


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 0):
        # A list-backed circular buffer keeps random indexing O(1).  The
        # previous deque implementation is semantically fine but
        # random.sample(deque, ...) becomes increasingly expensive because
        # deque indexing is O(n), which made long Spatial-QMIX runs appear to
        # scale poorly.  Sampling remains uniform over the stored transitions.
        self.capacity = int(capacity)
        self.data: list[Transition] = []
        self.pos = 0
        self.rng = random.Random(seed)

    def add(self, transition: Transition):
        if len(self.data) < self.capacity:
            self.data.append(transition)
        else:
            self.data[self.pos] = transition
        self.pos = (self.pos + 1) % self.capacity

    def __len__(self):
        return len(self.data)

    def sample(self, batch_size: int):
        batch = self.rng.sample(self.data, batch_size)

        def stack(name):
            return np.stack([getattr(x, name) for x in batch], axis=0)

        return {
            "obs": stack("obs").astype(np.float32),
            "state": stack("state").astype(np.float32),
            "actions": stack("actions").astype(np.int64),
            "reward": np.asarray([x.reward for x in batch], dtype=np.float32),
            "next_obs": stack("next_obs").astype(np.float32),
            "next_state": stack("next_state").astype(np.float32),
            "next_action_mask": stack("next_action_mask").astype(np.bool_),
            "done": np.asarray([x.done for x in batch], dtype=np.float32),
        }


class QMIXLearner:
    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        state_dim: int,
        n_actions: int,
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

        self.agent = AgentQNetwork(obs_dim, n_actions, hidden_dim).to(self.device)
        self.mixer = QMixer(n_agents, state_dim, mixing_embed_dim).to(self.device)

        self.target_agent = AgentQNetwork(obs_dim, n_actions, hidden_dim).to(self.device)
        self.target_mixer = QMixer(n_agents, state_dim, mixing_embed_dim).to(self.device)
        self.update_targets()

        params = list(self.agent.parameters()) + list(self.mixer.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr)

    @torch.no_grad()
    def act(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        epsilon: float,
        rng: np.random.Generator,
    ):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        q = self.agent(obs_t).cpu().numpy()

        actions = []
        for i in range(self.n_agents):
            valid = np.flatnonzero(action_mask[i])
            if rng.random() < epsilon:
                actions.append(int(rng.choice(valid)))
            else:
                qi = q[i].copy()
                qi[~action_mask[i].astype(bool)] = -1e9
                actions.append(int(np.argmax(qi)))
        return actions

    def train_batch(self, batch: dict) -> float:
        obs = torch.as_tensor(batch["obs"], device=self.device)
        states = torch.as_tensor(batch["state"], device=self.device)
        actions = torch.as_tensor(batch["actions"], device=self.device)
        rewards = torch.as_tensor(batch["reward"], device=self.device)
        next_obs = torch.as_tensor(batch["next_obs"], device=self.device)
        next_states = torch.as_tensor(batch["next_state"], device=self.device)
        next_masks = torch.as_tensor(batch["next_action_mask"], device=self.device)
        done = torch.as_tensor(batch["done"], device=self.device)

        B = obs.shape[0]

        q_all = self.agent(obs.view(B * self.n_agents, -1)).view(B, self.n_agents, self.n_actions)
        chosen_q = torch.gather(q_all, 2, actions.unsqueeze(-1)).squeeze(-1)
        q_tot = self.mixer(chosen_q, states)

        with torch.no_grad():
            # Double-Q: online network selects, target network evaluates.
            online_next_q = self.agent(next_obs.view(B * self.n_agents, -1)).view(
                B, self.n_agents, self.n_actions
            )
            online_next_q = online_next_q.masked_fill(~next_masks, -1e9)
            next_actions = online_next_q.argmax(dim=-1, keepdim=True)

            target_next_q_all = self.target_agent(next_obs.view(B * self.n_agents, -1)).view(
                B, self.n_agents, self.n_actions
            )
            target_next_q = torch.gather(target_next_q_all, 2, next_actions).squeeze(-1)

            target_q_tot = self.target_mixer(target_next_q, next_states)
            td_target = rewards + self.gamma * (1.0 - done) * target_q_tot

        loss = F.smooth_l1_loss(q_tot, td_target)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.agent.parameters()) + list(self.mixer.parameters()), 10.0
        )
        self.optimizer.step()

        return float(loss.item())

    def update_targets(self, tau: float = 1.0):
        """Update target networks.

        tau=1.0 performs the original hard copy. For 0<tau<1, use a
        Polyak/soft update, which can reduce oscillation in deterministic
        small environments.
        """
        if not 0.0 < tau <= 1.0:
            raise ValueError("tau must be in (0, 1].")
        if tau == 1.0:
            self.target_agent.load_state_dict(self.agent.state_dict())
            self.target_mixer.load_state_dict(self.mixer.state_dict())
            return

        with torch.no_grad():
            for target, source in zip(self.target_agent.parameters(), self.agent.parameters()):
                target.data.mul_(1.0 - tau).add_(source.data, alpha=tau)
            for target, source in zip(self.target_mixer.parameters(), self.mixer.parameters()):
                target.data.mul_(1.0 - tau).add_(source.data, alpha=tau)

    def save(self, path: str):
        torch.save(
            {
                "agent": self.agent.state_dict(),
                "mixer": self.mixer.state_dict(),
            },
            path,
        )

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.agent.load_state_dict(checkpoint["agent"])
        self.mixer.load_state_dict(checkpoint["mixer"])
        self.update_targets()


class SeparateAgentQMIXLearner:
    """QMIX variant with one Q-network per UAV instead of parameter sharing.

    This is useful for the small fixed-size thesis fleet because it removes a
    major symmetry-breaking burden from the shared policy while retaining the
    same monotonic centralized mixer and decentralized per-UAV execution.
    """

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        state_dim: int,
        n_actions: int,
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

        self.agents = nn.ModuleList(
            [AgentQNetwork(obs_dim, n_actions, hidden_dim) for _ in range(n_agents)]
        ).to(self.device)
        self.target_agents = nn.ModuleList(
            [AgentQNetwork(obs_dim, n_actions, hidden_dim) for _ in range(n_agents)]
        ).to(self.device)
        self.mixer = QMixer(n_agents, state_dim, mixing_embed_dim).to(self.device)
        self.target_mixer = QMixer(n_agents, state_dim, mixing_embed_dim).to(self.device)
        self.update_targets()

        params = list(self.agents.parameters()) + list(self.mixer.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr)

    @torch.no_grad()
    def act(self, obs, action_mask, epsilon, rng):
        actions = []
        for i in range(self.n_agents):
            valid = np.flatnonzero(action_mask[i])
            if rng.random() < epsilon:
                actions.append(int(rng.choice(valid)))
                continue
            o = torch.as_tensor(obs[i : i + 1], dtype=torch.float32, device=self.device)
            q = self.agents[i](o)[0].cpu().numpy()
            q[~action_mask[i].astype(bool)] = -1e9
            actions.append(int(np.argmax(q)))
        return actions

    def _all_q(self, networks, obs):
        # obs: [B,N,O] -> [B,N,A]
        return torch.stack(
            [networks[i](obs[:, i, :]) for i in range(self.n_agents)],
            dim=1,
        )

    def train_batch(self, batch):
        obs = torch.as_tensor(batch["obs"], device=self.device)
        states = torch.as_tensor(batch["state"], device=self.device)
        actions = torch.as_tensor(batch["actions"], device=self.device)
        rewards = torch.as_tensor(batch["reward"], device=self.device)
        next_obs = torch.as_tensor(batch["next_obs"], device=self.device)
        next_states = torch.as_tensor(batch["next_state"], device=self.device)
        next_masks = torch.as_tensor(batch["next_action_mask"], device=self.device)
        done = torch.as_tensor(batch["done"], device=self.device)

        q_all = self._all_q(self.agents, obs)
        chosen_q = torch.gather(q_all, 2, actions.unsqueeze(-1)).squeeze(-1)
        q_tot = self.mixer(chosen_q, states)

        with torch.no_grad():
            online_next = self._all_q(self.agents, next_obs).masked_fill(~next_masks, -1e9)
            next_actions = online_next.argmax(dim=-1, keepdim=True)
            target_next_all = self._all_q(self.target_agents, next_obs)
            target_next_q = torch.gather(target_next_all, 2, next_actions).squeeze(-1)
            target_q_tot = self.target_mixer(target_next_q, next_states)
            td_target = rewards + self.gamma * (1.0 - done) * target_q_tot

        loss = F.smooth_l1_loss(q_tot, td_target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.agents.parameters()) + list(self.mixer.parameters()), 10.0
        )
        self.optimizer.step()
        return float(loss.item())

    def update_targets(self, tau: float = 1.0):
        if not 0.0 < tau <= 1.0:
            raise ValueError("tau must be in (0,1].")
        if tau == 1.0:
            self.target_agents.load_state_dict(self.agents.state_dict())
            self.target_mixer.load_state_dict(self.mixer.state_dict())
            return
        with torch.no_grad():
            for target, source in zip(self.target_agents.parameters(), self.agents.parameters()):
                target.data.mul_(1.0 - tau).add_(source.data, alpha=tau)
            for target, source in zip(self.target_mixer.parameters(), self.mixer.parameters()):
                target.data.mul_(1.0 - tau).add_(source.data, alpha=tau)

    def save(self, path: str):
        torch.save({"agents": self.agents.state_dict(), "mixer": self.mixer.state_dict()}, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.agents.load_state_dict(checkpoint["agents"])
        self.mixer.load_state_dict(checkpoint["mixer"])
        self.update_targets()
