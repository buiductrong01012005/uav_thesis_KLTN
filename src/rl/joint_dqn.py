from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class JointQNetwork(nn.Module):
    def __init__(self, state_dim: int, n_joint_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_joint_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


@dataclass
class JointTransition:
    state: np.ndarray
    joint_action: int
    reward: float
    next_state: np.ndarray
    next_joint_mask: np.ndarray
    done: float


class JointReplayBuffer:
    def __init__(self, capacity: int, seed: int = 0):
        self.data = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def add(self, transition: JointTransition):
        self.data.append(transition)

    def __len__(self):
        return len(self.data)

    def sample(self, batch_size: int):
        batch = self.rng.sample(self.data, batch_size)
        return {
            "state": np.stack([x.state for x in batch]).astype(np.float32),
            "joint_action": np.asarray([x.joint_action for x in batch], dtype=np.int64),
            "reward": np.asarray([x.reward for x in batch], dtype=np.float32),
            "next_state": np.stack([x.next_state for x in batch]).astype(np.float32),
            "next_joint_mask": np.stack([x.next_joint_mask for x in batch]).astype(np.bool_),
            "done": np.asarray([x.done for x in batch], dtype=np.float32),
        }


class JointDQN:
    def __init__(
        self,
        state_dim: int,
        n_agents: int,
        n_local_actions: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        hidden_dim: int = 128,
        device: str = "cpu",
    ):
        self.n_agents = n_agents
        self.n_local_actions = n_local_actions
        self.n_joint_actions = n_local_actions ** n_agents
        self.gamma = gamma
        self.device = torch.device(device)

        # Precompute all joint-action tuples once. This makes action masking
        # vectorized instead of decoding 1000 actions in Python every step.
        self._joint_actions = np.asarray(
            [self._decode_index(idx) for idx in range(self.n_joint_actions)],
            dtype=np.int64,
        )

        self.q = JointQNetwork(state_dim, self.n_joint_actions, hidden_dim).to(self.device)
        self.target_q = JointQNetwork(state_dim, self.n_joint_actions, hidden_dim).to(self.device)
        self.target_q.load_state_dict(self.q.state_dict())
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=lr)

    def encode_joint_action(self, local_actions) -> int:
        idx = 0
        for a in local_actions:
            idx = idx * self.n_local_actions + int(a)
        return idx

    def _decode_index(self, idx: int) -> list[int]:
        actions = [0] * self.n_agents
        x = int(idx)
        for i in range(self.n_agents - 1, -1, -1):
            actions[i] = x % self.n_local_actions
            x //= self.n_local_actions
        return actions

    def decode_joint_action(self, idx: int) -> list[int]:
        return self._joint_actions[int(idx)].tolist()

    def joint_mask(self, local_mask: np.ndarray) -> np.ndarray:
        # local_mask: [n_agents, n_local_actions]
        # Gather each agent's validity for every precomputed joint tuple.
        validity = np.ones(self.n_joint_actions, dtype=bool)
        for i in range(self.n_agents):
            validity &= local_mask[i, self._joint_actions[:, i]].astype(bool)
        return validity

    @torch.no_grad()
    def act(self, state: np.ndarray, local_mask: np.ndarray, epsilon: float, rng: np.random.Generator):
        mask = self.joint_mask(local_mask)
        valid = np.flatnonzero(mask)
        if rng.random() < epsilon:
            joint = int(rng.choice(valid))
        else:
            s = torch.as_tensor(state[None, :], dtype=torch.float32, device=self.device)
            q = self.q(s)[0].cpu().numpy()
            q[~mask] = -1e9
            joint = int(np.argmax(q))
        return joint, self.decode_joint_action(joint), mask

    def train_batch(self, batch: dict) -> float:
        s = torch.as_tensor(batch["state"], dtype=torch.float32, device=self.device)
        a = torch.as_tensor(batch["joint_action"], dtype=torch.int64, device=self.device)
        r = torch.as_tensor(batch["reward"], dtype=torch.float32, device=self.device)
        ns = torch.as_tensor(batch["next_state"], dtype=torch.float32, device=self.device)
        nm = torch.as_tensor(batch["next_joint_mask"], dtype=torch.bool, device=self.device)
        done = torch.as_tensor(batch["done"], dtype=torch.float32, device=self.device)

        q_all = self.q(s)
        chosen = q_all.gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            online_next = self.q(ns).masked_fill(~nm, -1e9)
            next_a = online_next.argmax(dim=1, keepdim=True)
            target_next = self.target_q(ns).gather(1, next_a).squeeze(1)
            target = r + self.gamma * (1.0 - done) * target_next

        loss = F.smooth_l1_loss(chosen, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.optimizer.step()
        return float(loss.item())

    def soft_update(self, tau: float = 0.01):
        with torch.no_grad():
            for target, source in zip(self.target_q.parameters(), self.q.parameters()):
                target.data.mul_(1.0 - tau).add_(source.data, alpha=tau)

    def save(self, path: str):
        torch.save({"q": self.q.state_dict()}, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.q.load_state_dict(checkpoint["q"])
        self.target_q.load_state_dict(self.q.state_dict())
