from __future__ import annotations

from dataclasses import dataclass
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class RoleQNetwork(nn.Module):
    def __init__(self, state_dim: int, n_actions: int = 32, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x):
        return self.net(x)


@dataclass
class RoleTransition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: float


class RoleReplayBuffer:
    def __init__(self, capacity: int = 5000, seed: int = 0):
        self.capacity = int(capacity)
        self.data: list[RoleTransition] = []
        self.pos = 0
        self.rng = random.Random(seed)

    def add(self, x: RoleTransition):
        if len(self.data) < self.capacity:
            self.data.append(x)
        else:
            self.data[self.pos] = x
        self.pos = (self.pos + 1) % self.capacity

    def __len__(self):
        return len(self.data)

    def sample(self, batch_size: int):
        batch = self.rng.sample(self.data, batch_size)
        return {
            "state": np.stack([x.state for x in batch]).astype(np.float32),
            "action": np.asarray([x.action for x in batch], dtype=np.int64),
            "reward": np.asarray([x.reward for x in batch], dtype=np.float32),
            "next_state": np.stack([x.next_state for x in batch]).astype(np.float32),
            "done": np.asarray([x.done for x in batch], dtype=np.float32),
        }


class JointRoleDQN:
    """Small central DQN over the 2^5 Access/Relay role vectors."""

    def __init__(
        self,
        state_dim: int,
        n_actions: int = 32,
        lr: float = 3e-4,
        gamma: float = 0.99,
        hidden_dim: int = 128,
        device: str = "cpu",
    ):
        self.n_actions = int(n_actions)
        self.gamma = float(gamma)
        self.device = torch.device(device)
        self.q = RoleQNetwork(state_dim, n_actions, hidden_dim).to(self.device)
        self.target = RoleQNetwork(state_dim, n_actions, hidden_dim).to(self.device)
        self.target.load_state_dict(self.q.state_dict())
        self.optim = torch.optim.Adam(self.q.parameters(), lr=lr)

    @torch.no_grad()
    def act(self, state: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
        if rng.random() < epsilon:
            return int(rng.integers(self.n_actions))
        x = torch.as_tensor(state[None], dtype=torch.float32, device=self.device)
        return int(self.q(x).argmax(dim=-1).item())

    def train_batch(self, batch: dict) -> float:
        s = torch.as_tensor(batch["state"], dtype=torch.float32, device=self.device)
        a = torch.as_tensor(batch["action"], dtype=torch.int64, device=self.device)
        r = torch.as_tensor(batch["reward"], dtype=torch.float32, device=self.device)
        ns = torch.as_tensor(batch["next_state"], dtype=torch.float32, device=self.device)
        d = torch.as_tensor(batch["done"], dtype=torch.float32, device=self.device)
        q = self.q(s).gather(1, a[:, None]).squeeze(1)
        with torch.no_grad():
            na = self.q(ns).argmax(dim=1, keepdim=True)
            nq = self.target(ns).gather(1, na).squeeze(1)
            y = r + self.gamma * (1.0 - d) * nq
        loss = F.smooth_l1_loss(q, y)
        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.optim.step()
        return float(loss.item())

    def update_target(self, tau: float = 1.0):
        if tau == 1.0:
            self.target.load_state_dict(self.q.state_dict())
            return
        with torch.no_grad():
            for t, s in zip(self.target.parameters(), self.q.parameters()):
                t.data.mul_(1.0 - tau).add_(s.data, alpha=tau)

    def save(self, path: str):
        torch.save({"q": self.q.state_dict()}, path)

    def load(self, path: str):
        ck = torch.load(path, map_location=self.device)
        self.q.load_state_dict(ck["q"])
        self.target.load_state_dict(self.q.state_dict())


class SpatialRoleQNetwork(nn.Module):
    """Light spatial global-state encoder for the 5-UAV role controller."""

    UAV_FEAT = 8

    def __init__(
        self,
        state_dim: int,
        n_agents: int = 5,
        grid_size: int = 20,
        n_actions: int = 32,
        hidden_dim: int = 96,
    ):
        super().__init__()
        import torch.nn.functional as F  # noqa: F401 - documented dependency for forward

        self.n_agents = n_agents
        self.grid_size = grid_size
        self.n_actions = n_actions
        self.uav_block = n_agents * self.UAV_FEAT
        self.cells = grid_size * grid_size
        self.expected_state_dim = self.uav_block + 4 * self.cells
        if state_dim != self.expected_state_dim:
            raise ValueError(f"Expected V4 state_dim={self.expected_state_dim}, got {state_dim}")
        self.encoder = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 24, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        scalar_dim = n_agents * self.UAV_FEAT  # exact UAV state + coarse maps
        self.head = nn.Sequential(
            nn.Linear(24 * 5 * 5 + scalar_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, state):
        import torch.nn.functional as F

        if state.ndim != 2 or state.shape[-1] != self.expected_state_dim:
            raise ValueError("SpatialRoleQNetwork state shape mismatch")
        b = state.shape[0]
        g = self.grid_size
        uav = state[:, : self.uav_block].view(b, self.n_agents, self.UAV_FEAT)
        p = self.uav_block
        demand = state[:, p : p + self.cells].view(b, 1, g, g)
        p += self.cells
        surviving = state[:, p : p + self.cells].view(b, 1, g, g)
        p += self.cells
        failed = state[:, p : p + self.cells].view(b, 1, g, g)
        p += self.cells
        stations = state[:, p : p + self.cells].view(b, 1, g, g)

        coords = torch.round(uav[:, :, :2] * max(g - 1, 1)).long().clamp(0, g - 1)
        idx = coords[:, :, 0] * g + coords[:, :, 1]

        def scatter(weights):
            x = torch.zeros((b, g * g), dtype=state.dtype, device=state.device)
            x.scatter_add_(1, idx, weights)
            return x.clamp_max_(1.0).view(b, 1, g, g)

        access = scatter((uav[:, :, 2] > 0.5).to(state.dtype))
        relay = scatter((uav[:, :, 3] > 0.5).to(state.dtype))
        returning = scatter((uav[:, :, 6] > 0.5).to(state.dtype))
        charging = scatter((uav[:, :, 7] > 0.5).to(state.dtype))

        demand_p = F.avg_pool2d(demand, 4, 4) * 16.0
        entities = [surviving, failed, stations, access, relay, returning, charging]
        ent_p = [F.max_pool2d(x, 4, 4) for x in entities]
        maps = torch.cat([demand_p, *ent_p], dim=1)
        z = self.encoder(maps).flatten(1)
        scalars = uav.reshape(b, -1)
        return self.head(torch.cat([z, scalars], dim=1))


class SpatialJointRoleDQN(JointRoleDQN):
    def __init__(
        self,
        state_dim: int,
        n_actions: int = 32,
        n_agents: int = 5,
        grid_size: int = 20,
        lr: float = 3e-4,
        gamma: float = 0.99,
        hidden_dim: int = 96,
        device: str = "cpu",
    ):
        self.n_actions = int(n_actions)
        self.gamma = float(gamma)
        self.device = torch.device(device)
        self.q = SpatialRoleQNetwork(state_dim, n_agents, grid_size, n_actions, hidden_dim).to(
            self.device
        )
        self.target = SpatialRoleQNetwork(state_dim, n_agents, grid_size, n_actions, hidden_dim).to(
            self.device
        )
        self.target.load_state_dict(self.q.state_dict())
        self.optim = torch.optim.Adam(self.q.parameters(), lr=lr)
