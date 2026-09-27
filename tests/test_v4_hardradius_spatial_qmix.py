import numpy as np
import torch

from src.envs.v4_fixed_hardradius_battery_env import V4FixedHardRadiusEnv
from src.rl.v4_hardradius_spatial_qmix import (
    V4HardRadiusSpatialAgentQNetwork,
    V4HardRadiusSpatialQMIXLearner,
)


def test_compact_spatial_network_matches_hardradius_obs():
    env = V4FixedHardRadiusEnv()
    out = env.reset(seed=7)
    net = V4HardRadiusSpatialAgentQNetwork(5, 20, 5, 10)
    q = net(torch.as_tensor(out["obs"], dtype=torch.float32))
    assert q.shape == (5, 10)
    assert torch.isfinite(q).all()


def test_compact_spatial_qmix_actions_respect_mask():
    env = V4FixedHardRadiusEnv()
    out = env.reset(seed=8)
    learner = V4HardRadiusSpatialQMIXLearner(
        5, env.obs_dim, env.state_dim, 10, grid_size=20, coarse_size=5
    )
    acts = learner.act(out["obs"], out["action_mask"], 0.0, np.random.default_rng(1))
    assert len(acts) == 5
    assert all(out["action_mask"][i, a] for i, a in enumerate(acts))
