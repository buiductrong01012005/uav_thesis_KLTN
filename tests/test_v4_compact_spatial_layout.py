import torch

from src.envs.v4_final_fixed_map_env import V4FinalFixedMapConfig, V4FinalFixedMapEnv
from src.rl.v4_hardradius_spatial_qmix import V4HardRadiusSpatialAgentQNetwork


def test_compact_spatial_parser_matches_env_traffic_first_layout():
    env = V4FinalFixedMapEnv(
        V4FinalFixedMapConfig(
            coarse_obs_size=5,
            enable_ue_mobility=False,
            enable_battery_cycle=False,
        )
    )
    out = env.reset(seed=7)
    net = V4HardRadiusSpatialAgentQNetwork(5, 20, 5, 10)
    obs = torch.as_tensor(out["obs"][:1], dtype=torch.float32)
    state = obs[:, : net.state_dim]
    demand = state[:, : net.coarse_cells]
    fleet = state[:, net.coarse_cells :]

    # The first block must exactly match the environment's coarse traffic map.
    expected_demand = torch.as_tensor(env._coarse_demand_map().reshape(1, -1))
    assert torch.allclose(demand, expected_demand)

    # The fleet block starts with UAV0 normalized x/y, not traffic values.
    den = env.cfg.grid_size - 1
    expected_xy = torch.as_tensor(env.uav_positions[0] / den, dtype=torch.float32)
    assert torch.allclose(fleet[0, :2], expected_xy)

    q = net(obs)
    assert q.shape == (1, 10)


def test_compact_spatial_parser_supports_10x10_traffic():
    env = V4FinalFixedMapEnv(
        V4FinalFixedMapConfig(
            coarse_obs_size=10,
            enable_ue_mobility=False,
            enable_battery_cycle=False,
        )
    )
    out = env.reset(seed=8)
    net = V4HardRadiusSpatialAgentQNetwork(5, 20, 10, 10)
    q = net(torch.as_tensor(out["obs"][:2], dtype=torch.float32))
    assert q.shape == (2, 10)
