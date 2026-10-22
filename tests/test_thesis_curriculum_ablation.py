from src.envs.thesis_cartesian_r6_curriculum_env import (
    CartesianR6BatteryConfig,
    CartesianR6BatteryEnv,
)


def test_direct_ablation_environment_is_battery_on_cartesian():
    env = CartesianR6BatteryEnv(CartesianR6BatteryConfig(seed=1101, horizon=10))
    out = env.reset(seed=1101)
    assert env.cfg.enable_battery_cycle is True
    assert env.n_actions == 10
    assert out["action_mask"].shape == (env.cfg.num_uavs, 10)


def test_direct_ablation_uses_fixed_residual_battery():
    env = CartesianR6BatteryEnv(
        CartesianR6BatteryConfig(
            seed=1101,
            horizon=10,
            residual_initial_battery=30.0,
            randomize_residual_uav=True,
        )
    )
    env.reset(seed=1101)
    idx = int(env.residual_uav_idx)
    assert float(env.battery[idx]) == 30.0
