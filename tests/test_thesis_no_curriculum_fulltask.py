import numpy as np

from src.envs.thesis_cartesian_r6_curriculum_env import (
    CartesianR6BatteryConfig,
    CartesianR6BatteryEnv,
)


def test_fulltask_ablation_is_cartesian_and_battery_on():
    env = CartesianR6BatteryEnv(
        CartesianR6BatteryConfig(
            seed=1101,
            horizon=10,
            residual_initial_battery=30.0,
            randomize_residual_uav=True,
        )
    )
    out = env.reset(seed=1101)
    assert env.cfg.enable_battery_cycle is True
    assert env.n_actions == 10
    assert out["action_mask"].shape == (env.cfg.num_uavs, 10)


def test_training_battery_randomization_can_be_applied_before_reset():
    env = CartesianR6BatteryEnv(
        CartesianR6BatteryConfig(
            seed=1101,
            horizon=10,
            residual_initial_battery=30.0,
            randomize_residual_uav=True,
        )
    )
    rng = np.random.default_rng(123)
    sampled = float(rng.uniform(24.0, 36.0))
    env.cfg.residual_initial_battery = sampled
    env.reset(seed=1101)

    idx = int(env.residual_uav_idx)
    assert 24.0 <= float(env.battery[idx]) <= 36.0
    assert np.isclose(float(env.battery[idx]), sampled, atol=1e-5)
