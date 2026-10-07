import numpy as np
import pytest

from src.envs.thesis_cartesian_r6_curriculum_env import CartesianR6BatteryConfig
from src.envs.thesis_cartesian_r4_relay_stress_env import (
    CartesianR4RelayStressBatteryConfig,
    CartesianR4RelayStressBatteryEnv,
)
from scripts.audit_relay_stress_feasibility import geometry_masks


def test_r4_is_separate_from_locked_r6():
    assert CartesianR6BatteryConfig().backhaul_radius_cells == pytest.approx(6.0)
    assert CartesianR4RelayStressBatteryConfig().backhaul_radius_cells == pytest.approx(4.0)


def test_r4_keeps_action_and_access_geometry():
    env = CartesianR4RelayStressBatteryEnv(CartesianR4RelayStressBatteryConfig(seed=3101, horizon=5))
    out = env.reset(seed=3101)
    assert env.n_actions == 10
    assert env.cfg.access_radius_cells == pytest.approx(4.0)
    assert env.cfg.cell_size_m == pytest.approx(50.0)
    assert env.cfg.backhaul_radius_cells * env.cfg.cell_size_m == pytest.approx(200.0)
    assert out['obs'].shape[0] == env.cfg.num_uavs


def test_relay_required_geometry_exists_in_r4_seed3101():
    env = CartesianR4RelayStressBatteryEnv(CartesianR4RelayStressBatteryConfig(seed=3101, horizon=5))
    env.reset(seed=3101)
    required, feasible = geometry_masks(env, env.ue_positions)
    assert required.shape == (env.cfg.num_ues,)
    assert feasible.shape == required.shape
    # The purpose of R4 is to create a real relay-requiring subset, unlike R6.
    assert int(required.sum()) > 0
    assert int((required & feasible).sum()) > 0
