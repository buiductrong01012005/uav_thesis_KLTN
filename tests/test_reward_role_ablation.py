"""Simple reward-shaping ablation sanity checks; no training required."""
import numpy as np
import pytest

from scripts.run_thesis_no_curriculum_fulltask import shaped_training_reward
from src.envs.thesis_cartesian_r6_curriculum_env import (
    CartesianR6BatteryConfig, CartesianR6BatteryEnv,
)


def calc(U, before, after, n, e=0., r=0.):
    return shaped_training_reward(U, before, after, n, 5, 0.4, e, r)


def test_original_reward_unchanged():
    result, energy, switches = calc(.60, [100]*5, [99.7]*5, 3)
    assert result == pytest.approx(.60)
    assert energy == pytest.approx(.75)
    assert switches == pytest.approx(.6)


def test_energy_penalty_and_role_bonus():
    result, energy, switches = calc(.6, [100]*5, [99.7]*5, 2, .05, .02)
    assert result == pytest.approx(.6 - .05*.75 + .02*(2/5))


def test_charging_is_not_negative_energy():
    result, energy, switches = calc(.4, [100,10,100,100,100], [99.8,15,99.8,99.8,99.8], 0, .05, .02)
    assert energy == pytest.approx(.8/2.0)
    assert result == pytest.approx(.4-.05*.4)


def test_role_bonus_only_counts_active_intent_not_rejoin():
    env = CartesianR6BatteryEnv(CartesianR6BatteryConfig(seed=3101, horizon=4))
    obs = env.reset(seed=3101)
    before_roles = env.roles.copy()
    masks = obs['action_mask']
    actions = [int(np.flatnonzero(mask)[0]) for mask in masks]
    # In action encoding (move*2+role), parity one signals Relay.
    for i, mask in enumerate(masks):
        if env.status[i] == 0:
            actions[i] = int(np.flatnonzero(mask & (np.arange(10)%2 == 1))[0])
            break
    out = env.step(actions)
    assert out['info']['role_switches_this_slot'] >= 1
    before = np.array([100.,50.,100.,100.,100.])
    after = np.array([99.7,49.7,99.7,99.7,99.7])
    reward, _, switched = calc(out['reward'], before, after, out['info']['role_switches_this_slot'], r=.02)
    assert reward > out['reward']
    assert switched > 0


def test_invalid_count_is_rejected():
    with pytest.raises(ValueError):
        calc(.2,[100]*5,[99]*5,6)


def test_spatial_penalty_zero_at_or_beyond_100m():
    from scripts.run_thesis_no_curriculum_fulltask import active_pairwise_spatial_metrics
    # Grid cells are 50 m apart. Two-cell spacing = 100 m, so no penalty.
    pos = np.array([[0, 0], [0, 2]], dtype=float)
    penalty, mean_dist, close_frac = active_pairwise_spatial_metrics(pos, [1, 1], 50.0, 100.0)
    assert penalty == pytest.approx(0.0)
    assert mean_dist == pytest.approx(100.0)
    assert close_frac == pytest.approx(0.0)


def test_spatial_penalty_for_one_cell_spacing():
    from scripts.run_thesis_no_curriculum_fulltask import active_pairwise_spatial_metrics
    pos = np.array([[0, 0], [0, 1]], dtype=float)
    penalty, mean_dist, close_frac = active_pairwise_spatial_metrics(pos, [1, 1], 50.0, 100.0)
    assert penalty == pytest.approx(0.5)
    assert mean_dist == pytest.approx(50.0)
    assert close_frac == pytest.approx(1.0)


def test_spatial_penalty_excludes_nonactive_uavs():
    from scripts.run_thesis_no_curriculum_fulltask import active_pairwise_spatial_metrics
    # Third UAV is co-located but non-active; it must not create a penalty.
    pos = np.array([[0, 0], [0, 2], [0, 0]], dtype=float)
    penalty, mean_dist, close_frac = active_pairwise_spatial_metrics(pos, [1, 1, 0], 50.0, 100.0)
    assert penalty == pytest.approx(0.0)
    assert mean_dist == pytest.approx(100.0)
    assert close_frac == pytest.approx(0.0)
