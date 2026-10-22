"""Regression tests for adding 5A to the shared role-search/recovery suite."""

from __future__ import annotations

import numpy as np

from scripts.evaluation.run_uav_continuous_eval import make_continuous_env
from src.baselines.recovery_aware_all_access_greedy import RecoveryAwareAllAccessGreedy
from src.baselines.recovery_aware_role_greedy import _ServiceRecoveryMixin
from src.baselines.thesis_exhaustive_cartesian_role_greedy import (
    ExhaustiveCartesianRoleGreedy,
)
from src.envs.v0_env import ACCESS
from src.envs.v4_fixed_mobile_battery_env import ACTIVE, CHARGING


def test_5a_reuses_exactly_the_same_movement_and_recovery_logic():
    assert issubclass(RecoveryAwareAllAccessGreedy, _ServiceRecoveryMixin)
    assert issubclass(RecoveryAwareAllAccessGreedy, ExhaustiveCartesianRoleGreedy)
    assert RecoveryAwareAllAccessGreedy._score is ExhaustiveCartesianRoleGreedy._score
    assert RecoveryAwareAllAccessGreedy._recovery_actions is _ServiceRecoveryMixin._recovery_actions
    assert RecoveryAwareAllAccessGreedy.observe_service is _ServiceRecoveryMixin.observe_service


def test_5a_is_one_role_vector_and_movement_search_is_action_mask_valid():
    e = make_continuous_env(3101, 500)
    out = e.reset(seed=3101)
    before = (
        e.uav_positions.copy(),
        e.roles.copy(),
        e.status.copy(),
        e.battery.copy(),
        e.ue_positions.copy(),
    )
    baseline = RecoveryAwareAllAccessGreedy(e)
    actions = baseline.act(out["action_mask"])
    after = (e.uav_positions, e.roles, e.status, e.battery, e.ue_positions)
    assert all(np.array_equal(a, b) for a, b in zip(before, after)), (
        "What-if search must not mutate the simulator"
    )
    assert baseline.last_stats["role_vectors_evaluated"] == 1
    assert baseline.last_stats["selected_relay_count"] == 0
    assert baseline.last_stats["selected_access_count"] == 5
    assert baseline.last_stats["effective_composition"] == "5A0R"
    assert all(out["action_mask"][i, a] for i, a in enumerate(actions))
    assert all(e.decode_action(int(a))[1] == ACCESS for a in actions)


def test_5a_when_some_uavs_are_unavailable_stays_all_access():
    e = make_continuous_env(3101, 500)
    e.reset(seed=3101)
    e.status[1] = CHARGING
    b = RecoveryAwareAllAccessGreedy(e)
    actions = b.act(e.get_action_mask())
    assert b.last_stats["selected_access_count"] == 4
    assert b.last_stats["selected_relay_count"] == 0
    assert b.last_stats["effective_composition"] == "4A0R"
    assert all(
        e.decode_action(int(actions[i]))[1] == ACCESS
        for i in range(e.cfg.num_uavs)
        if e.status[i] == ACTIVE
    )


def test_5a_recovery_trigger_and_nonteleport_movement():
    e = make_continuous_env(3101, 500)
    out = e.reset(seed=3101)
    b = RecoveryAwareAllAccessGreedy(e, zero_service_patience=3)
    for _ in range(3):
        b.observe_service(0.0)
    actions = b.act(out["action_mask"])
    assert b.last_stats["recovery_active"] is True
    assert len(b.last_stats["recovery_access_goals"]) >= 1
    assert all(bool(out["action_mask"][i, int(a)]) for i, a in enumerate(actions))
    before = e.uav_positions.copy()
    e.step(actions)
    assert np.max(np.abs(e.uav_positions - before)) <= 1
    b.observe_service(0.1)
    _ = b.act(e.get_action_mask())
    assert b.last_stats["recovery_active"] is False
