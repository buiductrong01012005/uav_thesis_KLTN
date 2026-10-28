"""The reference must observe hidden state without mutating the live env."""

import pickle
import numpy as np
from src.envs.bs5_service_v2 import BS5ServiceV2Env
from src.baselines.service_v2_privileged_search import PrivilegedPlacementSearch


def test_planning_does_not_mutate_live_state_or_rng():
    env = BS5ServiceV2Env(horizon=20)
    env.reset(seed=620001, assignment_seed=620001)
    before = pickle.dumps(env)
    policy = PrivilegedPlacementSearch(env, budget=30)
    actions = policy.act(env.get_action_mask())
    assert pickle.dumps(env) == before
    assert all(env.get_action_mask()[a, x] for a, x in enumerate(actions))
    assert 1 <= policy.calls <= 30


def test_reference_deterministic_and_execution_keeps_constraints():
    traces = []
    for _ in range(2):
        env = BS5ServiceV2Env(horizon=3)
        env.reset(seed=620002, assignment_seed=620002)
        policy = PrivilegedPlacementSearch(env, seed=731, period=2, budget=30)
        trace = []
        for slot in range(3):
            old = env.uav_positions.copy()
            actions = policy.act(env.get_action_mask())
            out = env.step(actions)
            assert np.all(np.abs(env.uav_positions - old).sum(axis=1) <= 1)
            assert env.last_v2_network["radio_flow_violation_mbps"] <= 1e-6
            assert env.last_v2_network["schedule_time_sum"] <= 1 + 1e-6
            trace.append((actions, out["info"]["utility"]))
        traces.append(trace)
    assert traces[0] == traces[1]
