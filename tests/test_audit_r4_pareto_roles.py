import numpy as np

from scripts.analysis.audit_r4_pareto_roles import (
    enumerate_role_pareto,
    is_dominant,
    measures,
    pareto_indices,
)
from scripts.evaluation.eval_relay_stress_r4_qmix import make_env, direct_access_cells
from src.envs.v0_env import ACCESS, RELAY


def test_pareto_frontier_and_dominance():
    rows = [
        {"utility": 0.1, "req_score": 0.3},
        {"utility": 0.2, "req_score": 0.4},
        {"utility": 0.5, "req_score": 0.1},
        {"utility": 0.2, "req_score": 0.4},
    ]
    assert is_dominant(rows[1], rows[0])
    assert not is_dominant(rows[1], rows[2])
    assert pareto_indices(rows) == [1, 2]


def test_req_zero_ue_handled():
    net = {"utility": 0.5, "ue_rate_mbps": np.array([1.0, 4.0])}
    res = measures(net, np.array([False, False]), 4.0)
    assert res["req_score"] == 0 and res["req_total"] == 0


def test_pareto_is_read_only_and_contains_original():
    env = make_env(3106, 4)
    env.reset(seed=3106)
    env.roles[:] = [ACCESS, RELAY, ACCESS, RELAY, ACCESS]
    env.last_network = None
    env.compute_network()
    snap = env.last_network
    originals = {
        name: getattr(env, name).copy()
        for name in ("uav_positions", "ue_positions", "roles", "status")
    }
    ret, front = enumerate_role_pareto(env, snap, direct_access_cells(env), 0.02)
    assert ret["n_role_candidates"] == 2**5
    assert ret["best_role_utility"] + 1e-8 >= ret["current_utility"]
    assert ret["max_req_score"] + 1e-8 >= ret["current_req_score"]
    assert ret["best_req_with_utility_slack"] + 1e-8 >= ret["current_req_score"]
    assert len(front) > 0
    assert not any(is_dominant(f, g) for f in front for g in front if f is not g)
    for name, value in originals.items():
        assert np.array_equal(getattr(env, name), value), name
    assert np.isclose(env.last_network["utility"], snap["utility"])


def test_original_when_all_access():
    env = make_env(3102, 3)
    env.reset(seed=3102)
    env.roles[:] = ACCESS
    env.compute_network()
    ret, front = enumerate_role_pareto(env, env.last_network, direct_access_cells(env))
    assert ret["current_nrelay"] == 0
    assert ret["n_role_candidates"] == 32


def test_pareto_audit_does_not_mutate_future_trajectory():
    a = make_env(3117, 7)
    b = make_env(3117, 7)
    a.reset(seed=3117)
    b.reset(seed=3117)
    grid = direct_access_cells(a)
    for i in range(5):
        flags = [a.encode_action(4, ACCESS)] * a.cfg.num_uavs
        ta = a.step(flags)
        tb = b.step(flags)
        assert np.isclose(ta["reward"], tb["reward"])
        assert np.array_equal(a.ue_positions, b.ue_positions)
        assert np.array_equal(a.uav_positions, b.uav_positions)
        assert np.allclose(a.battery, b.battery)
        assert np.array_equal(a.roles, b.roles)
        assert np.array_equal(a.status, b.status)
        if i % 2 == 0:
            enumerate_role_pareto(a, a.last_network, grid)
            assert np.array_equal(a.ue_positions, b.ue_positions)
            assert np.array_equal(a.roles, b.roles)
            assert np.array_equal(a.status, b.status)
