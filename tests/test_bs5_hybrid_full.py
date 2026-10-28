import numpy as np
import pytest

from src.diagnostics.bs5_hybrid_full import HybridFullConfig, score_hybrid_full
from src.diagnostics.bs5_hybrid_fdm import HybridFDMConfig, edge_budget, score_hybrid_fdm
from src.diagnostics.bs5_hybrid_lifecycle import HybridLifecycleProbe
from src.envs.bs5_tcf_env import BS5TCFEnv
from tests.test_bs5_hybrid_fdm import fixture


def test_full_endpoint_budget_is_legacy_pool_not_another_pool():
    env = fixture()
    config = HybridFullConfig()
    roles = [2, 0, 0, 0, 0]
    assert edge_budget(env, "b0", "u0", roles, config) == pytest.approx((90e6, 100))
    assert edge_budget(env, "u0", "u1", roles, config) == pytest.approx((90e6, 100))
    full = score_hybrid_full(env, roles)
    assert full["power_bounds"][0]["combined_tx_bound_mw"] == pytest.approx(10**2.7 + 100)
    assert full["metrics"]["flow_violation_mbps"] <= 1e-6


def test_full_mode_cannot_silently_be_used_as_a_split():
    with pytest.raises(ValueError):
        HybridFullConfig(access_fraction=0.5)


def test_full_can_keep_relay_service_and_add_local_when_gateway_has_capacity():
    env = fixture()
    relay = score_hybrid_fdm(env, env.roles)
    split = score_hybrid_fdm(env, [2, 0, 0, 0, 0], HybridFDMConfig())
    full = score_hybrid_full(env, [2, 0, 0, 0, 0])
    assert full["metrics"]["own_hybrid_mbps"] > 0
    assert full["metrics"]["hybrid_forwarded_mbps"] > 0
    assert full["metrics"]["throughput_mbps"] > relay["metrics"]["throughput_mbps"]
    assert full["metrics"]["throughput_mbps"] > split["metrics"]["throughput_mbps"]


@pytest.mark.parametrize("mode,expected", [("split", 0.25), ("full", 0.4)])
def test_active_energy_debit_includes_hybrid_radio(mode, expected):
    env = HybridLifecycleProbe(mode=mode)
    env.reset(seed=910801, assignment_seed=910801)
    env.battery[:] = 100
    before = env.battery.copy()
    out = env.step([9, 8, 8, 8, 8], hybrid_ids=[0])
    assert before[0] - env.battery[0] == pytest.approx(expected, abs=1e-5)
    assert before[1] - env.battery[1] == pytest.approx(0.3, abs=1e-5)
    assert out["diagnostic_only"] and "obs" not in out and "state" not in out


def test_no_hybrid_probe_matches_old_lifecycle_radio_and_random_stream():
    old, new = BS5TCFEnv(), HybridLifecycleProbe()
    old.reset(seed=910801, assignment_seed=910801)
    new.reset(seed=910801, assignment_seed=910801)
    for _ in range(16):
        old.step([9, 9, 8, 8, 8])
        new.step([9, 9, 8, 8, 8])
        for name in ["status", "roles", "battery", "uav_positions", "ue_positions"]:
            np.testing.assert_array_equal(getattr(old, name), getattr(new, name))
        np.testing.assert_allclose(
            old.last_v2_network["ue_rate_mbps"],
            new.last_v2_network["ue_rate_mbps"],
            atol=1e-8,
            rtol=0,
        )


def test_insufficient_energy_forces_return_without_double_debit_or_service():
    env = HybridLifecycleProbe()
    env.reset(seed=910801, assignment_seed=910801)
    env.battery[0] = 0.3
    env.step([9, 8, 8, 8, 8], hybrid_ids=[0])
    assert 0 in env.last_return_control["forced_return_ids"]
    assert env.hybrid_extra_debit[0] == 0
    assert env.battery[0] == pytest.approx(0.1, abs=1e-6)
    assert 0 not in env.last_probe_result["association"]


def test_charging_does_not_pay_hybrid_energy_and_rejoins_with_legacy_hysteresis():
    env = HybridLifecycleProbe()
    env.reset(seed=910801, assignment_seed=910801)
    env.status[0] = 2
    env.uav_positions[0] = env.charging_stations[0]
    env.battery[0] = 76
    env.step([8, 8, 8, 8, 8], hybrid_ids=[0])
    assert env.status[0] == 0 and env.battery[0] == 81
    assert env.hybrid_extra_debit[0] == 0


def test_additional_dual_chain_overhead_is_not_free():
    a, b = HybridLifecycleProbe(), HybridLifecycleProbe(hybrid_radio_overhead=0.1)
    for env in [a, b]:
        env.reset(seed=910801, assignment_seed=910801)
        env.battery[:] = 100
        env.step([9, 8, 8, 8, 8], hybrid_ids=[0])
    assert a.battery[0] - b.battery[0] == pytest.approx(0.1, abs=1e-5)
