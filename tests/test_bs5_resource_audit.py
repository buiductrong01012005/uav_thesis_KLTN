"""Physical/accounting checks for the new offline audit, no long rollouts."""
import json
from pathlib import Path

import numpy as np
import pytest

from src.diagnostics.bs5_resource_audit import (
    access_raw_rate, backhaul_rates, gateway_design, hybrid_energy,
    hybrid_flow_budget, make_radio, minimum_access_bandwidth, shared_relay_scene, static_scene,
)

ROOT = Path(__file__).resolve().parents[1]
PROFILES = json.loads((ROOT / "configs/bs5_resource_audit_v1.json").read_text())["resource_profiles"]


def test_fixed_total_power_and_bandwidth_sharing_not_per_ue_subsidy():
    env = make_radio()
    for distance in [0, 100, 200]:
        total = access_raw_rate(env, distance, 1)
        assert 25 * access_raw_rate(env, distance, 25) == pytest.approx(total)
    assert access_raw_rate(env, 0, 25) < 4
    assert 3.97 < access_raw_rate(make_radio(PROFILES[1]), 200, 25) < 4
    assert access_raw_rate(make_radio(PROFILES[2]), 200, 25) > 4


def test_inverse_bandwidth_matches_target_at_fixed_power():
    env = make_radio()
    bw = minimum_access_bandwidth(env, distance_m=200, users=25, target_mbps=4)
    assert 10e6 < bw < 11e6
    env.cfg.access_bw_hz = bw
    assert access_raw_rate(env, 200, 25) == pytest.approx(4)
    env.cfg.access_tx_dbm = -100
    assert minimum_access_bandwidth(env, distance_m=200, users=25, target_mbps=4) is None


@pytest.mark.parametrize("access,gateway,users", [(0, 0, 25), (200, 300, 25), (201, 100, 25), (0, 0, 30)])
def test_real_router_conserves_demand_with_disjoint_loss_waterfall(access, gateway, users):
    row = static_scene(PROFILES[0], users, access, gateway)
    assert row["conservation_residual_mbps"] == pytest.approx(0, abs=1e-8)
    assert row["delivered_mbps"] == pytest.approx(users * 4 * row["utility"])
    assert row["associated_ues"] <= 25
    assert all(row[k] >= 0 for k in row if k.endswith("loss_mbps"))
    if access > 200:
        assert row["geometry_loss_mbps"] == users * 4
    if users > 25 and access == 0:
        assert row["association_cap_loss_mbps"] == (users - 25) * 4


def test_hard_cutoff_and_zero_guaranteed_minimum():
    at_edge = static_scene(PROFILES[2], 25, 0, 300)
    outside = static_scene(PROFILES[2], 25, 0, 301)
    assert at_edge["delivered_mbps"] > 0
    assert outside["delivered_mbps"] == 0
    assert outside["backhaul_loss_mbps"] == pytest.approx(100)
    assert outside["min_ue_mbps"] == 0
    assert backhaul_rates(make_radio(), 301)[0] > 0  # raw rate is not feasible rate


def test_access_upgrade_cannot_fix_a_backhaul_bottleneck():
    small = static_scene(PROFILES[0], 25, 0, 300)
    large = static_scene(PROFILES[2], 25, 0, 300)
    assert small["delivered_mbps"] == pytest.approx(large["delivered_mbps"])
    assert large["backhaul_loss_mbps"] > small["backhaul_loss_mbps"]


def test_actual_shared_relay_cannot_multiply_gateway_capacity():
    for profile in PROFILES:
        scene = shared_relay_scene(profile, 250)
        assert scene["relay_clients"] == 2
        assert scene["delivered_mbps"] <= scene["relay_gateway_mbps"] + 1e-8
        assert scene["allocated_relay_mbps"] <= scene["relay_gateway_mbps"] + 1e-8
        assert scene["conservation_residual_mbps"] == pytest.approx(0, abs=1e-8)


def test_target_specific_gateway_radius_and_power():
    env = make_radio()
    design = gateway_design(env, 100)
    distance = design["max_gateway_distance_for_target_m"]
    assert 100 < distance < 200
    assert backhaul_rates(env, distance)[0] == pytest.approx(100)
    env.cfg.backhaul_tx_dbm = design["required_gateway_tx_at_radius_dbm"]
    assert backhaul_rates(env, 300)[0] == pytest.approx(100)
    assert gateway_design(make_radio(), 1e6)["max_gateway_distance_for_target_m"] is None


def test_hybrid_airtime_flow_and_energy_no_free_forwarding():
    rows = hybrid_flow_budget(60, 40, 40, 40, half_duplex=True, fractions=np.linspace(0, 1, 11))
    for row in rows:
        assert row["receive_fraction"] + row["forward_fraction"] == pytest.approx(1)
        assert row["total_delivered_mbps"] <= row["gateway_allocated_mbps"] + 1e-9
        assert row["remote_delivered_mbps"] <= row["forward_allocated_mbps"] + 1e-9
        if row["mode"] == "Relay":
            assert row["local_delivered_mbps"] == 0
    env = make_radio()
    relay_cost = hybrid_energy(env, moving=False, hybrid=False)
    hybrid_cost = hybrid_energy(env, moving=False, hybrid=True)
    access_cost = env.cfg.hover_drain + env.cfg.access_radio_drain
    assert hybrid_cost > max(relay_cost, access_cost)
    assert hybrid_cost == pytest.approx(0.4)
    assert hybrid_energy(env, moving=True, hybrid=True) == pytest.approx(0.5)


def test_legacy_radio_configuration_is_not_changed_by_candidate_instances():
    before = make_radio().cfg
    make_radio(PROFILES[-1])
    after = make_radio().cfg
    assert before == after
    assert after.access_bw_hz == 5e6
    assert after.backhaul_tx_dbm == 20
