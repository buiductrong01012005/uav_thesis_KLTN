"""Analytical controls for interference PSD, flow conservation and radio conflicts."""

import json
from pathlib import Path

import numpy as np
import pytest

from src.diagnostics.bs5_spectrum_airtime import (
    Edge,
    access_rates,
    allocate_flows,
    audit_case,
    fixture_env,
    schedule_matrix,
    spectral_rate,
    spectrum_colors,
    valid_half_duplex_state,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/bs5_spectrum_airtime_audit_v2.json").read_text())


def test_shannon_and_efficiency_analytic_control():
    assert spectral_rate(3, 0, 1, 1e6) == pytest.approx(2)
    assert spectral_rate(3, 2, 1, 1e6) == pytest.approx(1)
    assert spectral_rate(3, 0, 1, 1e6, 0.75) == pytest.approx(1.5)


def test_two_hop_half_duplex_is_harmonic_capacity_not_minimum():
    for first, second in [(100, 100), (100, 50), (50, 200)]:
        cap = np.array([[first, 0], [0, second]], dtype=float)
        flows, times, _, violation = allocate_flows({0: 1000}, {0: [0, 1]}, cap)
        assert flows[0] == pytest.approx(first * second / (first + second), abs=1e-6)
        assert sum(times) <= 1 + 1e-7
        assert violation < 1e-6


def test_shared_gateway_is_not_multiplied_per_client():
    cap = np.array([[100, 0], [0, 100]], dtype=float)
    flows, times, minimum, _ = allocate_flows({0: 100, 1: 100}, {0: [0], 1: [1]}, cap)
    assert sum(flows.values()) == pytest.approx(100, abs=1e-6)
    assert flows[0] == pytest.approx(50, abs=1e-6)
    assert flows[1] == pytest.approx(50, abs=1e-6)
    assert minimum == pytest.approx(0.5)
    assert sum(times) == pytest.approx(1)


def test_disjoint_links_can_reuse_time_with_explicit_simultaneous_state():
    cap = np.array([[100, 0, 100], [0, 100, 100]], dtype=float)
    flows, _, _, _ = allocate_flows({0: 100, 1: 100}, {0: [0], 1: [1]}, cap)
    assert sum(flows.values()) == pytest.approx(200, abs=1e-6)


def test_one_shared_relay_ingress_counts_both_remote_flows():
    cap = np.diag([100, 100, 100]).astype(float)
    flows, times, _, _ = allocate_flows({0: 100, 1: 100}, {0: [0, 1], 1: [0, 2]}, cap)
    assert sum(flows.values()) == pytest.approx(50, abs=1e-6)
    assert np.dot(cap[0], times) >= sum(flows.values()) - 1e-6


def test_disconnected_flow_cannot_be_served():
    flows, times, minimum, _ = allocate_flows({0: 50}, {0: []}, np.zeros((0, 1)))
    assert flows == {0: 0}
    assert not times.any()
    assert minimum == 0


def test_conflicts_include_bs_beam_and_relay_receive_transmit():
    edges = [
        Edge("b0", "u0", 100),
        Edge("b0", "u1", 100),
        Edge("u0", "u2", 100),
        Edge("b1", "u3", 100),
    ]
    assert not valid_half_duplex_state(edges, (0, 1))  # same donor radio
    assert not valid_half_duplex_state(edges, (0, 2))  # Relay cannot RX+TX
    assert valid_half_duplex_state(edges, (0, 3))


def test_sinr_psd_and_total_budget_not_five_free_channels():
    env = fixture_env("near_gateway", "5A", CONFIG)
    association = env.compute_network()["association"]
    reuse, _, shared = access_rates(env, association, 20, "reuse1", 1)
    ortho, _, dedicated = access_rates(env, association, 20, "orthogonal_per_uav", 1)
    pool, _, pooled = access_rates(env, association, 20, "orthogonal_pool", 1)
    assert np.all(reuse <= ortho + 1e-9)
    assert np.all(pool <= ortho + 1e-9)
    assert shared["access_reserved_fleet_mhz"] == pooled["access_reserved_fleet_mhz"] == 20
    assert dedicated["access_reserved_fleet_mhz"] == 100
    assert pooled["access_per_uav_mhz"] == 4


def test_no_interference_equals_snr_and_fixed_power_bandwidth_split():
    env = fixture_env("near_gateway", "5A", CONFIG)
    association = {0: list(range(25))}
    env.ue_positions[:25] = env.uav_positions[0]
    a, _, _ = access_rates(env, association, 5, "reuse1", 1)
    b, _, _ = access_rates(env, association, 5, "orthogonal_per_uav", 1)
    assert a == pytest.approx(b)
    assert a[:25].sum() == pytest.approx(79.17911303205, abs=1e-7)


def test_interference_is_not_cut_off_at_association_radius():
    env = fixture_env("distributed", "5A", CONFIG)
    env.uav_positions[0] = [6, 4]
    env.uav_positions[1] = [6, 9]  # 250m from UE0, outside Access radius
    env.ue_positions[0] = env.uav_positions[0]
    env.ue_positions[1] = env.uav_positions[1]
    # Use low enough bandwidth-normalized signal to avoid the 4Mbps cap masking interference.
    env.cfg.ue_demand_mbps = 1000
    rates, _, _ = access_rates(env, {0: [0], 1: [1]}, 5, "reuse1", 1)
    isolated, _, _ = access_rates(env, {0: [0], 1: []}, 5, "reuse1", 1)
    assert rates[0] < isolated[0]


def test_routes_schedule_and_all_rates_match_in_full_case():
    env = fixture_env("relay_corridor", "3A2R", CONFIG)
    metric, detail = audit_case(
        env,
        access_bw=15,
        backhaul_bw=40,
        spectrum_mode="reuse1",
        backhaul_mode="shared_half_duplex",
        efficiency=0.75,
    )
    assert metric["delivered_mbps"] == pytest.approx(sum(detail["ue_delivered_rates"]))
    assert metric["delivered_mbps"] <= metric["access_offer_mbps"] + 1e-6
    assert metric["schedule_time_sum"] <= 1 + 1e-6
    edges = [Edge(**e) for e in detail["edges"]]
    for state in detail["schedules"]:
        assert valid_half_duplex_state(edges, tuple(state["active_edges"]))
    assert metric["relay_routes"] > 0


def test_ue_fixture_matched_across_roles_and_does_not_mutate_default_config():
    a = fixture_env("distributed", "5A", CONFIG)
    b = fixture_env("distributed", "3A2R", CONFIG)
    assert np.array_equal(a.ue_positions, b.ue_positions)
    assert a.cfg.access_bw_hz == 5e6 and a.cfg.backhaul_bw_hz == 20e6


def test_simultaneous_bh_interference_cannot_increase_solo_capacity():
    env = fixture_env("near_gateway", "5A", CONFIG)
    edges = [Edge("b0", "u0", 1), Edge("b1", "u1", 1)]
    matrix, states = schedule_matrix(env, edges, 20, 1, "shared_half_duplex")
    both = states.index((0, 1))
    assert matrix[0, both] < matrix[0, states.index((0,))]
    assert matrix[1, both] < matrix[1, states.index((1,))]


def test_coloring_is_exact_for_complete_graph_and_reuses_distant_channels():
    env = fixture_env("near_gateway", "5A", CONFIG)
    colors = spectrum_colors(env, list(range(5)))
    assert len(set(colors.values())) == 5
    env.uav_positions[:2] = [[0, 0], [19, 19]]
    assert spectrum_colors(env, [0, 1]) == {0: 0, 1: 0}


def test_colored_scheme_counts_occupied_channels_and_no_false_zero_interference():
    env = fixture_env("near_gateway", "5A", CONFIG)
    association = env.compute_network()["association"]
    colored, _, budget = access_rates(env, association, 20, "colored_reuse", 1)
    orthogonal, _, _ = access_rates(env, association, 20, "orthogonal_per_uav", 1)
    assert colored == pytest.approx(orthogonal)
    assert budget["access_distinct_used_mhz"] == 100
    env.uav_positions[:2] = [[0, 0], [10, 0]]
    env.ue_positions[0] = env.uav_positions[0]
    env.ue_positions[1] = env.uav_positions[1]
    env.cfg.ue_demand_mbps = 1000
    colored, _, _ = access_rates(env, {0: [0], 1: [1]}, 20, "colored_reuse", 1)
    orthogonal, _, _ = access_rates(env, {0: [0], 1: [1]}, 20, "orthogonal_per_uav", 1)
    assert colored[0] < orthogonal[0]
