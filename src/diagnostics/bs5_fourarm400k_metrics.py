"""Post-action observers only; no diagnostic enters the actor."""

import numpy as np
from src.diagnostics.bs5_episode_service_metrics import longest_true

PRIMARY_METRICS = [
    "throughput_mbps",
    "throughput_norm",
    "service_coverage_fraction",
    "service_coverage_ues",
    "positive_fraction",
    "positive_ues",
    "qos_fraction",
    "qos_ues",
    "ue_outage_fraction",
    "jain_raw",
    "jain_service_qualified",
    "service_score",
    "reward_throughput",
    "reward_coverage",
    "reward_fairness",
    "fairness_used",
    "jain_ema",
    "jain_instant_qualified",
]


def _observe_frozen(env, info):
    snap, net = (env.last_service_snapshot, env.last_v2_network)
    active = np.flatnonzero(snap["status"] == 0)
    access = [a for a in active if snap["roles"][a] in (0, 2)]
    relays = [a for a in active if snap["roles"][a] == 1]
    covered = np.zeros(env.cfg.num_ues, dtype=bool)
    for aid in access:
        covered |= (
            np.linalg.norm(snap["ue_positions"] - snap["uav_positions"][aid], axis=1)
            <= env.cfg.access_radius_cells + 1e-09
        )
    paths = env.last_v2_radio_detail["paths"]
    forwarding = {a for a, flow in net["forwarded_mbps_by_uav"].items() if flow > 1e-09}
    pure_forwarding = {a for a in forwarding if snap["roles"][a] == 1}
    assert net["schedule_time_sum"] <= 1 + 1e-06
    assert net["radio_flow_violation_mbps"] <= 1e-06
    return {
        **{k: float(info[k]) for k in PRIMARY_METRICS},
        "utility": float(info["utility"]),
        "covered_ues": float(covered.sum()),
        "associated_ues": float(sum(map(len, net["association"].values()))),
        "reachable_ues": float(
            sum((len(net["association"][a]) for a, path in paths.items() if path))
        ),
        "active_access": float(sum((snap["roles"][a] == 0 for a in active))),
        "active_relay": float(len(relays)),
        "active_hybrid": float(sum((snap["roles"][a] == 2 for a in active))),
        "forwarding_relay": float(len(pure_forwarding)),
        "forwarding_total": float(len(forwarding)),
        "forwarding_hybrid": float(sum((snap["roles"][a] == 2 for a in forwarding))),
        "own_hybrid_mbps": float(env.last_probe_result["metrics"]["own_hybrid_mbps"]),
        "hybrid_forwarded_mbps": float(env.last_probe_result["metrics"]["hybrid_forwarded_mbps"]),
        "qos_ues_original": float(info["qos_ues_original"]),
        "qos_fraction_original": float(info["qos_fraction_original"]),
        **env.last_energy_probe,
        "access_offer_mbps": float(net["access_raw_ue_rates"].sum()),
        "reachable_offer_mbps": float(
            sum((net["access_offer_mbps"][a] for a, path in paths.items() if path))
        ),
        "schedule_airtime": float(net["schedule_time_sum"]),
        "zero_service_slot_fraction": float(info["positive_ues"] == 0),
        "returning_uavs": float(np.sum(snap["status"] == 1)),
        "swapping_service_snapshot_uavs": float(np.sum(snap["status"] == 2)),
        "stranded_slot_fraction": float(bool(env.last_return_control["stranded_ids_post"])),
        "flow_violation_mbps": float(net["radio_flow_violation_mbps"]),
        **{
            key: float(env.last_probe_result["metrics"][key])
            for key in [
                "mean_wireless_hops_delivered",
                "mean_relay_hops_delivered",
                "max_wireless_hops_used",
                "multihop_beyond_one_relay_mbps",
                "routed_via_relay_mbps",
                *[f"delivered_mbps_hops_{i}" for i in range(1, 6)],
            ]
        },
        "hybrid_both_count": float(
            sum(
                (
                    snap["roles"][a] == 2
                    and net["backhaul_caps_mbps"].get(a, 0) > 1e-06
                    and (net["forwarded_mbps_by_uav"].get(a, 0) > 1e-06)
                    for a in active
                )
            )
        ),
    }


def observe_step(env, info):
    row = _observe_frozen(env, info)
    effective = np.asarray(env.last_energy_audit["effective_status"])
    row.update(
        effective_active_fraction=float(np.mean(effective == 0)),
        effective_return_fraction=float(np.mean(effective == 1)),
        effective_swap_fraction=float(np.mean(effective == 2)),
        effective_inactive_fraction=float(np.mean(effective == 3)),
        effective_unavailable_fraction=float(np.mean(np.isin(effective, [1, 2]))),
        role_switches=float(info["role_switches_this_slot"]),
    )
    return row


def rolling_unique(mask, width):
    if len(mask) < width:
        return None
    cumulative = np.vstack([np.zeros((1, mask.shape[1]), int), np.cumsum(mask, axis=0)])
    return float(((cumulative[width:] - cumulative[:-width]) > 0).sum(axis=1).mean())


def validation_metrics(rates, samples, energy, env, collision=None):
    rates = np.maximum(np.asarray(rates), 0.0)
    qualified = rates >= env.coverage_floor_mbps - 1e-9
    mean = rates.mean(0)
    arrays = {k: np.asarray([e[k] for e in energy]) for k in energy[0]}
    arrivals = [
        float(
            arrays["energy_after_j"][e["slot"] - 1, e["agent"]] / env.energy_config.capacity_j * 100
        )
        for e in env.energy_events
        if e["event"] == "swap_start"
    ]
    zero = np.asarray([s["throughput_mbps"] <= 1e-9 for s in samples])
    return dict(
        rolling_unique_ge1_1min=rolling_unique(qualified, 12),
        rolling_unique_ge1_5min=rolling_unique(qualified, 60),
        longest_below1_gap_slots_max=max(longest_true(~qualified[:, u]) for u in range(90)),
        min_ue_mean_rate_mbps=float(mean.min()),
        median_ue_mean_rate_mbps=float(np.median(mean)),
        longest_zero_network_gap_slots=longest_true(zero),
        arrival_events=len(arrivals),
        arrival_below20_events=sum(soc < 20 - 1e-7 for soc in arrivals),
        min_arrival_soc_percent=min(arrivals) if arrivals else None,
        longest_blocked_run_slots=max(
            longest_true(np.asarray([x["active_blocked"][a] for x in collision], bool))
            for a in range(5)
        )
        if collision
        else None,
        energy_balance_max_abs_j=float(np.abs(arrays["balance_residual_j"]).max()),
    )


def training_metrics(samples):
    joules = sum(s["consumed_j"] for s in samples)
    mbit = sum(s["throughput_mbps"] * 5 for s in samples)
    requests = sum(s["policy_move_requests"] for s in samples)
    return dict(
        joules_per_mbit=joules / mbit if mbit > 0 else None,
        blocked_per_policy_move=sum(s["active_blocked_uavs"] for s in samples) / requests
        if requests
        else None,
    )
