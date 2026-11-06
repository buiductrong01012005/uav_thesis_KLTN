"""Post-action scalar observer copied verbatim from the frozen training evaluator.
No true-state logging enters the actor.
"""
import numpy as np

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
    "tcf_score",
    "reward_throughput",
    "reward_coverage",
    "reward_fairness",
]

def observe_step(env, info):
    snap, net = env.last_service_snapshot, env.last_v2_network
    active = np.flatnonzero(snap["status"] == 0)
    access = [a for a in active if snap["roles"][a] in (0, 2)]
    relays = [a for a in active if snap["roles"][a] == 1]
    covered = np.zeros(env.cfg.num_ues, dtype=bool)
    for aid in access:
        covered |= (
            np.linalg.norm(snap["ue_positions"] - snap["uav_positions"][aid], axis=1)
            <= env.cfg.access_radius_cells + 1e-9
        )
    paths = env.last_v2_radio_detail["paths"]
    forwarding = {a for a, flow in net["forwarded_mbps_by_uav"].items() if flow > 1e-9}
    pure_forwarding = {a for a in forwarding if snap["roles"][a] == 1}
    assert net["schedule_time_sum"] <= 1 + 1e-6
    assert net["radio_flow_violation_mbps"] <= 1e-6
    return {
        **{k: float(info[k]) for k in PRIMARY_METRICS},
        "utility": float(info["utility"]),
        "covered_ues": float(covered.sum()),
        "associated_ues": float(sum(map(len, net["association"].values()))),
        "reachable_ues": float(
            sum(len(net["association"][a]) for a, path in paths.items() if path)
        ),
        "active_access": float(sum(snap["roles"][a] == 0 for a in active)),
        "active_relay": float(len(relays)),
        "active_hybrid": float(sum(snap["roles"][a] == 2 for a in active)),
        "forwarding_relay": float(len(pure_forwarding)),
        "forwarding_total": float(len(forwarding)),
        "forwarding_hybrid": float(sum(snap["roles"][a] == 2 for a in forwarding)),
        "own_hybrid_mbps": float(env.last_probe_result["metrics"]["own_hybrid_mbps"]),
        "hybrid_forwarded_mbps": float(env.last_probe_result["metrics"]["hybrid_forwarded_mbps"]),
        "qos_ues_original": float(info["qos_ues_original"]),
        "qos_fraction_original": float(info["qos_fraction_original"]),
        **env.last_energy_probe,
        "access_offer_mbps": float(net["access_raw_ue_rates"].sum()),
        "reachable_offer_mbps": float(
            sum(net["access_offer_mbps"][a] for a, path in paths.items() if path)
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
                snap["roles"][a] == 2
                and net["backhaul_caps_mbps"].get(a, 0) > 1e-6
                and net["forwarded_mbps_by_uav"].get(a, 0) > 1e-6
                for a in active
            )
        ),
    }
