"""Static audit of the actual BS5 router plus explicitly labelled link abstractions.

No env.step(), training, test-seed rollout, policy selection or legacy mutation.
Synthetic co-located UEs isolate load effects; they are not a mobility scenario.
Hybrid is a two-node link-budget experiment, not a trained Hybrid environment.
"""

from __future__ import annotations

from dataclasses import asdict
import math
from typing import Iterable

import numpy as np

from src.envs.bs5_launch_ab_env import BS5LaunchConfig, BS5LaunchEnv
from src.envs.v0_env import ACCESS, RELAY
from src.envs.v4_fixed_mobile_battery_env import ACTIVE


def make_radio(profile: dict | None = None, *, num_ues: int = 25) -> BS5LaunchEnv:
    """Use the same config/radio/router inherited by Return-v4.

    Resource changes live only in new diagnostic instances. No reset is needed
    for fully specified static fixtures, and no UE trajectory is consumed.
    """
    profile = profile or {}
    allowed = {"access_bw_hz", "backhaul_bw_hz", "backhaul_tx_dbm"}
    overrides = {key: profile[key] for key in allowed if key in profile}
    if any(overrides.get(k, 1) <= 0 for k in ("access_bw_hz", "backhaul_bw_hz")):
        raise ValueError("Bandwidth must be positive")
    return BS5LaunchEnv(BS5LaunchConfig(num_ues=num_ues, **overrides))


def access_raw_rate(env, distance_m: float, users: int) -> float:
    if users < 1 or distance_m < 0:
        raise ValueError("Positive UE count and nonnegative distance required")
    cfg = env.cfg
    loss = env._a2g_path_loss_db(
        [distance_m, 0], cfg.ue_height_m, [0, 0], cfg.uav_altitude_m, cfg.access_fc_hz
    )
    return env._shannon_rate_mbps(
        cfg.access_tx_dbm - 10 * math.log10(users) - loss, cfg.access_bw_hz / users
    )


def backhaul_rates(env, distance_m: float) -> tuple[float, float]:
    if distance_m < 0:
        raise ValueError("Distance must be nonnegative")
    cfg = env.cfg
    return (
        env._bh_a2g_rate([0, 0], cfg.uav_altitude_m, [distance_m, 0], cfg.bs_height_m),
        env._bh_a2a_rate([0, 0], [distance_m, 0]),
    )


def minimum_access_bandwidth(env, *, distance_m: float, users: int, target_mbps: float):
    """Inverse of current Shannon model at fixed *total* transmit power.

    Expanding bandwidth also expands noise. Return None if no finite bandwidth
    can meet the target; do not silently assume linear capacity scaling.
    """
    if users < 1 or target_mbps <= 0 or distance_m < 0:
        raise ValueError("Invalid design target")
    cfg = env.cfg
    loss = env._a2g_path_loss_db(
        [distance_m, 0], cfg.ue_height_m, [0, 0], cfg.uav_altitude_m, cfg.access_fc_hz
    )
    rx = cfg.access_tx_dbm - 10 * math.log10(users) - loss
    # Received power divided by noise spectral density, in Hz.
    power_over_n0 = 10 ** ((rx - cfg.noise_psd_dbm_hz - cfg.noise_figure_db) / 10)
    limit_mbps = power_over_n0 / math.log(2) / 1e6
    if target_mbps >= limit_mbps:
        return None
    lo, hi = 1.0, cfg.access_bw_hz / users
    while env._shannon_rate_mbps(rx, hi) < target_mbps:
        hi *= 2
    for _ in range(80):
        mid = (lo + hi) / 2
        if env._shannon_rate_mbps(rx, mid) < target_mbps:
            lo = mid
        else:
            hi = mid
    return hi * users


def gateway_design(env, target_mbps: float) -> dict:
    """Solve a target-specific radius, not a universal radio coverage claim."""
    if target_mbps <= 0:
        raise ValueError("Target must be positive")
    radius = env.cfg.backhaul_radius_cells * env.cfg.cell_size_m
    at_zero = backhaul_rates(env, 0)[0]
    at_edge = backhaul_rates(env, radius)[0]
    if at_zero < target_mbps:
        max_distance = None
    elif at_edge >= target_mbps:
        max_distance = radius
    else:
        lo, hi = 0.0, radius
        for _ in range(80):
            mid = (lo + hi) / 2
            if backhaul_rates(env, mid)[0] >= target_mbps:
                lo = mid
            else:
                hi = mid
        max_distance = lo
    exponent = target_mbps * 1e6 / env.cfg.backhaul_bw_hz * math.log(2)
    log_snr_required = exponent if exponent > 50 else math.log(math.expm1(exponent))
    snr_current = math.expm1(at_edge * 1e6 / env.cfg.backhaul_bw_hz * math.log(2))
    required_tx = env.cfg.backhaul_tx_dbm + 10 / math.log(10) * (
        log_snr_required - math.log(snr_current)
    )
    watts_exponent = (required_tx - 30) / 10
    return {
        "target_mbps": target_mbps,
        "gateway_rate_at_radius_mbps": at_edge,
        "max_gateway_distance_for_target_m": max_distance,
        "required_gateway_tx_at_radius_dbm": required_tx,
        "required_gateway_tx_at_radius_w": 10**watts_exponent if watts_exponent < 300 else None,
        "margin_db": 0,
        "note": "Model-only Shannon target; no interference/fading margin or hardware validation",
    }


def loss_waterfall(env, net: dict) -> dict:
    """Exact demand decomposition for one static state under current allocation.

    Loss order is geometry -> association cap -> Access rate -> backhaul rate.
    This is accounting under the configured router, not causal attribution or
    an oracle upper bound over alternative associations/positions/schedules.
    """
    demand = env.cfg.ue_demand_mbps
    active_access = np.flatnonzero((env.status == ACTIVE) & (env.roles == ACCESS))
    feasible = np.zeros(env.cfg.num_ues, dtype=bool)
    for aid in active_access:
        feasible |= np.linalg.norm(env.ue_positions - env.uav_positions[aid], axis=1) <= (
            env.cfg.access_radius_cells + 1e-9
        )
    associated = {u for users in net["association"].values() for u in users}
    offered = sum(net["access_offer_mbps"].values())
    delivered = float(np.sum(net["ue_rate_mbps"]))
    losses = {
        "demand_mbps": env.cfg.num_ues * demand,
        "geometry_loss_mbps": int((~feasible).sum()) * demand,
        "association_cap_loss_mbps": (int(feasible.sum()) - len(associated)) * demand,
        "access_radio_loss_mbps": max(0.0, len(associated) * demand - offered),
        "backhaul_loss_mbps": max(0.0, offered - delivered),
        "delivered_mbps": delivered,
        "geometrically_covered_ues": int(feasible.sum()),
        "associated_ues": len(associated),
    }
    losses["conservation_residual_mbps"] = losses["demand_mbps"] - sum(
        losses[k]
        for k in (
            "geometry_loss_mbps",
            "association_cap_loss_mbps",
            "access_radio_loss_mbps",
            "backhaul_loss_mbps",
            "delivered_mbps",
        )
    )
    if abs(losses["conservation_residual_mbps"]) > 1e-7:
        raise AssertionError("Demand loss accounting does not conserve flow")
    return losses


def static_scene(profile: dict, users: int, access_m: float, gateway_m: float) -> dict:
    if min(access_m, gateway_m) < 0 or users < 1:
        raise ValueError("Invalid static scene")
    env = make_radio(profile, num_ues=users)
    # Position all specified synthetic UEs on one side, gateway on the other.
    # All fixture coordinates remain within the 1 km world at tested distances.
    env.status[:] = 3  # existing CurriculumInactive status; no ghost UAVs
    env.status[0] = ACTIVE
    env.roles[:] = ACCESS
    env.uav_positions[0] = [10, 10]
    env.ue_positions[:] = env.uav_positions[0] + [0, access_m / env.cfg.cell_size_m]
    xy = env._cell_to_xy(env.uav_positions[0])
    env.surviving_bs_xy = np.asarray([xy + [0, -gateway_m]])
    net = env.compute_network()
    rates = net["ue_rate_mbps"]
    return dict(
        profile=profile["name"],
        fixture="synthetic_colocated_ues_one_access_one_gateway",
        users=users,
        access_distance_m=access_m,
        gateway_distance_m=gateway_m,
        access_bandwidth_mhz=env.cfg.access_bw_hz / 1e6,
        gateway_bandwidth_mhz=env.cfg.backhaul_bw_hz / 1e6,
        gateway_tx_dbm=env.cfg.backhaul_tx_dbm,
        utility=net["utility"],
        positive_service_ues=int(np.sum(rates > 1e-9)),
        qos_ues=int(np.sum(rates >= env.cfg.ue_demand_mbps - 1e-9)),
        min_ue_mbps=float(np.min(rates)),
        mean_ue_mbps=float(np.mean(rates)),
        **loss_waterfall(env, net),
    )


def shared_relay_scene(profile: dict, gateway_m: float) -> dict:
    """Two disjoint groups, two Access UAVs, one shared Relay, actual BS5 router."""
    env = make_radio(profile, num_ues=50)
    env.status[:] = 3
    env.status[:3] = ACTIVE
    env.roles[:] = ACCESS
    env.roles[2] = RELAY
    env.uav_positions[:3] = [[6, 6], [6, 14], [6, 10]]
    env.ue_positions[:25] = env.uav_positions[0]
    env.ue_positions[25:] = env.uav_positions[1]
    hub_xy = env._cell_to_xy(env.uav_positions[2])
    env.surviving_bs_xy = np.asarray([hub_xy + [0, gateway_m]])
    net = env.compute_network()
    selected = [a for a, r in net["routes"].items() if r == 2]
    relay_rate = net["relay_gateway_caps_mbps"][2]
    allocated = sum(net["backhaul_caps_mbps"][a] for a in selected)
    if allocated > relay_rate + 1e-7:
        raise AssertionError("Shared Relay allocates more than its gateway capacity")
    return dict(
        profile=profile["name"],
        fixture="synthetic_two_access_shared_relay_50_ues",
        relay_gateway_distance_m=gateway_m,
        access_relay_distance_m=200,
        relay_clients=len(selected),
        relay_gateway_mbps=relay_rate,
        allocated_relay_mbps=allocated,
        direct_fallback_rejections=len(net["admission_rejected_access_ids"]),
        qos_ues=int(np.sum(net["ue_rate_mbps"] >= 4 - 1e-9)),
        **loss_waterfall(env, net),
    )


def hybrid_flow_budget(
    gateway_mbps: float,
    forward_mbps: float,
    local_offer: float,
    remote_offer: float,
    *,
    half_duplex: bool,
    fractions: Iterable[float],
):
    """2-UAV abstraction: gateway -> hub -> remote Access + hub's local UEs.

    Each UAV has its own fixed Access radio. Hybrid activates hub's Access radio
    as well as its backhaul radio. Full radio costs are charged when enabled.
    Half-duplex backhaul: hub receive airtime + forward airtime <= 1.
    Separate access/backhaul bands may operate concurrently (explicit assumption).
    Local-first allocation is deterministic; alternatives need a scheduler study.
    """
    if min(gateway_mbps, forward_mbps, local_offer, remote_offer) < 0:
        raise ValueError("Negative capacity or offer")
    fractions = tuple(fractions)
    rows = []
    for hybrid in (False, True):
        for fraction in fractions:
            if not 0 <= fraction <= 1:
                raise ValueError("Airtime fraction outside [0,1]")
            receive = fraction if half_duplex else 1.0
            forward = 1 - fraction if half_duplex else 1.0
            incoming = receive * gateway_mbps
            local = min(local_offer if hybrid else 0, incoming)
            remote = min(remote_offer, forward * forward_mbps, max(0, incoming - local))
            rows.append(
                dict(
                    mode="Hybrid" if hybrid else "Relay",
                    duplex="HD" if half_duplex else "ideal_parallel",
                    receive_fraction=receive,
                    forward_fraction=forward,
                    local_delivered_mbps=local,
                    remote_delivered_mbps=remote,
                    total_delivered_mbps=local + remote,
                    gateway_allocated_mbps=incoming,
                    forward_allocated_mbps=forward * forward_mbps,
                    receive_load_mbps=local + remote,
                    forward_load_mbps=remote,
                )
            )
    return rows


def hybrid_energy(env, *, moving: bool, hybrid: bool) -> float:
    """Provisional normalized model, not Joules and not a battery-cycle rollout."""
    cfg = env.cfg
    return (
        cfg.hover_drain
        + cfg.move_extra_drain * moving
        + cfg.relay_radio_drain
        + (cfg.access_radio_drain if hybrid else 0)
    )


def run_audit(config: dict) -> dict:
    rows, links, designs, hybrid_rows, relay_rows = [], [], [], [], []
    profiles = config["resource_profiles"]
    for profile in profiles:
        env = make_radio(profile)
        for distance in config["access_distances_m"]:
            links.append(
                dict(
                    profile=profile["name"],
                    link="Access",
                    distance_m=distance,
                    raw_capacity_mbps=access_raw_rate(env, distance, 25) * 25,
                    raw_per_ue_25_mbps=access_raw_rate(env, distance, 25),
                    feasible=distance <= env.cfg.access_radius_cells * env.cfg.cell_size_m,
                )
            )
        for distance in config["gateway_distances_m"]:
            gateway, forward = backhaul_rates(env, distance)
            for kind, rate in [("UAV-BS", gateway), ("UAV-UAV", forward)]:
                links.append(
                    dict(
                        profile=profile["name"],
                        link=kind,
                        distance_m=distance,
                        raw_capacity_mbps=rate,
                        raw_per_ue_25_mbps=None,
                        feasible=distance <= env.cfg.backhaul_radius_cells * env.cfg.cell_size_m,
                    )
                )
        required_access_bw = minimum_access_bandwidth(
            env,
            distance_m=200,
            users=config["target_load_ues"],
            target_mbps=config["target_ue_mbps"],
        )
        designs.append(
            dict(
                profile=profile["name"],
                min_access_bw_at_edge_mhz=None
                if required_access_bw is None
                else required_access_bw / 1e6,
                **gateway_design(env, config["target_load_ues"] * config["target_ue_mbps"]),
            )
        )
        for users in config["ue_counts"]:
            for access in config["access_distances_m"]:
                for gateway in config["gateway_distances_m"]:
                    rows.append(static_scene(profile, users, access, gateway))
        for gateway in [100, 200, 250, 300]:
            relay_rows.append(shared_relay_scene(profile, gateway))
        # Resource-preserving duplex checks for a fixed two-UAV link abstraction.
        gateway = backhaul_rates(env, config["hybrid_gateway_distance_m"])[0]
        forward = backhaul_rates(env, config["hybrid_forward_distance_m"])[1]
        # Offers are bounded by actual Access rate, n=10, d=100m and 4Mbps demand.
        per_node_offer = min(10 * 4.0, 10 * access_raw_rate(env, 100, 10))
        for hd in (False, True):
            fractions = np.linspace(0, 1, 101) if hd else [1.0]
            for row in hybrid_flow_budget(
                gateway,
                forward,
                min(config["hybrid_local_offer_mbps"], per_node_offer),
                min(config["hybrid_remote_offer_mbps"], per_node_offer),
                half_duplex=hd,
                fractions=fractions,
            ):
                # Existing drain constants were assigned at original radio power.
                # Do not report a higher-power radio as having free extra power.
                energy_valid = env.cfg.backhaul_tx_dbm == BS5LaunchConfig().backhaul_tx_dbm
                hub_energy = (
                    hybrid_energy(env, moving=False, hybrid=row["mode"] == "Hybrid")
                    if energy_valid
                    else None
                )
                remote_energy = (
                    env.cfg.hover_drain + env.cfg.access_radio_drain if energy_valid else None
                )
                hybrid_rows.append(
                    dict(
                        profile=profile["name"],
                        gateway_capacity_mbps=gateway,
                        forward_capacity_mbps=forward,
                        local_priority=True,
                        hub_energy_units=hub_energy,
                        remote_energy_units=remote_energy,
                        fleet_energy_units=hub_energy + remote_energy if energy_valid else None,
                        energy_status="provisional_normalized"
                        if energy_valid
                        else "unvalidated_power_change",
                        **row,
                    )
                )
    return dict(
        static_scenes=rows,
        shared_relay=relay_rows,
        link_budgets=links,
        design_targets=designs,
        hybrid_airtime=hybrid_rows,
        baseline_config=asdict(make_radio().cfg),
        receipt=dict(
            version=config["version"],
            scope=config["scope"],
            environment_step_calls=0,
            training_updates=0,
            checkpoint_evaluations=0,
            scene_evaluations=len(rows) + len(relay_rows),
            direct_scenes=len(rows),
            shared_relay_scenes=len(relay_rows),
            hybrid_flow_evaluations=len(hybrid_rows),
            legacy_files_modified=False,
            random_trajectories_consumed=False,
            assumptions=[
                "Current simulator mean-path-loss/Shannon model; no interference or fading margin",
                "Static synthetic geometry, not coverage of realistic mobile demand",
                "Legacy relay routing uses shared capacity but no explicit half-duplex airtime ledger",
                "Hybrid is an offline two-node local-first scheduler with additive normalized radio energy",
                "Power30 profile requires energy/hardware validation before any training",
                "20MHz Access is a candidate only, not a proven optimal resource budget",
            ],
        ),
    )
