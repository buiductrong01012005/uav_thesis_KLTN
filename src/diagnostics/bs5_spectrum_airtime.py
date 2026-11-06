"""Versioned fleet-level static spectrum/airtime abstraction, not a new RL env.

Reuses BS5 mean-path-loss radio, UE cap and association. Adds explicit PSD-based
Access interference and a globally feasible half-duplex backhaul schedule.
The LP is an offline oracle for fixed association/routes, not an online policy.
No env.step(), battery trajectory, weights or test scenario is used.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from src.diagnostics.bs5_resource_audit import make_radio
from src.envs.v0_env import ACCESS, RELAY


LAYOUTS = {
    "near_gateway": [[4, 3], [4, 5], [6, 3], [6, 5], [5, 4]],
    "distributed": [[6, 4], [6, 8], [10, 6], [12, 10], [14, 12]],
    "relay_corridor": [[13, 10], [13, 13], [15, 14], [16, 16], [17, 17]],
}
ROLE_FIXTURES = {"5A": [], "4A1R": [0], "3A2R": [0, 1]}


def dbm_to_mw(dbm):
    return np.power(10.0, np.asarray(dbm) / 10)


def spectral_rate(signal_mw: float, interference_mw: float, noise_mw: float,
                  bandwidth_hz: float, efficiency: float = 1.0) -> float:
    if signal_mw < 0 or interference_mw < 0 or noise_mw <= 0 or bandwidth_hz <= 0:
        raise ValueError("Invalid radio budget")
    if not 0 < efficiency <= 1:
        raise ValueError("Efficiency outside (0,1]")
    return float(efficiency * bandwidth_hz * np.log2(1 + signal_mw / (noise_mw + interference_mw)) / 1e6)


def fixture_env(name: str, roles: str, config: dict):
    env = make_radio(num_ues=config["users"])
    env.cfg.access_tx_dbm = config["access_power_dbm"]
    env.cfg.backhaul_tx_dbm = config["backhaul_power_dbm"]
    env.cfg.ue_demand_mbps = config["target_ue_mbps"]
    env.uav_positions[:] = LAYOUTS[name]
    env.roles[:] = ACCESS
    env.roles[ROLE_FIXTURES[roles]] = RELAY
    # Same UE geometry for every role/resource choice, no reset/RNG leakage.
    rng = np.random.default_rng(np.random.SeedSequence([config["diagnostic_seed"], list(LAYOUTS).index(name)]))
    centers = np.asarray(LAYOUTS[name], dtype=float)
    group = np.arange(config["users"]) % 5
    radius = rng.uniform(0, 1.2, config["users"])
    angle = rng.uniform(0, 2 * np.pi, config["users"])
    offset = radius[:, None] * np.column_stack([np.cos(angle), np.sin(angle)])
    env.ue_positions[:] = np.clip(centers[group] + offset, 0, 19.999)
    return env


def spectrum_colors(env, ids: list[int]) -> dict[int, int]:
    """Exact coloring of <=5 Access UAVs; paper-inspired conflict radius 2R.

    Distant same-color transmitters still contribute interference in the radio
    calculation. The conflict graph is a planning heuristic, not an RF cutoff.
    """
    conflicts = [(i, j) for i, a in enumerate(ids) for j, b in enumerate(ids) if i < j
                 and np.linalg.norm(env.uav_positions[a] - env.uav_positions[b]) <= 2 * env.cfg.access_radius_cells + 1e-9]
    for count in range(1, len(ids) + 1):
        for tail in itertools.product(range(count), repeat=max(0, len(ids) - 1)):
            colors = (0,) + tail
            if all(colors[i] != colors[j] for i, j in conflicts):
                return dict(zip(ids, colors))
    return {}


def access_rates(env, association: dict, bandwidth_mhz: float, mode: str, efficiency: float):
    """Always-on flat PSD over allocated band; no interference distance cutoff.

    An interferer contributes P_total * gain * (victim_subband / interferer_band).
    Orthogonal modes have disjoint bands; reuse1 fully overlaps all UAV bands.
    Each serving UAV retains fixed total power split across its associated UEs.
    """
    if bandwidth_mhz <= 0 or mode not in {"orthogonal_pool", "orthogonal_per_uav", "reuse1", "colored_reuse"}:
        raise ValueError("Unknown spectrum allocation")
    cfg = env.cfg
    ids = sorted(association)
    count = len(ids)
    rates = np.zeros(cfg.num_ues)
    sinr_db = np.full(cfg.num_ues, np.nan)
    if not count:
        return rates, sinr_db, {}
    per_uav_bw = bandwidth_mhz * 1e6 / count if mode == "orthogonal_pool" else bandwidth_mhz * 1e6
    colors = spectrum_colors(env, ids) if mode == "colored_reuse" else {a: j for j, a in enumerate(ids)}
    tx = float(dbm_to_mw(cfg.access_tx_dbm))
    emitting = [a for a in ids if association[a]]
    for aid in ids:
        users = association[aid]
        if not users:
            continue
        user_bw = per_uav_bw / len(users)
        noise = float(dbm_to_mw(cfg.noise_psd_dbm_hz + 10 * np.log10(user_bw) + cfg.noise_figure_db))
        for u in users:
            ue_xy = env._cell_to_xy(env.ue_positions[u])
            loss = env._a2g_path_loss_db(ue_xy, cfg.ue_height_m, env._cell_to_xy(env.uav_positions[aid]),
                                        cfg.uav_altitude_m, cfg.access_fc_hz)
            signal = tx * 10 ** (-loss / 10) / len(users)
            interference = 0.0
            if mode in {"reuse1", "colored_reuse"}:
                for other in emitting:
                    if other == aid or (mode == "colored_reuse" and colors[other] != colors[aid]):
                        continue
                    other_loss = env._a2g_path_loss_db(
                        ue_xy, cfg.ue_height_m, env._cell_to_xy(env.uav_positions[other]),
                        cfg.uav_altitude_m, cfg.access_fc_hz,
                    )
                    interference += tx * 10 ** (-other_loss / 10) * user_bw / per_uav_bw
            sinr_db[u] = 10 * np.log10(signal / (noise + interference))
            rates[u] = min(cfg.ue_demand_mbps,
                           spectral_rate(signal, interference, noise, user_bw, efficiency))
    budget = dict(
        access_per_uav_mhz=per_uav_bw / 1e6,
        access_distinct_used_mhz=bandwidth_mhz * (
            len(set(colors.values())) if mode == "colored_reuse" else count if mode == "orthogonal_per_uav" else 1
        ),
        access_reserved_fleet_mhz=bandwidth_mhz * 5 if mode in {"orthogonal_per_uav", "colored_reuse"} else bandwidth_mhz,
        emitting_access_uavs=len(emitting),
    )
    return rates, sinr_db, budget


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    solo_mbps: float


def node_position(env, node: str):
    if node[0] == "b":
        return env.surviving_bs_xy[int(node[1:])], env.cfg.bs_height_m
    return env._cell_to_xy(env.uav_positions[int(node[1:])]), env.cfg.uav_altitude_m


def link_gain(env, source: str, target: str) -> float:
    src_xy, src_height = node_position(env, source)
    dst_xy, dst_height = node_position(env, target)
    if source[0] == target[0] == "u":
        loss = env._a2a_path_loss_db(src_xy, dst_xy, env.cfg.backhaul_fc_hz)
    else:
        loss = env._a2g_path_loss_db(src_xy, src_height, dst_xy, dst_height, env.cfg.backhaul_fc_hz)
    return float(10 ** (-loss / 10))


def build_routes(env, association: dict, bandwidth_mhz: float, efficiency: float):
    """Fixed one-relay routes by best isolated bottleneck, ties deterministic.

    Does not jointly optimize route/schedule. Original radius gates retained.
    All BS and UAV transmitters use the same 20dBm reference by default, as the
    legacy symmetric BH model; this is not copied from cited LTE deployments.
    """
    cfg = env.cfg
    bw = bandwidth_mhz * 1e6
    noise = float(dbm_to_mw(cfg.noise_psd_dbm_hz + 10 * np.log10(bw) + cfg.noise_figure_db))
    tx = float(dbm_to_mw(cfg.backhaul_tx_dbm))
    access_ids = sorted(association)
    relays = np.flatnonzero(env.roles == RELAY)

    def possible(source, target):
        return np.linalg.norm(node_position(env, source)[0] - node_position(env, target)[0]) <= cfg.backhaul_radius_cells * cfg.cell_size_m + 1e-9

    def cap(source, target):
        return spectral_rate(tx * link_gain(env, source, target), 0, noise, bw, efficiency)

    paths, chosen_edges = {}, {}
    for aid in access_ids:
        target = f"u{aid}"
        candidates = []
        for b in range(len(env.surviving_bs_xy)):
            bs = f"b{b}"
            if possible(bs, target):
                candidates.append((cap(bs, target), ((bs, target),)))
            for r in relays:
                relay = f"u{r}"
                if possible(bs, relay) and possible(relay, target):
                    candidates.append((min(cap(bs, relay), cap(relay, target)), ((bs, relay), (relay, target))))
        if not candidates:
            paths[aid] = []
            continue
        _, route = min(candidates, key=lambda x: (-x[0], len(x[1]), x[1]))
        paths[aid] = list(route)
        for source, destination in route:
            chosen_edges[source, destination] = Edge(source, destination, cap(source, destination))
    edges = [chosen_edges[key] for key in sorted(chosen_edges)]
    index = {(edge.source, edge.target): i for i, edge in enumerate(edges)}
    return edges, {a: [index[key] for key in route] for a, route in paths.items()}


def valid_half_duplex_state(edges: list[Edge], active: tuple[int, ...]) -> bool:
    """One single-beam, single-radio link per endpoint; includes each BS."""
    nodes = [n for i in active for n in (edges[i].source, edges[i].target)]
    return len(nodes) == len(set(nodes))


def schedule_matrix(env, edges: list[Edge], bandwidth_mhz: float, efficiency: float, mode: str):
    if mode == "ideal_parallel_bound":
        # Deliberately optimistic, not physically feasible on a single radio.
        return np.asarray([[e.solo_mbps] for e in edges]).reshape(len(edges), 1), [tuple(range(len(edges)))]
    if mode != "shared_half_duplex":
        raise ValueError("Unknown backhaul mode")
    states = [tuple(i for i, flag in enumerate(mask) if flag)
              for mask in itertools.product([0, 1], repeat=len(edges)) if any(mask)]
    states = [state for state in states if valid_half_duplex_state(edges, state)]
    matrix = np.zeros((len(edges), len(states)))
    bw = bandwidth_mhz * 1e6
    cfg = env.cfg
    noise = float(dbm_to_mw(cfg.noise_psd_dbm_hz + 10 * np.log10(bw) + cfg.noise_figure_db))
    tx = float(dbm_to_mw(cfg.backhaul_tx_dbm))
    for j, active in enumerate(states):
        for i in active:
            edge = edges[i]
            interference = sum(tx * link_gain(env, edges[k].source, edge.target) for k in active if k != i)
            matrix[i, j] = spectral_rate(tx * link_gain(env, edge.source, edge.target),
                                         interference, noise, bw, efficiency)
    return matrix, states


def allocate_flows(offers: dict[int, float], paths: dict[int, list[int]], capacities: np.ndarray):
    """Lexicographic LP: maximum Mbps, then max-min offer fulfillment.

    q_a uses the same proportional scaling of UE rates within each Access as
    the legacy env. Global constraints: D q <= C t, sum(t)<=1. Fixed-route
    steady-state fluid capacity only, not packet delay or empty-buffer startup.
    """
    ids = sorted(offers)
    n, edges, states = len(ids), *capacities.shape
    if np.any(~np.isfinite(capacities)) or np.any(capacities < 0):
        raise ValueError("Invalid link capacities")
    demand = np.zeros((edges, n))
    for j, aid in enumerate(ids):
        for edge in paths[aid]:
            demand[edge, j] = 1
    eligible = [j for j, aid in enumerate(ids) if paths[aid] and offers[aid] > 1e-12]
    if not eligible:
        return {aid: 0.0 for aid in ids}, np.zeros(states), 0.0, 0.0
    # Variables: per-Access Mbps, state durations, minimum satisfaction lambda.
    size = n + states + 1
    constraints, rhs = [], []
    for e in range(edges):
        row = np.zeros(size)
        row[:n] = demand[e]
        row[n:n + states] = -capacities[e]
        constraints.append(row)
        rhs.append(0.0)
    row = np.zeros(size)
    row[n:n + states] = 1
    constraints.append(row)
    rhs.append(1.0)
    for j in eligible:
        row = np.zeros(size)
        row[j] = -1
        row[-1] = offers[ids[j]]
        constraints.append(row)
        rhs.append(0.0)
    bounds = [(0, offers[a] if paths[a] else 0) for a in ids] + [(0, 1)] * states + [(0, 1)]
    a_ub, b_ub = np.asarray(constraints), np.asarray(rhs)
    options = {"dual_feasibility_tolerance": 1e-8, "primal_feasibility_tolerance": 1e-8}
    objective = np.zeros(size)
    objective[:n] = -1
    first = linprog(objective, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs", options=options)
    if not first.success:
        raise RuntimeError(first.message)
    maximum_total = float(first.x[:n].sum())
    total_constraint = np.zeros(size)
    total_constraint[:n] = -1
    a_ub = np.vstack([a_ub, total_constraint])
    b_ub = np.append(b_ub, -max(0, maximum_total - 1e-8))
    objective[:] = 0
    objective[-1] = -1
    second = linprog(objective, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs", options=options)
    if not second.success:
        raise RuntimeError(second.message)
    violation = max(0.0, float(np.max(a_ub @ second.x - b_ub)))
    if violation > 1e-6:
        raise AssertionError(f"Infeasible LP result: {violation}")
    minimum = float(second.x[-1])
    return dict(zip(ids, second.x[:n])), second.x[n:n + states], minimum, violation


def audit_case(env, *, access_bw: float, backhaul_bw: float, spectrum_mode: str,
               backhaul_mode: str, efficiency: float):
    legacy = env.compute_network()
    association = legacy["association"]  # matched, legacy capped signal-strength association
    before, sinr, spectrum = access_rates(env, association, access_bw, spectrum_mode, efficiency)
    offers = {aid: float(before[users].sum()) for aid, users in association.items()}
    edges, paths = build_routes(env, association, backhaul_bw, efficiency)
    matrix, states = schedule_matrix(env, edges, backhaul_bw, efficiency, backhaul_mode)
    allocated, durations, minimum, violation = allocate_flows(offers, paths, matrix)
    final = np.zeros(env.cfg.num_ues)
    for aid, users in association.items():
        if offers[aid] > 0:
            final[users] = before[users] * allocated[aid] / offers[aid]
    totals = sum(offers.values())
    served = float(final.sum())
    if served > totals + 1e-6 or np.any(final > env.cfg.ue_demand_mbps + 1e-6):
        raise AssertionError("Delivered data exceeds offer/demand")
    sinr_present = sinr[np.isfinite(sinr)]
    metric = dict(
        access_bw_label_mhz=access_bw, backhaul_total_bandwidth_mhz=backhaul_bw,
        spectrum_mode=spectrum_mode, backhaul_mode=backhaul_mode, rate_efficiency=efficiency,
        access_tx_dbm=env.cfg.access_tx_dbm, backhaul_tx_dbm=env.cfg.backhaul_tx_dbm,
        access_offer_mbps=totals, delivered_mbps=served,
        backhaul_loss_mbps=max(0, totals - served), utility=served / (env.cfg.num_ues * env.cfg.ue_demand_mbps),
        associated_ues=sum(len(users) for users in association.values()),
        positive_service_ues=int(np.sum(final > 1e-7)),
        qos_ues=int(np.sum(final >= env.cfg.ue_demand_mbps - 1e-6)),
        sinr_median_db=None if not len(sinr_present) else float(np.median(sinr_present)),
        reachable_access_uavs=sum(bool(paths[a]) and offers[a] > 0 for a in association),
        relay_routes=sum(len(path) == 2 for path in paths.values()),
        edge_count=len(edges), schedule_states=len(states), schedule_time_sum=float(durations.sum()),
        min_access_offer_fulfillment=minimum, lp_constraint_violation_mbps=violation, **spectrum,
    )
    detail = dict(
        association=association, paths=paths, access_offers=offers, access_flows=allocated,
        ue_access_rates=before.tolist(), ue_delivered_rates=final.tolist(),
        edges=[dict(source=e.source, target=e.target, solo_mbps=e.solo_mbps) for e in edges],
        schedules=[dict(active_edges=list(active), duration=float(durations[j]),
                        capacity_mbps=matrix[:, j].tolist()) for j, active in enumerate(states) if durations[j] > 1e-10],
    )
    return metric, detail
