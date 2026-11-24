"""Access and backhaul spectrum/airtime calculations for Hybrid capabilities."""

from copy import copy

from dataclasses import dataclass, asdict

import itertools

import math

import numpy as np

from src.envs.radio.airtime import (
    Edge,
    access_rates,
    allocate_flows,
    build_routes,
    dbm_to_mw,
    link_gain,
    node_position,
    schedule_matrix,
    spectral_rate,
    spectrum_colors,
    valid_half_duplex_state,
)

from src.envs.radio.direct import score_network

from src.envs.reward import TCFWeights, rate_metrics

ACCESS, RELAY, HYBRID = (0, 1, 2)

VERSION = "bs5_hybrid_fdm_static_1"


@dataclass(frozen=True)
class HybridFDMConfig:
    access_fraction: float = 0.5
    qos_tolerance_mbps: float = 1e-6

    def __post_init__(self):
        if not math.isfinite(self.access_fraction) or not 0 < self.access_fraction < 1:
            raise ValueError("Hybrid fractions must be positive and sum to one")
        if not math.isfinite(self.qos_tolerance_mbps) or not 0 <= self.qos_tolerance_mbps <= 1e-4:
            raise ValueError("Invalid numerical QoS tolerance")

    @property
    def backhaul_fraction(self):
        return 1 - self.access_fraction


def overlap_rate(bandwidth_hz, signal_psd, interferers, noise_psd, efficiency=1.0):
    """Integrate Shannon on nested [0, B] bands; PSD is received mW/Hz.

    Each interferer is (emitted_bandwidth_hz, received_PSD). This avoids using
    average interference over a wide victim band when only half is overlapped.
    Access uses uniformly interleaved UE subcarriers, so caller scales this
    full allocated-band result by the serving UE's bandwidth fraction.
    """
    if bandwidth_hz <= 0 or signal_psd < 0 or noise_psd <= 0:
        raise ValueError("Invalid PSD/bandwidth")
    if any(b <= 0 or p < 0 for b, p in interferers):
        raise ValueError("Invalid interferer")
    cuts = sorted(
        {0.0, float(bandwidth_hz), *(float(b) for b, _ in interferers if b < bandwidth_hz)}
    )
    total = 0.0
    for left, right in zip(cuts[:-1], cuts[1:]):
        midpoint = (left + right) / 2
        interference = sum(p for b, p in interferers if b > midpoint)
        total += spectral_rate(signal_psd, interference, noise_psd, right - left, efficiency)
    return total


def fractions(roles, config):
    return np.where(np.asarray(roles) == HYBRID, config.access_fraction, 1.0), np.where(
        np.asarray(roles) == HYBRID, config.backhaul_fraction, 1.0
    )


def associate_fdm(env, roles, config):
    """Legacy capped strongest-signal association, with actual Hybrid TX power."""
    ids = [a for a in range(len(roles)) if env.status[a] == 0 and roles[a] in (ACCESS, HYBRID)]
    scores = np.full((env.cfg.num_ues, len(ids)), -np.inf)
    alpha, _ = fractions(roles, config)
    for j, aid in enumerate(ids):
        for user, ue in enumerate(env.ue_positions):
            if np.linalg.norm(ue - env.uav_positions[aid]) <= env.cfg.access_radius_cells + 1e-9:
                loss = env._a2g_path_loss_db(
                    env._cell_to_xy(ue),
                    env.cfg.ue_height_m,
                    env._cell_to_xy(env.uav_positions[aid]),
                    env.cfg.uav_altitude_m,
                    env.cfg.access_fc_hz,
                )
                scores[user, j] = env.cfg.access_tx_dbm + 10 * np.log10(alpha[aid]) - loss
    feasible = np.isfinite(scores).sum(axis=1)
    if len(ids) > 1:
        top = np.partition(np.where(np.isfinite(scores), scores, -1e9), -2, axis=1)[:, -2:]
        order = np.lexsort((np.abs(top[:, 1] - top[:, 0]), feasible))
    else:
        order = np.argsort(feasible)
    association = {a: [] for a in ids}
    for user in order:
        if feasible[user] == 0:
            continue
        for j in np.argsort(scores[user])[::-1]:
            aid = ids[j]
            if np.isfinite(scores[user, j]) and len(association[aid]) < env.cfg.max_ues_per_access:
                association[aid].append(int(user))
                break
    return association


def access_fdm(env, association, roles, config):
    c, cfg = env.service_config, env.cfg
    bw = c.access_channel_mhz * c.access_occupied_fraction * 1e6
    ids = sorted(association)
    alpha, _ = fractions(roles, config)
    colors = spectrum_colors(env, ids)
    rates = np.zeros(cfg.num_ues)
    noise_psd = float(dbm_to_mw(cfg.noise_psd_dbm_hz + cfg.noise_figure_db))
    nominal_power = float(dbm_to_mw(cfg.access_tx_dbm))
    for aid in ids:
        users = association[aid]
        if not users:
            continue
        band = bw * alpha[aid]
        power = nominal_power * alpha[aid]
        for user in users:
            ue = env._cell_to_xy(env.ue_positions[user])
            loss = env._a2g_path_loss_db(
                ue,
                cfg.ue_height_m,
                env._cell_to_xy(env.uav_positions[aid]),
                cfg.uav_altitude_m,
                cfg.access_fc_hz,
            )
            signal_psd = power * 10 ** (-loss / 10) / band
            interferers = []
            for other in ids:
                if other == aid or not association[other] or colors[other] != colors[aid]:
                    continue
                loss = env._a2g_path_loss_db(
                    ue,
                    cfg.ue_height_m,
                    env._cell_to_xy(env.uav_positions[other]),
                    cfg.uav_altitude_m,
                    cfg.access_fc_hz,
                )
                other_band = bw * alpha[other]
                interferers.append(
                    (other_band, nominal_power * alpha[other] * 10 ** (-loss / 10) / other_band)
                )
            capacity = overlap_rate(band, signal_psd, interferers, noise_psd, c.rate_efficiency)
            rates[user] = min(cfg.ue_demand_mbps, capacity / len(users))
    return rates, colors


def edge_budget(env, source, target, roles, config):
    """Return negotiated occupied Hz and source RF mW, never extra fleet MHz."""
    _, beta = fractions(roles, config)
    nominal = (
        env.service_config.backhaul_total_mhz * env.service_config.backhaul_occupied_fraction * 1e6
    )
    source_fraction = 1.0 if source.startswith("b") else beta[int(source[1:])]
    target_fraction = 1.0 if target.startswith("b") else beta[int(target[1:])]
    bandwidth = nominal * min(source_fraction, target_fraction)
    power = float(dbm_to_mw(env.cfg.backhaul_tx_dbm)) * source_fraction
    return bandwidth, power


def bh_capacity(env, edges, active, victim, roles, config):
    edge = edges[victim]
    bw, power = edge_budget(env, edge.source, edge.target, roles, config)
    signal = power * link_gain(env, edge.source, edge.target) / bw
    interferers = []
    for k in active:
        if k == victim:
            continue
        other = edges[k]
        other_bw, other_power = edge_budget(env, other.source, other.target, roles, config)
        interferers.append(
            (other_bw, other_power * link_gain(env, other.source, edge.target) / other_bw)
        )
    noise = float(dbm_to_mw(env.cfg.noise_psd_dbm_hz + env.cfg.noise_figure_db))
    return overlap_rate(bw, signal, interferers, noise, env.service_config.rate_efficiency)


def routes_fdm(env, association, roles, config):
    forwarding = [
        a for a in range(len(roles)) if env.status[a] == 0 and roles[a] in (RELAY, HYBRID)
    ]

    def possible(source, target):
        return (
            np.linalg.norm(node_position(env, source)[0] - node_position(env, target)[0])
            <= env.cfg.backhaul_radius_cells * env.cfg.cell_size_m + 1e-9
        )

    def cap(source, target):
        return bh_capacity(env, [Edge(source, target, 0)], (0,), 0, roles, config)

    chosen, paths = {}, {}
    for aid in sorted(association):
        target = f"u{aid}"
        candidates = []
        for b in range(len(env.surviving_bs_xy)):
            source = f"b{b}"
            if possible(source, target):
                candidates.append((cap(source, target), ((source, target),)))
            # A forwarding Hybrid must be BS-adjacent; its own service is direct.
            if roles[aid] == HYBRID:
                continue
            for rid in forwarding:
                relay = f"u{rid}"
                if rid != aid and possible(source, relay) and possible(relay, target):
                    candidates.append(
                        (
                            min(cap(source, relay), cap(relay, target)),
                            ((source, relay), (relay, target)),
                        )
                    )
        if not candidates:
            paths[aid] = []
            continue
        _, path = min(candidates, key=lambda x: (-x[0], len(x[1]), x[1]))
        paths[aid] = list(path)
        for source, target in path:
            chosen[source, target] = Edge(source, target, cap(source, target))
    edges = [chosen[key] for key in sorted(chosen)]
    index = {(e.source, e.target): i for i, e in enumerate(edges)}
    return edges, {a: [index[key] for key in path] for a, path in paths.items()}


def schedules_fdm(env, edges, roles, config):
    states = [
        tuple(i for i, flag in enumerate(mask) if flag)
        for mask in itertools.product([0, 1], repeat=len(edges))
        if any(mask)
    ]
    states = [state for state in states if valid_half_duplex_state(edges, state)]
    matrix = np.zeros((len(edges), len(states)))
    for j, state in enumerate(states):
        for i in state:
            matrix[i, j] = bh_capacity(env, edges, state, i, roles, config)
    return matrix, states


def score_hybrid_fdm(env, roles, config=None, weights=None):
    """Pure snapshot scorer; no role/action/lifecycle/observation mutation.

    No-H assignments use the historical radio functions exactly. For H, the
    historical signal-order/cap heuristic uses the actual reduced TX power.
    Global flow allocation keeps the old max-throughput then max-min-offer LP.
    Thus this is not an oracle over all associations/routes/UE allocations.
    """
    config = config or HybridFDMConfig()
    weights = weights or TCFWeights(0.45, 0.45, 0.10)
    roles = np.asarray(roles, dtype=int)
    if roles.shape != env.roles.shape or not np.isin(roles, [ACCESS, RELAY, HYBRID]).all():
        raise ValueError("One legal role per UAV required")
    c = env.service_config
    if c.spectrum_mode != "colored_reuse":
        raise ValueError("This probe preserves legacy colored_reuse only")
    if (
        env.cfg.access_radius_cells * env.cfg.cell_size_m != 200
        or env.cfg.backhaul_tx_dbm != 20
        or env.cfg.uav_altitude_m != 100
        or env.cfg.backhaul_radius_cells * env.cfg.cell_size_m != 300
    ):
        raise ValueError("Expected old R200/h100/Pbh20/Rbh300 geometry and radio")
    view = copy(env)
    view.roles = np.where(roles == HYBRID, ACCESS, roles)
    has_hybrid = bool(np.any((roles == HYBRID) & (env.status == 0)))
    association = (
        associate_fdm(view, roles, config)
        if has_hybrid
        else score_network(view, allow_relay=False)["association"]
    )
    view.roles = roles.copy()
    view.roles[env.status != 0] = -1
    if has_hybrid:
        before, colors = access_fdm(view, association, roles, config)
        edges, paths = routes_fdm(view, association, roles, config)
        matrix, states = schedules_fdm(view, edges, roles, config)
    else:
        before, _, _ = access_rates(
            view,
            association,
            c.access_channel_mhz * c.access_occupied_fraction,
            c.spectrum_mode,
            c.rate_efficiency,
        )
        colors = spectrum_colors(view, sorted(association))
        edges, paths = build_routes(
            view,
            association,
            c.backhaul_total_mhz * c.backhaul_occupied_fraction,
            c.rate_efficiency,
        )
        matrix, states = schedule_matrix(
            view,
            edges,
            c.backhaul_total_mhz * c.backhaul_occupied_fraction,
            c.rate_efficiency,
            "shared_half_duplex",
        )
    offers = {a: float(before[users].sum()) for a, users in association.items()}
    flows, durations, minimum, violation = allocate_flows(offers, paths, matrix)
    rates = np.zeros(env.cfg.num_ues)
    for aid, users in association.items():
        if offers[aid] > 0:
            rates[users] = before[users] * flows[aid] / offers[aid]
    incidence = np.zeros((len(edges), len(association)))
    ids = sorted(association)
    for j, aid in enumerate(ids):
        incidence[paths[aid], j] = 1
    load = incidence @ np.asarray([flows[a] for a in ids])
    capacity = matrix @ durations
    actual_violation = float(np.maximum(load - capacity, 0).max(initial=0))
    assert durations.sum() <= 1 + 1e-6 and actual_violation <= 1e-6
    assert rates.sum() <= before.sum() + 1e-6
    covered = np.zeros(env.cfg.num_ues, dtype=bool)
    for a in ids:
        covered |= (
            np.linalg.norm(env.ue_positions - env.uav_positions[a], axis=1)
            <= env.cfg.access_radius_cells + 1e-9
        )
    metrics = rate_metrics(rates, weights=weights)
    metrics.update(
        qos_ues_tol=float(np.sum(rates >= 4 - config.qos_tolerance_mbps)),
        qos_fraction_tol=float(np.mean(rates >= 4 - config.qos_tolerance_mbps)),
        covered_ues=int(covered.sum()),
        associated_ues=sum(map(len, association.values())),
        reachable_ues=sum(len(association[a]) for a, path in paths.items() if path),
        active_access=int(np.sum((env.status == 0) & (roles == ACCESS))),
        active_relay=int(np.sum((env.status == 0) & (roles == RELAY))),
        active_hybrid=int(np.sum((env.status == 0) & (roles == HYBRID))),
        access_offer_mbps=float(before.sum()),
        schedule_airtime=float(durations.sum()),
        flow_violation_mbps=max(actual_violation, violation),
        own_hybrid_mbps=sum(flows[a] for a in ids if roles[a] == HYBRID),
        hybrid_forwarded_mbps=sum(
            flows[a]
            for a in ids
            if len(paths[a]) == 2 and roles[int(edges[paths[a][0]].target[1:])] == HYBRID
        ),
    )
    alpha, beta = fractions(roles, config)
    power_bounds = []
    for aid in range(len(roles)):
        active = env.status[aid] == 0
        access_power = (
            float(dbm_to_mw(env.cfg.access_tx_dbm)) * alpha[aid]
            if active and aid in association and association[aid]
            else 0.0
        )
        can_forward = active and roles[aid] in (RELAY, HYBRID)
        bh_power = float(dbm_to_mw(env.cfg.backhaul_tx_dbm)) * beta[aid] if can_forward else 0.0
        power_bounds.append(
            dict(
                agent=aid,
                access_tx_mw=access_power,
                bh_tx_limit_mw=bh_power,
                combined_tx_bound_mw=access_power + bh_power,
            )
        )
    return dict(
        version=VERSION,
        config=asdict(config),
        roles=roles.copy(),
        association=association,
        paths=paths,
        edges=edges,
        capacities=matrix,
        states=states,
        durations=durations,
        edge_load_mbps=load,
        edge_capacity_mbps=capacity,
        flows=flows,
        ue_rate_mbps=rates,
        access_raw_ue_rates=before,
        colors=colors,
        min_offer_fulfillment=minimum,
        power_bounds=power_bounds,
        metrics=metrics,
    )
