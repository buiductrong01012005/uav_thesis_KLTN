"""BS-rooted minimum-airtime forest routing and final network scoring."""

from copy import copy

import heapq

import numpy as np

from src.envs.radio.full import HybridFullConfig, score_hybrid_full

from src.envs.radio.hybrid import associate_fdm, access_fdm, bh_capacity, schedules_fdm

from src.envs.radio.airtime import Edge, node_position, access_rates, allocate_flows

from src.envs.radio.direct import score_network

from src.envs.reward import rate_metrics, TCFWeights

VERSION = "bs5_multihop_forest_1"


def forest_routes(env, association, roles):
    config = HybridFullConfig()
    active = [a for a in range(5) if env.status[a] == 0]
    forwarders = [a for a in active if roles[a] in (1, 2)]
    sources = [f"b{i}" for i in range(len(env.surviving_bs_xy))] + [f"u{a}" for a in forwarders]
    graph = {s: [] for s in sources}
    caps = {}
    for source in sources:
        for aid in active:
            target = f"u{aid}"
            if source == target:
                continue
            if (
                np.linalg.norm(node_position(env, source)[0] - node_position(env, target)[0])
                > env.cfg.backhaul_radius_cells * env.cfg.cell_size_m + 1e-9
            ):
                continue
            capacity = bh_capacity(env, [Edge(source, target, 0)], (0,), 0, roles, config)
            if capacity > 1e-12:
                caps[source, target] = capacity
                graph[source].append((target, 1.0 / capacity))
    distances, parent, queue = {}, {}, []
    for source in sources:
        if source.startswith("b"):
            distances[source] = (0.0, 0, (source,))
            heapq.heappush(queue, (0.0, 0, (source,), source))
    while queue:
        cost, hops, signature, source = heapq.heappop(queue)
        if distances[source] != (cost, hops, signature):
            continue
        for target, extra in graph.get(source, []):
            proposal = (cost + extra, hops + 1, signature + (target,))
            if target not in distances or proposal < distances[target]:
                distances[target] = proposal
                parent[target] = source
                heapq.heappush(queue, (*proposal, target))
    paths, chosen = {}, {}
    for aid in sorted(association):
        node, route, seen = f"u{aid}", [], set()
        while node in parent:
            assert node not in seen, "Routing cycle"
            seen.add(node)
            source = parent[node]
            assert distances[source][0] < distances[node][0]
            route.append((source, node))
            node = source
        paths[aid] = list(reversed(route)) if node.startswith("b") else []
        for edge in paths[aid]:
            chosen[edge] = Edge(*edge, caps[edge])
    edges = [chosen[key] for key in sorted(chosen)]
    index = {(edge.source, edge.target): i for i, edge in enumerate(edges)}
    assert len(edges) <= len(active)
    return edges, {a: [index[e] for e in path] for a, path in paths.items()}, parent


def enrich(result, env, roles):
    edges, paths, flows = result["edges"], result["paths"], result["flows"]
    forwarded = {a: 0.0 for a in range(5)}
    relay_touch = set()
    for aid, path in paths.items():
        for i in path[:-1]:
            node = edges[i].target
            assert node.startswith("u")
            forwarded[int(node[1:])] += flows[aid]
            if roles[int(node[1:])] == 2:
                relay_touch.add(aid)
    total = sum(flows.values())
    hops = {a: len(path) for a, path in paths.items()}
    metrics = result["metrics"]
    metrics.update(
        own_hybrid_mbps=sum(flows[a] for a in flows if roles[a] == 2),
        # Count each end-to-end destination flow once, even across several Hs.
        hybrid_forwarded_mbps=sum(flows[a] for a in relay_touch),
        mean_wireless_hops_delivered=sum(flows[a] * hops[a] for a in flows) / total
        if total > 1e-9
        else 0.0,
        mean_relay_hops_delivered=sum(flows[a] * max(0, hops[a] - 1) for a in flows) / total
        if total > 1e-9
        else 0.0,
        max_wireless_hops_used=max([hops[a] for a in flows if flows[a] > 1e-9], default=0),
        multihop_beyond_one_relay_mbps=sum(flows[a] for a in flows if hops[a] > 2),
        routed_via_relay_mbps=sum(flows[a] for a in flows if hops[a] > 1),
    )
    for hop in range(1, 6):
        metrics[f"delivered_mbps_hops_{hop}"] = sum(flows[a] for a in flows if hops[a] == hop)
    result.update(routing_version=VERSION, forwarded_mbps_by_uav=forwarded, path_wireless_hops=hops)
    return result


def score_multihop(env, roles, weights=None):
    weights = weights or TCFWeights()
    roles = np.asarray(roles, dtype=int)
    if roles.shape != (5,) or not np.isin(roles, [0, 1, 2]).all():
        raise ValueError("Expected five A/R/H roles")
    if (
        env.cfg.access_radius_cells * env.cfg.cell_size_m != 200
        or env.cfg.backhaul_radius_cells * env.cfg.cell_size_m != 300
        or env.cfg.backhaul_tx_dbm != 20
        or env.cfg.uav_altitude_m != 100
    ):
        raise ValueError("Expected legacy R200/Rbh300/BS20/h100 radio")
    if env.service_config.spectrum_mode != "colored_reuse":
        raise ValueError("Expected legacy colored reuse")
    # Exact old numerical radio for C-N / no-forwarder assignments.
    if not np.any((env.status == 0) & np.isin(roles, [1, 2])):
        return enrich(score_hybrid_full(env, roles, weights), env, roles)
    cfg = HybridFullConfig()
    view = copy(env)
    view.roles = np.where(roles == 2, 0, roles)
    has_h = np.any((env.status == 0) & (roles == 2))
    association = (
        associate_fdm(view, roles, cfg)
        if has_h
        else score_network(view, allow_relay=False)["association"]
    )
    view.roles = roles.copy()
    view.roles[view.status != 0] = -1
    if has_h:
        before, colors = access_fdm(view, association, roles, cfg)
    else:
        before, _, _ = access_rates(
            view,
            association,
            env.service_config.access_channel_mhz * env.service_config.access_occupied_fraction,
            "colored_reuse",
            env.service_config.rate_efficiency,
        )
        colors = {}
    edges, paths, parent = forest_routes(view, association, roles)
    matrix, states = schedules_fdm(view, edges, roles, cfg)
    offers = {a: float(before[users].sum()) for a, users in association.items()}
    flows, durations, minimum, violation = allocate_flows(offers, paths, matrix)
    rates = np.zeros(env.cfg.num_ues)
    for a, users in association.items():
        if offers[a] > 0:
            rates[users] = before[users] * flows[a] / offers[a]
    incidence = np.zeros((len(edges), len(offers)))
    for j, a in enumerate(sorted(offers)):
        incidence[paths[a], j] = 1
    load = incidence @ np.asarray([flows[a] for a in sorted(offers)])
    capacity = matrix @ durations
    actual = float(np.maximum(load - capacity, 0).max(initial=0))
    assert durations.sum() <= 1 + 1e-6 and actual <= 1e-6
    assert rates.sum() <= before.sum() + 1e-6
    covered = np.zeros(env.cfg.num_ues, dtype=bool)
    for a in association:
        covered |= (
            np.linalg.norm(env.ue_positions - env.uav_positions[a], axis=1)
            <= env.cfg.access_radius_cells + 1e-9
        )
    metrics = rate_metrics(rates, weights=weights)
    metrics.update(
        qos_ues_tol=int(np.sum(rates >= 4 - 1e-6)),
        qos_fraction_tol=float(np.mean(rates >= 4 - 1e-6)),
        covered_ues=int(covered.sum()),
        associated_ues=sum(map(len, association.values())),
        reachable_ues=sum(len(association[a]) for a, path in paths.items() if path),
        active_access=int(np.sum((env.status == 0) & (roles == 0))),
        active_relay=int(np.sum((env.status == 0) & (roles == 1))),
        active_hybrid=int(np.sum((env.status == 0) & (roles == 2))),
        access_offer_mbps=float(before.sum()),
        schedule_airtime=float(durations.sum()),
        flow_violation_mbps=max(violation, actual),
    )
    return enrich(
        dict(
            version=VERSION,
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
            metrics=metrics,
            parents=parent,
        ),
        env,
        roles,
    )
