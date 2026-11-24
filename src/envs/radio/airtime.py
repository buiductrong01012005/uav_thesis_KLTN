"""Access spectrum, link interference and half-duplex flow/schedule LP."""

from __future__ import annotations

import itertools

from dataclasses import dataclass

import numpy as np

from scipy.optimize import linprog

from src.envs.config import RELAY


def dbm_to_mw(dbm):
    return np.power(10.0, np.asarray(dbm) / 10)


def spectral_rate(
    signal_mw: float,
    interference_mw: float,
    noise_mw: float,
    bandwidth_hz: float,
    efficiency: float = 1.0,
) -> float:
    if signal_mw < 0 or interference_mw < 0 or noise_mw <= 0 or bandwidth_hz <= 0:
        raise ValueError("Invalid radio budget")
    if not 0 < efficiency <= 1:
        raise ValueError("Efficiency outside (0,1]")
    return float(
        efficiency * bandwidth_hz * np.log2(1 + signal_mw / (noise_mw + interference_mw)) / 1e6
    )


def spectrum_colors(env, ids: list[int]) -> dict[int, int]:
    """Exact coloring of <=5 Access UAVs; paper-inspired conflict radius 2R.

    Distant same-color transmitters still contribute interference in the radio
    calculation. The conflict graph is a planning heuristic, not an RF cutoff.
    """
    conflicts = [
        (i, j)
        for i, a in enumerate(ids)
        for j, b in enumerate(ids)
        if i < j
        and np.linalg.norm(env.uav_positions[a] - env.uav_positions[b])
        <= 2 * env.cfg.access_radius_cells + 1e-9
    ]
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
    if bandwidth_mhz <= 0 or mode not in {
        "orthogonal_pool",
        "orthogonal_per_uav",
        "reuse1",
        "colored_reuse",
    }:
        raise ValueError("Unknown spectrum allocation")
    cfg = env.cfg
    ids = sorted(association)
    count = len(ids)
    rates = np.zeros(cfg.num_ues)
    sinr_db = np.full(cfg.num_ues, np.nan)
    if not count:
        return rates, sinr_db, {}
    per_uav_bw = bandwidth_mhz * 1e6 / count if mode == "orthogonal_pool" else bandwidth_mhz * 1e6
    colors = (
        spectrum_colors(env, ids) if mode == "colored_reuse" else {a: j for j, a in enumerate(ids)}
    )
    tx = float(dbm_to_mw(cfg.access_tx_dbm))
    emitting = [a for a in ids if association[a]]
    for aid in ids:
        users = association[aid]
        if not users:
            continue
        user_bw = per_uav_bw / len(users)
        noise = float(
            dbm_to_mw(cfg.noise_psd_dbm_hz + 10 * np.log10(user_bw) + cfg.noise_figure_db)
        )
        for u in users:
            ue_xy = env._cell_to_xy(env.ue_positions[u])
            loss = env._a2g_path_loss_db(
                ue_xy,
                cfg.ue_height_m,
                env._cell_to_xy(env.uav_positions[aid]),
                cfg.uav_altitude_m,
                cfg.access_fc_hz,
            )
            signal = tx * 10 ** (-loss / 10) / len(users)
            interference = 0.0
            if mode in {"reuse1", "colored_reuse"}:
                for other in emitting:
                    if other == aid or (mode == "colored_reuse" and colors[other] != colors[aid]):
                        continue
                    other_loss = env._a2g_path_loss_db(
                        ue_xy,
                        cfg.ue_height_m,
                        env._cell_to_xy(env.uav_positions[other]),
                        cfg.uav_altitude_m,
                        cfg.access_fc_hz,
                    )
                    interference += tx * 10 ** (-other_loss / 10) * user_bw / per_uav_bw
            sinr_db[u] = 10 * np.log10(signal / (noise + interference))
            rates[u] = min(
                cfg.ue_demand_mbps, spectral_rate(signal, interference, noise, user_bw, efficiency)
            )
    budget = dict(
        access_per_uav_mhz=per_uav_bw / 1e6,
        access_distinct_used_mhz=bandwidth_mhz
        * (
            len(set(colors.values()))
            if mode == "colored_reuse"
            else count
            if mode == "orthogonal_per_uav"
            else 1
        ),
        access_reserved_fleet_mhz=bandwidth_mhz * 5
        if mode in {"orthogonal_per_uav", "colored_reuse"}
        else bandwidth_mhz,
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
        return (
            np.linalg.norm(node_position(env, source)[0] - node_position(env, target)[0])
            <= cfg.backhaul_radius_cells * cfg.cell_size_m + 1e-9
        )

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
                    candidates.append(
                        (min(cap(bs, relay), cap(relay, target)), ((bs, relay), (relay, target)))
                    )
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
        return np.asarray([[e.solo_mbps] for e in edges]).reshape(len(edges), 1), [
            tuple(range(len(edges)))
        ]
    if mode != "shared_half_duplex":
        raise ValueError("Unknown backhaul mode")
    states = [
        tuple(i for i, flag in enumerate(mask) if flag)
        for mask in itertools.product([0, 1], repeat=len(edges))
        if any(mask)
    ]
    states = [state for state in states if valid_half_duplex_state(edges, state)]
    matrix = np.zeros((len(edges), len(states)))
    bw = bandwidth_mhz * 1e6
    cfg = env.cfg
    noise = float(dbm_to_mw(cfg.noise_psd_dbm_hz + 10 * np.log10(bw) + cfg.noise_figure_db))
    tx = float(dbm_to_mw(cfg.backhaul_tx_dbm))
    for j, active in enumerate(states):
        for i in active:
            edge = edges[i]
            interference = sum(
                tx * link_gain(env, edges[k].source, edge.target) for k in active if k != i
            )
            matrix[i, j] = spectral_rate(
                tx * link_gain(env, edge.source, edge.target), interference, noise, bw, efficiency
            )
    return matrix, states


LP_NUMERICAL_GUARD_VERSION = "lp_bound_roundoff_v1"

LP_SOLVER_TOL = 1e-08

LP_MAX_REPAIR_MBPS = 1e-06

_LP_EVENT_COUNT = 0


def _lp_diagnostic(stage, ids, offers, paths, capacities, raw, clean, reason):
    import json
    import os
    from pathlib import Path

    global _LP_EVENT_COUNT
    _LP_EVENT_COUNT += 1
    record = dict(
        version=LP_NUMERICAL_GUARD_VERSION,
        stage=stage,
        reason=reason,
        ids=[int(a) for a in ids],
        offers=[float(offers[a]) for a in ids],
        paths=[[int(edge) for edge in paths[a]] for a in ids],
        capacities=np.asarray(capacities).tolist(),
        raw=np.asarray(raw).tolist(),
        cleaned=None if clean is None else np.asarray(clean).tolist(),
    )

    # Nonfinite values stay explicit in rejected diagnostics; they are never
    # turned into a zero rate or a valid training sample.
    def finite_json(value):
        if isinstance(value, float) and not np.isfinite(value):
            return repr(value)
        if isinstance(value, list):
            return [finite_json(v) for v in value]
        if isinstance(value, dict):
            return {k: finite_json(v) for k, v in value.items()}
        return value

    folder = os.environ.get("UAV_LP_DIAGNOSTIC_DIR")
    if folder:
        try:
            destination = Path(folder)
            destination.mkdir(parents=True, exist_ok=True)
            # One process writes one output directory; append preserves earlier
            # diagnostics when a fresh Kaggle session resumes the same run.
            with (destination / "lp_numerics.jsonl").open("a") as stream:
                stream.write(json.dumps(finite_json(record), allow_nan=False) + "\n")
        except OSError as exc:
            print("LP diagnostic write failed:", exc, flush=True)
    if clean is None or _LP_EVENT_COUNT <= 8 or _LP_EVENT_COUNT & (_LP_EVENT_COUNT - 1) == 0:
        print(f"LP numerical guard {stage}: {reason}", flush=True)


def _clean_lp_solution(solution, ids, offers, paths, capacities, demand, a_ub, b_ub, bounds, stage):
    n = len(ids)
    states = capacities.shape[1]
    raw = np.asarray(solution, dtype=float)

    def reject(message):
        _lp_diagnostic(stage, ids, offers, paths, capacities, raw, None, message)
        raise FloatingPointError("Invalid LP output: " + message)

    if raw.shape != (len(bounds),) or not np.isfinite(raw).all():
        reject("nonfinite or wrong-sized solution")
    lower = np.asarray([b[0] for b in bounds], dtype=float)
    upper = np.asarray([b[1] for b in bounds], dtype=float)
    margin = (
        64
        * np.finfo(float).eps
        * max(1.0, float(np.abs(raw).max(initial=0)), float(upper.max(initial=0)))
    )
    tol = LP_SOLVER_TOL + margin
    bound_error = max(float((lower - raw).max(initial=0)), float((raw - upper).max(initial=0)))
    residual = float((a_ub @ raw - b_ub).max(initial=0))
    if bound_error > tol or residual > 1e-6:
        reject(f"bounds={bound_error:.9g}, constraints={residual:.9g}")
    cleaned = np.clip(raw, lower, upper)
    if np.array_equal(cleaned, raw):
        # Preserve every already bounded solution bit-for-bit. In particular,
        # do not perturb ordinary positive optima or retune the LP objective.
        return raw, residual
    duration = cleaned[n : n + states]
    if duration.sum() > 1:
        if duration.sum() - 1 > tol:
            reject("schedule time exceeds the solver tolerance")
        duration /= duration.sum()
    q = cleaned[:n]
    capacity = capacities @ duration
    # Removing a tiny negative flow can expose a tiny shared-edge overload.
    # Reduce only flows crossing that edge; a zero-capacity leaf must not zero
    # unrelated, large flows elsewhere in the forest. No rate is increased.
    for edge in range(len(capacity)):
        users = np.flatnonzero(demand[edge] > 0)
        load = float(q[users].sum())
        if load > capacity[edge] and load > 0:
            q[users] *= max(0.0, float(capacity[edge])) / load
    eligible = [j for j, a in enumerate(ids) if paths[a] and offers[a] > 1e-12]
    cleaned[-1] = min(
        float(cleaned[-1]), min((q[j] / offers[ids[j]] for j in eligible), default=0.0)
    )
    max_change = float(np.abs(cleaned[:n] - raw[:n]).sum())
    after = float((a_ub @ cleaned - b_ub).max(initial=0))
    if max_change > LP_MAX_REPAIR_MBPS or after > 1e-6:
        reject(f"repair too large: {max_change:.9g} Mbps, residual={after:.9g}")
    _lp_diagnostic(
        stage,
        ids,
        offers,
        paths,
        capacities,
        raw,
        cleaned,
        f"bounded roundoff; max_bound_error={bound_error:.9g}, total_flow_change={max_change:.9g} Mbps",
    )
    return cleaned, max(0.0, after)


def allocate_flows(offers: dict[int, float], paths: dict[int, list[int]], capacities: np.ndarray):
    """Lexicographic LP: maximum Mbps, then max-min offer fulfillment.

    q_a uses the same proportional scaling of UE rates within each Access as
    the legacy env. Global constraints: D q <= C t, sum(t)<=1. Fixed-route
    steady-state fluid capacity only, not packet delay or empty-buffer startup.
    """
    ids = sorted(offers)
    capacities = np.asarray(capacities, dtype=float)
    if capacities.ndim != 2 or any(not np.isfinite(offers[a]) or offers[a] < 0 for a in ids):
        raise ValueError("Invalid offers or capacity shape")
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
        row[n : n + states] = -capacities[e]
        constraints.append(row)
        rhs.append(0.0)
    row = np.zeros(size)
    row[n : n + states] = 1
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
    options = {
        "dual_feasibility_tolerance": LP_SOLVER_TOL,
        "primal_feasibility_tolerance": LP_SOLVER_TOL,
    }
    objective = np.zeros(size)
    objective[:n] = -1
    first = linprog(objective, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs", options=options)
    if not first.success:
        raise RuntimeError(first.message)
    first_x, _ = _clean_lp_solution(
        first.x, ids, offers, paths, capacities, demand, a_ub, b_ub, bounds, "max-throughput"
    )
    maximum_total = float(first_x[:n].sum())
    total_constraint = np.zeros(size)
    total_constraint[:n] = -1
    a_ub = np.vstack([a_ub, total_constraint])
    b_ub = np.append(b_ub, -max(0, maximum_total - 1e-8))
    objective[:] = 0
    objective[-1] = -1
    second = linprog(
        objective, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs", options=options
    )
    if not second.success:
        raise RuntimeError(second.message)
    result, violation = _clean_lp_solution(
        second.x, ids, offers, paths, capacities, demand, a_ub, b_ub, bounds, "max-min"
    )
    minimum = float(result[-1])
    return dict(zip(ids, result[:n])), result[n : n + states], minimum, violation


def audit_case(
    env,
    *,
    access_bw: float,
    backhaul_bw: float,
    spectrum_mode: str,
    backhaul_mode: str,
    efficiency: float,
):
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
        access_bw_label_mhz=access_bw,
        backhaul_total_bandwidth_mhz=backhaul_bw,
        spectrum_mode=spectrum_mode,
        backhaul_mode=backhaul_mode,
        rate_efficiency=efficiency,
        access_tx_dbm=env.cfg.access_tx_dbm,
        backhaul_tx_dbm=env.cfg.backhaul_tx_dbm,
        access_offer_mbps=totals,
        delivered_mbps=served,
        backhaul_loss_mbps=max(0, totals - served),
        utility=served / (env.cfg.num_ues * env.cfg.ue_demand_mbps),
        associated_ues=sum(len(users) for users in association.values()),
        positive_service_ues=int(np.sum(final > 1e-7)),
        qos_ues=int(np.sum(final >= env.cfg.ue_demand_mbps - 1e-6)),
        sinr_median_db=None if not len(sinr_present) else float(np.median(sinr_present)),
        reachable_access_uavs=sum(bool(paths[a]) and offers[a] > 0 for a in association),
        relay_routes=sum(len(path) == 2 for path in paths.values()),
        edge_count=len(edges),
        schedule_states=len(states),
        schedule_time_sum=float(durations.sum()),
        min_access_offer_fulfillment=minimum,
        lp_constraint_violation_mbps=violation,
        **spectrum,
    )
    detail = dict(
        association=association,
        paths=paths,
        access_offers=offers,
        access_flows=allocated,
        ue_access_rates=before.tolist(),
        ue_delivered_rates=final.tolist(),
        edges=[dict(source=e.source, target=e.target, solo_mbps=e.solo_mbps) for e in edges],
        schedules=[
            dict(
                active_edges=list(active),
                duration=float(durations[j]),
                capacity_mbps=matrix[:, j].tolist(),
            )
            for j, active in enumerate(states)
            if durations[j] > 1e-10
        ],
    )
    return metric, detail
