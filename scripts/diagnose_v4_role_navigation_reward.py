from __future__ import annotations

import argparse
import copy
import json
from collections import deque
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.envs.v0_env import ACCESS, RELAY, MOVE_DELTAS
from src.envs.v4_fixed_mobile_battery_env import ACTIVE, RETURNING
from src.envs.v4_final_fixed_map_env import V4FinalFixedMapConfig, V4FinalFixedMapEnv
from src.rl.v4_hardradius_spatial_qmix import V4HardRadiusSpatialQMIXLearner

ROLES_A0 = np.asarray([ACCESS, ACCESS, ACCESS, ACCESS, RELAY], dtype=np.int64)
REFERENCE_POSITIONS = np.asarray(
    [[5, 0], [6, 3], [14, 4], [9, 6], [5, 8]], dtype=np.int64
)
HORIZONS = (1, 5, 10, 20)


def make_env(layout_seed: int = 1101, backhaul_radius: float = 5.0) -> V4FinalFixedMapEnv:
    env = V4FinalFixedMapEnv(
        V4FinalFixedMapConfig(
            seed=layout_seed,
            enable_ue_mobility=False,
            enable_battery_cycle=False,
            backhaul_radius_cells=backhaul_radius,
        )
    )
    env.reset(seed=layout_seed)
    env.roles[:] = ROLES_A0
    return env


def fixed_role_mask(env: V4FinalFixedMapEnv) -> np.ndarray:
    base = env.get_action_mask()
    out = np.zeros_like(base)
    for i in range(env.cfg.num_uavs):
        for move in range(5):
            a = env.encode_action(move, int(ROLES_A0[i]))
            out[i, a] = base[i, a]
    return out


def collect_learned_snapshots(checkpoint: Path, times: tuple[int, ...]) -> dict[int, np.ndarray]:
    torch.set_num_threads(1)
    env = make_env()
    learner = V4HardRadiusSpatialQMIXLearner(
        env.cfg.num_uavs,
        env.obs_dim,
        env.state_dim,
        env.n_actions,
        grid_size=env.cfg.grid_size,
        coarse_size=env.cfg.coarse_obs_size,
        lr=3e-4,
        gamma=0.99,
        hidden_dim=64,
        mixing_embed_dim=32,
        device="cpu",
    )
    learner.load(str(checkpoint))
    rng = np.random.default_rng(12345)

    out = env.reset(seed=1101)
    env.roles[:] = ROLES_A0
    out["obs"] = env.get_obs()
    out["state"] = env.get_state()
    out["action_mask"] = fixed_role_mask(env)

    wanted = set(times)
    snapshots: dict[int, np.ndarray] = {}
    if 0 in wanted:
        snapshots[0] = env.uav_positions.copy()

    for t in range(1, max(times) + 1):
        actions = learner.act(out["obs"], out["action_mask"], 0.0, rng)
        out = env.step(actions)
        env.roles[:] = ROLES_A0
        out["obs"] = env.get_obs()
        out["state"] = env.get_state()
        out["action_mask"] = fixed_role_mask(env)
        if t in wanted:
            snapshots[t] = env.uav_positions.copy()
    return snapshots


def network_utility(env: V4FinalFixedMapEnv, positions: np.ndarray, roles: np.ndarray, dropped_id: int = 4) -> float:
    env.uav_positions[:] = positions
    env.roles[:] = roles
    env.status[:] = ACTIVE
    env.status[dropped_id] = RETURNING
    return float(env.compute_network()["utility"])


def best_relay_cell(
    env: V4FinalFixedMapEnv,
    positions: np.ndarray,
    candidate: int,
    dropped_id: int = 4,
) -> tuple[float, tuple[int, int]]:
    roles = np.full(env.cfg.num_uavs, ACCESS, dtype=np.int64)
    roles[candidate] = RELAY
    occupied = {
        tuple(map(int, positions[j]))
        for j in range(env.cfg.num_uavs)
        if j not in (candidate, dropped_id) and env.status[j] == ACTIVE
    }
    best_u = -1.0
    best_cell = tuple(map(int, positions[candidate]))
    trial = positions.copy()
    radius_m = env.cfg.backhaul_radius_cells * env.cfg.cell_size_m
    access_positions = [positions[j].copy() for j in range(env.cfg.num_uavs) if j not in (candidate, dropped_id)]
    for r in range(env.cfg.grid_size):
        for c in range(env.cfg.grid_size):
            if (r, c) in occupied:
                continue
            cell = np.asarray([r, c], dtype=np.int64)
            xy = env._cell_to_xy(cell)
            # A useful one-hop Relay must reach at least one surviving BS and
            # at least one remaining Access. Skip cells that cannot do either.
            if min(np.linalg.norm(xy - bs) for bs in env.surviving_bs_xy) > radius_m + 1e-9:
                continue
            if min(np.linalg.norm(cell.astype(float) - a.astype(float)) for a in access_positions) > env.cfg.backhaul_radius_cells + 1e-9:
                continue
            trial[candidate] = cell
            u = network_utility(env, trial, roles, dropped_id)
            if u > best_u + 1e-12:
                best_u = u
                best_cell = (r, c)
    return float(best_u), best_cell


def shortest_path(start: tuple[int, int], goal: tuple[int, int], blocked: set[tuple[int, int]], grid_size: int) -> list[tuple[int, int]]:
    if start == goal:
        return [start]
    q = deque([start])
    prev = {start: None}
    # MOVE_DELTAS order is N,S,E,W,Stay; ignore Stay.
    steps = [tuple(map(int, x)) for x in MOVE_DELTAS[:4]]
    while q:
        cur = q.popleft()
        for dr, dc in steps:
            nxt = (cur[0] + dr, cur[1] + dc)
            if not (0 <= nxt[0] < grid_size and 0 <= nxt[1] < grid_size):
                continue
            if nxt in blocked or nxt in prev:
                continue
            prev[nxt] = cur
            if nxt == goal:
                path = [nxt]
                p = cur
                while p is not None:
                    path.append(p)
                    p = prev[p]
                return list(reversed(path))
            q.append(nxt)
    raise RuntimeError(f"No path from {start} to {goal}")


def path_rewards(
    env: V4FinalFixedMapEnv,
    positions: np.ndarray,
    candidate: int,
    target: tuple[int, int],
    dropped_id: int = 4,
) -> dict:
    blocked = {
        tuple(map(int, positions[j]))
        for j in range(env.cfg.num_uavs)
        if j not in (candidate, dropped_id)
    }
    start = tuple(map(int, positions[candidate]))
    blocked.discard(start)
    blocked.discard(target)
    path = shortest_path(start, target, blocked, env.cfg.grid_size)

    # Baseline: keep all remaining UAVs Access and fixed.
    all_access = np.full(env.cfg.num_uavs, ACCESS, dtype=np.int64)
    base_u = network_utility(env, positions.copy(), all_access, dropped_id)

    # Immediate switch: candidate becomes Relay at t=1 and moves one cell/slot.
    immediate = []
    # Delayed switch: candidate travels as Access; it becomes Relay only after arrival.
    delayed = []
    pos_i = positions.copy()
    for step in range(1, max(HORIZONS) + 1):
        path_idx = min(step, len(path) - 1)
        pos_i[candidate] = path[path_idx]

        roles_now = np.full(env.cfg.num_uavs, ACCESS, dtype=np.int64)
        roles_now[candidate] = RELAY
        immediate.append(network_utility(env, pos_i, roles_now, dropped_id))

        delayed_roles = np.full(env.cfg.num_uavs, ACCESS, dtype=np.int64)
        if path_idx == len(path) - 1:
            delayed_roles[candidate] = RELAY
        delayed.append(network_utility(env, pos_i, delayed_roles, dropped_id))

    gamma = 0.99
    horizon_rows = {}
    for h in HORIZONS:
        weights = np.power(gamma, np.arange(h, dtype=np.float64))
        base_return = float(np.sum(weights * base_u))
        imm_return = float(np.sum(weights * np.asarray(immediate[:h])))
        delay_return = float(np.sum(weights * np.asarray(delayed[:h])))
        horizon_rows[str(h)] = {
            "baseline_mean": base_u,
            "immediate_switch_mean": float(np.mean(immediate[:h])),
            "delayed_switch_mean": float(np.mean(delayed[:h])),
            "baseline_discounted_return": base_return,
            "immediate_switch_discounted_return": imm_return,
            "delayed_switch_discounted_return": delay_return,
            "immediate_minus_baseline_return": imm_return - base_return,
            "delayed_minus_baseline_return": delay_return - base_return,
        }

    return {
        "start": list(start),
        "target": list(target),
        "path_length_moves": len(path) - 1,
        "path": [list(x) for x in path],
        "baseline_4A_utility": base_u,
        "immediate_switch_rewards": immediate,
        "delayed_switch_rewards": delayed,
        "horizons": horizon_rows,
    }



def optimal_relay_navigation_dp(
    env: V4FinalFixedMapEnv,
    positions: np.ndarray,
    candidate: int,
    dropped_id: int = 4,
    horizons: tuple[int, ...] = HORIZONS,
    gamma: float = 0.99,
) -> dict:
    """Exact low-level navigation upper bound with role fixed to Relay.

    Other active UAVs are frozen Accesses. The candidate becomes Relay
    immediately and may move N/S/E/W/Stay one grid cell per slot.
    Dynamic programming finds the path maximizing discounted team utility.
    """
    g = env.cfg.grid_size
    roles = np.full(env.cfg.num_uavs, ACCESS, dtype=np.int64)
    roles[candidate] = RELAY
    blocked = {
        tuple(map(int, positions[j]))
        for j in range(env.cfg.num_uavs)
        if j not in (candidate, dropped_id)
    }
    reward = np.full((g, g), -np.inf, dtype=np.float64)
    trial = positions.copy()
    for r in range(g):
        for c in range(g):
            if (r, c) in blocked:
                continue
            trial[candidate] = (r, c)
            reward[r, c] = network_utility(env, trial, roles, dropped_id)

    moves = [tuple(map(int, x)) for x in MOVE_DELTAS]
    start = tuple(map(int, positions[candidate]))
    base_roles = np.full(env.cfg.num_uavs, ACCESS, dtype=np.int64)
    base_u = network_utility(env, positions.copy(), base_roles, dropped_id)

    out = {}
    for H in horizons:
        # V[k,r,c] = best return over k remaining actions from current cell.
        V = np.zeros((H + 1, g, g), dtype=np.float64)
        policy = np.full((H + 1, g, g, 2), -1, dtype=np.int16)
        for k in range(1, H + 1):
            for r in range(g):
                for c in range(g):
                    if (r, c) in blocked:
                        V[k, r, c] = -1e9
                        continue
                    best = -1e18
                    bestn = (r, c)
                    for dr, dc in moves:
                        nr, nc = r + dr, c + dc
                        if not (0 <= nr < g and 0 <= nc < g):
                            continue
                        if (nr, nc) in blocked or not np.isfinite(reward[nr, nc]):
                            continue
                        val = reward[nr, nc] + gamma * V[k - 1, nr, nc]
                        if val > best:
                            best = val
                            bestn = (nr, nc)
                    V[k, r, c] = best
                    policy[k, r, c] = bestn

        path = [list(start)]
        cur = start
        rewards = []
        for k in range(H, 0, -1):
            nr, nc = map(int, policy[k, cur[0], cur[1]])
            cur = (nr, nc)
            path.append([nr, nc])
            rewards.append(float(reward[nr, nc]))
        weights = np.power(gamma, np.arange(H, dtype=np.float64))
        ret = float(np.sum(weights * np.asarray(rewards)))
        base_ret = float(np.sum(weights * base_u))
        out[str(H)] = {
            "optimal_discounted_return": ret,
            "baseline_discounted_return": base_ret,
            "delta_vs_4A": ret - base_ret,
            "mean_utility": float(np.mean(rewards)),
            "path": path,
            "rewards": rewards,
        }
    return out

def diagnose_state(
    env: V4FinalFixedMapEnv,
    name: str,
    positions: np.ndarray,
    dropped_id: int = 4,
) -> dict:
    positions = np.asarray(positions, dtype=np.int64).copy()
    env.status[:] = ACTIVE
    env.status[dropped_id] = RETURNING

    roles_4a = np.full(env.cfg.num_uavs, ACCESS, dtype=np.int64)
    base = network_utility(env, positions, roles_4a, dropped_id)
    candidates = []
    for candidate in range(env.cfg.num_uavs):
        if candidate == dropped_id:
            continue
        roles = roles_4a.copy()
        roles[candidate] = RELAY
        stay = network_utility(env, positions.copy(), roles, dropped_id)
        best_u, target = best_relay_cell(env, positions.copy(), candidate, dropped_id)
        path = path_rewards(env, positions.copy(), candidate, target, dropped_id)
        candidates.append(
            {
                "candidate": candidate,
                "stay_relay_utility": stay,
                "stay_switch_cost_pp": (stay - base) * 100.0,
                "oracle_target_utility": best_u,
                "oracle_gain_vs_4A_pp": (best_u - base) * 100.0,
                **path,
            }
        )

    best_candidate = max(candidates, key=lambda x: x["oracle_target_utility"])
    optimal_nav = optimal_relay_navigation_dp(
        env, positions.copy(), int(best_candidate["candidate"]), dropped_id
    )
    return {
        "name": name,
        "positions": positions.astype(int).tolist(),
        "dropped_uav": dropped_id,
        "baseline_4A_utility": base,
        "best_candidate": best_candidate["candidate"],
        "best_oracle_gain_vs_4A_pp": best_candidate["oracle_gain_vs_4A_pp"],
        "best_candidate_optimal_relay_navigation": optimal_nav,
        "candidates": candidates,
    }


def summarize(results: list[dict]) -> dict:
    useful = []
    switch_costs = []
    best_paths = []
    immediate_h = {str(h): [] for h in HORIZONS}
    delayed_h = {str(h): [] for h in HORIZONS}

    for state in results:
        best = max(state["candidates"], key=lambda x: x["oracle_target_utility"])
        useful.append(best["oracle_gain_vs_4A_pp"])
        switch_costs.append(best["stay_switch_cost_pp"])
        best_paths.append(best["path_length_moves"])
        for h in HORIZONS:
            row = best["horizons"][str(h)]
            immediate_h[str(h)].append(row["immediate_minus_baseline_return"])
            delayed_h[str(h)].append(row["delayed_minus_baseline_return"])

    def stats(xs):
        xs = np.asarray(xs, dtype=np.float64)
        return {"mean": float(xs.mean()), "min": float(xs.min()), "max": float(xs.max())}

    return {
        "states": len(results),
        "best_relay_oracle_gain_pp": stats(useful),
        "immediate_role_switch_cost_pp": stats(switch_costs),
        "best_relay_path_moves": stats(best_paths),
        "discounted_return_delta_immediate_switch": {k: stats(v) for k, v in immediate_h.items()},
        "discounted_return_delta_move_as_access_then_switch": {k: stats(v) for k, v in delayed_h.items()},
        "states_where_oracle_relay_beats_4A": int(sum(x > 1e-9 for x in useful)),
    }


def recommendation(summary: dict) -> dict:
    gain = summary["best_relay_oracle_gain_pp"]["mean"]
    cost = summary["immediate_role_switch_cost_pp"]["mean"]
    imm20 = summary["discounted_return_delta_immediate_switch"]["20"]["mean"]
    delayed20 = summary["discounted_return_delta_move_as_access_then_switch"]["20"]["mean"]
    useful_n = summary["states_where_oracle_relay_beats_4A"]
    n = summary["states"]

    if useful_n == 0:
        verdict = "NO_ROLE_NEED_IN_AUDITED_STATES"
        action = "Do not redesign around Relay assistance from these states; role replacement is not beneficial here."
    elif cost < -1.0 and gain > 1.0 and imm20 < 0.0:
        verdict = "ROLE_USEFUL_BUT_DELAYED_CREDIT_HARD"
        action = (
            "Keep pure utility reward, but separate the high-level Relay request/assignment from low-level Relay navigation. "
            "A RequestRelay/helper design is justified more strongly than adding role-specific reward shaping."
        )
    elif gain > 1.0 and imm20 >= 0.0:
        verdict = "ROLE_AND_NAVIGATION_LEARNABLE_WITH_CURRENT_REWARD_HORIZON"
        action = "Keep 10 actions first; the reward horizon does not hide the long-term Relay benefit strongly."
    else:
        verdict = "MIXED"
        action = "Use state-specific ablations before changing action space or reward."

    return {
        "verdict": verdict,
        "action": action,
        "mean_oracle_role_gain_pp": gain,
        "mean_immediate_switch_cost_pp": cost,
        "mean_immediate_20_return_delta": imm20,
        "mean_delayed_20_return_delta": delayed20,
        "useful_states": f"{useful_n}/{n}",
        "reward_note": (
            "The team utility objective is semantically correct: it already values a Relay when the Relay improves UE rates. "
            "A separate positive Relay bonus is not recommended unless a later audit proves the objective itself is misaligned."
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=ROOT.parent / "results_current" / "seed81_best.pt")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "v4_role_navigation_reward_diagnostic.json")
    ap.add_argument("--backhaul-radius", type=float, default=5.0)
    args = ap.parse_args()

    env = make_env(backhaul_radius=args.backhaul_radius)
    snapshots = collect_learned_snapshots(args.checkpoint, (4, 20))

    states = [("reference_target", REFERENCE_POSITIONS.copy())]
    states.extend((f"learned_t{t}", p) for t, p in sorted(snapshots.items()))

    results = [diagnose_state(env, name, positions) for name, positions in states]
    summary = summarize(results)
    rec = recommendation(summary)
    payload = {
        "protocol": {
            "layout_seed": 1101,
            "ue_mobility": False,
            "battery_dynamics_during_branch": False,
            "backhaul_radius_cells": args.backhaul_radius,
            "dropped_role": "original UAV4 Relay is marked unavailable",
            "baseline": "remaining four UAVs keep Access roles and positions fixed",
            "stay_relay": "one Access changes to Relay at its current position",
            "oracle_assisted": "same replacement Relay moves one grid cell/slot along a shortest path to its best static Relay cell",
            "reward_horizons": list(HORIZONS),
            "gamma": 0.99,
            "purpose": "separate role usefulness, Relay navigation difficulty, and delayed team-utility credit",
        },
        "summary": summary,
        "recommendation": rec,
        "states": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "recommendation": rec}, indent=2))
    print(f"WROTE {args.out}")


if __name__ == "__main__":
    main()
