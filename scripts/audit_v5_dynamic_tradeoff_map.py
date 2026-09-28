"""Necessity audit for the single fixed V5 map.

Question: can one *fixed physical topology* naturally require different role
compositions over time as UE groups move, while movement itself remains useful?

Battery is disabled in this audit to isolate the traffic/geometry effect.  At
sampled times on the same UE-mobility trajectory we approximately optimize UAV
positions under three frozen role counts: 5A, 4A+1R, 3A+2R.  We also evaluate
the t=0 deployment without moving it, so the gap to the current re-optimized
placement measures mobility-driven repositioning value for the same role count.

This is an environment/design audit, not learned-policy performance.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time
from collections import Counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.envs.v0_env import ACCESS, RELAY
from src.envs.v4_fixed_mobile_battery_env import ACTIVE
from src.envs.v5_dynamic_tradeoff_env import V5DynamicTradeoffConfig, V5DynamicTradeoffEnv

ROLE_SPECS = {"5A": 0, "4A1R": 1, "3A2R": 2}


def _inside(env, p):
    p = np.asarray(p, dtype=np.int64)
    return bool(np.all(p >= 0) and np.all(p < env.cfg.grid_size))


def _around(center: np.ndarray, radius: int = 2) -> list[np.ndarray]:
    c = np.rint(center).astype(np.int64)
    out = []
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            out.append(c + np.asarray([dr, dc], dtype=np.int64))
    return out


def _access_candidates(env) -> np.ndarray:
    cells: list[np.ndarray] = []
    for center in env.hotspot_centers:
        cells.extend(_around(center, 2))
    # Sparse failed-region support prevents the optimizer getting trapped only
    # at the three current center cells.
    stride = max(1, len(env.affected_cells) // 18)
    cells.extend(np.asarray(x, dtype=np.int64) for x in env.affected_cells[::stride][:18])
    cells.extend(np.asarray(x, dtype=np.int64) for x in env.start_positions)
    arr = np.asarray([x for x in cells if _inside(env, x)], dtype=np.int64)
    return np.unique(arr, axis=0)


def _relay_candidates(env, access_candidates: np.ndarray) -> np.ndarray:
    # A useful one-hop bridge must have a feasible second hop to some surviving
    # BS and be close enough to at least one plausible Access location.
    scored: list[tuple[float, np.ndarray]] = []
    centers = np.asarray(env.hotspot_centers, dtype=np.float64)
    for r in range(env.cfg.grid_size):
        for c in range(env.cfg.grid_size):
            cell = np.asarray([r, c], dtype=np.int64)
            xy = env._cell_to_xy(cell)
            if not any(
                env._xy_within_radius_m(xy, bsxy, env.cfg.backhaul_radius_cells)
                for bsxy in env.surviving_bs_xy
            ):
                continue
            # Prefer bridge cells near traffic but retain multiple gateway sides.
            d_center = float(np.min(np.linalg.norm(centers - cell[None, :], axis=1)))
            d_access = float(np.min(np.linalg.norm(access_candidates - cell[None, :], axis=1)))
            scored.append((0.7 * d_center + 0.3 * d_access, cell))
    scored.sort(key=lambda z: z[0])
    keep = [x for _, x in scored[:36]]
    keep.extend(np.asarray(x, dtype=np.int64) for x in env.surviving_bs_cells)
    arr = np.asarray([x for x in keep if _inside(env, x)], dtype=np.int64)
    return np.unique(arr, axis=0)


def _initial_positions(env, n_relays: int) -> np.ndarray:
    n_access = env.cfg.num_uavs - n_relays
    anchors = np.rint(env.hotspot_centers).astype(np.int64)
    pos = env.start_positions.copy()
    offsets = (
        np.asarray([0, 0]), np.asarray([0, 1]), np.asarray([1, 0]),
        np.asarray([0, -1]), np.asarray([-1, 0]),
    )
    for i in range(n_access):
        p = anchors[i % len(anchors)] + offsets[(i // len(anchors)) % len(offsets)]
        pos[i] = np.clip(p, 0, env.cfg.grid_size - 1)

    if n_relays:
        ac = _access_candidates(env)
        rc = _relay_candidates(env, ac)
        used = {tuple(pos[i]) for i in range(n_access)}
        chosen = []
        for cell in rc:
            if tuple(cell) in used:
                continue
            if any(np.linalg.norm(cell - x) < 2.0 for x in chosen):
                continue
            chosen.append(cell.copy())
            if len(chosen) == n_relays:
                break
        if len(chosen) < n_relays:
            raise RuntimeError("not enough relay bridge candidates")
        for k, cell in enumerate(chosen):
            pos[n_access + k] = cell
    return pos


def optimize_composition(
    env: V5DynamicTradeoffEnv,
    n_relays: int,
    start_positions: np.ndarray | None = None,
    sweeps: int = 1,
) -> tuple[float, np.ndarray]:
    roles = np.asarray([ACCESS] * (env.cfg.num_uavs - n_relays) + [RELAY] * n_relays, dtype=np.int64)
    env.roles[:] = roles
    env.status[:] = ACTIVE
    pos = _initial_positions(env, n_relays) if start_positions is None else np.asarray(start_positions, dtype=np.int64).copy()
    env.uav_positions[:] = pos

    access_candidates = _access_candidates(env)
    relay_candidates = _relay_candidates(env, access_candidates)

    for _ in range(max(1, sweeps)):
        changed = False
        for i in range(env.cfg.num_uavs):
            current = pos[i].copy()
            local = [x for x in _around(current, 1) if _inside(env, x)]
            global_candidates = access_candidates if roles[i] == ACCESS else relay_candidates
            search = np.unique(np.vstack([global_candidates, np.asarray(local, dtype=np.int64)]), axis=0)
            occupied = {tuple(pos[j]) for j in range(env.cfg.num_uavs) if j != i}
            best_u = -1.0
            best_cell = current.copy()
            for cell in search:
                if tuple(cell) in occupied:
                    continue
                env.uav_positions[i] = cell
                u = float(env.compute_network()["utility"])
                if u > best_u + 1e-12:
                    best_u = u
                    best_cell = cell.copy()
            env.uav_positions[i] = best_cell
            pos[i] = best_cell
            changed |= not np.array_equal(best_cell, current)
        if not changed:
            break

    return float(env.compute_network()["utility"]), pos.copy()


def utility_at_positions(env, n_relays: int, positions: np.ndarray) -> float:
    env = copy.deepcopy(env)
    env.status[:] = ACTIVE
    env.roles[:] = np.asarray([ACCESS] * (env.cfg.num_uavs - n_relays) + [RELAY] * n_relays, dtype=np.int64)
    env.uav_positions[:] = np.asarray(positions, dtype=np.int64)
    return float(env.compute_network()["utility"])


def run_seed(seed: int, horizon: int, checkpoint_every: int, sweeps: int, backhaul_radius: float = 4.0) -> dict:
    cfg = V5DynamicTradeoffConfig(
        horizon=horizon + 1,
        enable_ue_mobility=True,
        enable_battery_cycle=False,
        backhaul_radius_cells=float(backhaul_radius),
        randomize_start_identity=False,
        randomize_tradeoff_phase=False,
    )
    base = V5DynamicTradeoffEnv(cfg)
    base.reset(seed=seed)

    warm_positions: dict[str, np.ndarray | None] = {k: None for k in ROLE_SPECS}
    frozen_t0: dict[str, np.ndarray] = {}
    rows = []
    for t in range(horizon + 1):
        if t % checkpoint_every == 0:
            role_rows = {}
            for role_name, n_relays in ROLE_SPECS.items():
                e = copy.deepcopy(base)
                u, p = optimize_composition(e, n_relays, warm_positions[role_name], sweeps=sweeps)
                warm_positions[role_name] = p
                if role_name not in frozen_t0:
                    frozen_t0[role_name] = p.copy()
                frozen_u = utility_at_positions(base, n_relays, frozen_t0[role_name])
                role_rows[role_name] = {
                    "utility": float(u),
                    "frozen_t0_utility": float(frozen_u),
                    "movement_gain_pp": float(100.0 * (u - frozen_u)),
                    "positions": p.astype(int).tolist(),
                }
            ordered = sorted(role_rows, key=lambda x: role_rows[x]["utility"], reverse=True)
            winner, runner_up = ordered[0], ordered[1]
            rows.append({
                "t": int(t),
                "hotspot_centers": np.asarray(base.hotspot_centers).astype(float).tolist(),
                "roles": role_rows,
                "winner": winner,
                "winner_margin_pp": float(100.0 * (role_rows[winner]["utility"] - role_rows[runner_up]["utility"])),
            })
        if t < horizon:
            base.step([4] * base.cfg.num_uavs)

    clear_rows = [r for r in rows if r["winner_margin_pp"] >= 2.0]
    clear_counts = Counter(r["winner"] for r in clear_rows)
    all_counts = Counter(r["winner"] for r in rows)
    pairwise_5a_minus_4a1r_pp = [
        float(100.0 * (r["roles"]["5A"]["utility"] - r["roles"]["4A1R"]["utility"]))
        for r in rows
    ]
    late = rows[1:] if len(rows) > 1 else rows
    move_stats = {}
    for role in ROLE_SPECS:
        vals = [r["roles"][role]["movement_gain_pp"] for r in late]
        move_stats[role] = {
            "mean_gain_pp": float(np.mean(vals)) if vals else 0.0,
            "max_gain_pp": float(np.max(vals)) if vals else 0.0,
            "positive_fraction": float(np.mean(np.asarray(vals) > 1e-9)) if vals else 0.0,
        }
    return {
        "seed": int(seed),
        "rows": rows,
        "winner_counts": dict(all_counts),
        "clear_winner_counts_margin_ge_2pp": dict(clear_counts),
        "distinct_clear_winners": sorted(clear_counts),
        "pairwise_5a_minus_4a1r_pp": pairwise_5a_minus_4a1r_pp,
        "pairwise_sign_flip_5A_vs_4A1R": bool(
            any(x > 1.0 for x in pairwise_5a_minus_4a1r_pp)
            and any(x < -1.0 for x in pairwise_5a_minus_4a1r_pp)
        ),
        "movement_stats_vs_t0_frozen": move_stats,
    }


def aggregate(results: list[dict]) -> dict:
    winner_counts = Counter()
    clear_counts = Counter()
    seeds_with = {k: 0 for k in ROLE_SPECS}
    for x in results:
        winner_counts.update(x["winner_counts"])
        clear_counts.update(x["clear_winner_counts_margin_ge_2pp"])
        for role in x["distinct_clear_winners"]:
            seeds_with[role] += 1
    move = {}
    for role in ROLE_SPECS:
        vals = [x["movement_stats_vs_t0_frozen"][role]["mean_gain_pp"] for x in results]
        move[role] = {
            "mean_of_seed_means_pp": float(np.mean(vals)),
            "std_of_seed_means_pp": float(np.std(vals)),
        }
    sign_flip_fraction = float(np.mean([x["pairwise_sign_flip_5A_vs_4A1R"] for x in results])) if results else 0.0
    return {
        "winner_counts": dict(winner_counts),
        "clear_winner_counts_margin_ge_2pp": dict(clear_counts),
        "seeds_with_clear_winner": seeds_with,
        "pairwise_5A_vs_4A1R_sign_flip_fraction": sign_flip_fraction,
        "movement_gain": move,
        "criteria": {
            "role_tradeoff_supported": bool(sign_flip_fraction >= 0.8),
            "all_three_roles_observed_clearly": bool(all(clear_counts.get(k, 0) > 0 for k in ROLE_SPECS)),
            "movement_supported": bool(max(v["mean_of_seed_means_pp"] for v in move.values()) >= 5.0),
        },
    }


def write_md(result: dict, path: Path) -> None:
    a = result["aggregate"]
    lines = [
        "# V5 Single-Map Dynamic Trade-off Audit",
        "",
        "Environment/design audit only; this is not learned-policy performance.",
        "The BS topology is fixed. Battery is OFF here to isolate UE-mobility + role/movement necessity.",
        "",
        f"Seeds: {result['seeds']}",
        f"Winner counts: `{a['winner_counts']}`",
        f"Clear winner counts (margin >= 2 pp): `{a['clear_winner_counts_margin_ge_2pp']}`",
        f"Seeds with each clear winner: `{a['seeds_with_clear_winner']}`",
        f"5A vs 4A1R sign-flip fraction across seeds: **{100*a['pairwise_5A_vs_4A1R_sign_flip_fraction']:.0f}%**",
        "",
        "## Movement gain versus the t=0 frozen deployment (same role composition)",
        "",
    ]
    for role, st in a["movement_gain"].items():
        lines.append(f"- {role}: {st['mean_of_seed_means_pp']:+.2f} ± {st['std_of_seed_means_pp']:.2f} pp")
    lines += [
        "",
        "## Design criteria",
        "",
        f"- Role trade-off supported: **{a['criteria']['role_tradeoff_supported']}**",
        f"- All 5A / 4A1R / 3A2R observed as clear winners: **{a['criteria']['all_three_roles_observed_clearly']}**",
        f"- Movement supported (>=5 pp mean gain for at least one role): **{a['criteria']['movement_supported']}**",
        "",
        "Interpretation: if these criteria hold, one fixed map already supplies the required dynamic regimes; do not create separate maps merely to force each role composition to win.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[1101, 1102, 1103])
    ap.add_argument("--horizon", type=int, default=480)
    ap.add_argument("--checkpoint-every", type=int, default=160)
    ap.add_argument("--sweeps", type=int, default=1)
    ap.add_argument("--backhaul-radius", type=float, default=4.0)
    ap.add_argument("--outdir", type=Path, default=ROOT / "results" / "v5_dynamic_tradeoff_audit")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    per_seed = []
    for seed in args.seeds:
        print(f"AUDIT seed={seed}", flush=True)
        r = run_seed(seed, args.horizon, args.checkpoint_every, args.sweeps, args.backhaul_radius)
        per_seed.append(r)
        print(json.dumps({
            "seed": seed,
            "winner_counts": r["winner_counts"],
            "clear": r["clear_winner_counts_margin_ge_2pp"],
            "movement": r["movement_stats_vs_t0_frozen"],
        }), flush=True)

    result = {
        "audit_type": "single fixed topology: dynamic role + movement necessity",
        "method": "UE mobility ON; battery OFF; 5A/4A1R/3A2R roles frozen within each counterfactual; approximate structured coordinate placement; compare to t0 frozen placement.",
        "seeds": [int(x) for x in args.seeds],
        "horizon": int(args.horizon),
        "checkpoint_every": int(args.checkpoint_every),
        "backhaul_radius_cells": float(args.backhaul_radius),
        "per_seed": per_seed,
        "aggregate": aggregate(per_seed),
        "runtime_s": float(time.time() - t0),
    }
    (args.outdir / "dynamic_tradeoff_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_md(result, args.outdir / "DYNAMIC_TRADEOFF_AUDIT.md")
    print(json.dumps(result["aggregate"], indent=2), flush=True)
    print(f"runtime_s={result['runtime_s']:.1f}", flush=True)


if __name__ == "__main__":
    main()
