"""Read-only R4 causal bottleneck replay; no training, no checkpoints modified.

Replays existing policies on the identical 3102/3106/3117 scenario seeds and
captures PRE-MOBILITY network snapshots. Audits geometric Access coverage,
association, access-side offered rates, backhaul scaling and actual routes.

An optional same-geometry exhaustive ROLE ORACLE (not an online baseline) changes
role bits in memory ONLY on audited slots, holds all UAV/UE positions, active
statuses and demand fixed, and calls the exact deterministic radio scorer.
The original simulation state is restored before the next step.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluation.eval_relay_stress_r4_qmix import (
    direct_access_cells,
    make_env,
    make_learner,
    relay_required_ue_mask,
)
from src.envs.v0_env import ACCESS, RELAY
from src.envs.v4_fixed_mobile_battery_env import ACTIVE
from src.envs.v4_final_fixed_map_env import V4FinalFixedMapEnv

EPS = 1e-9
PATHS = {
    "original": "original_reward",
    "lam0p10": "a5_relay_required_lam0p10",
    "lam0p30": "a5_relay_required_lam0p30",
    "lam1p00": "a5_relay_required_lam1p00",
}


def checkpoint_path(name: str, seed: int) -> Path:
    return (
        ROOT / "results" / "relay_stress_r4_qmix" / PATHS[name] / f"seed{seed}_best_final_block.pt"
    )


def radio_offer(env, snap):
    """Reconstruct per-Access pre-backhaul throughput from exact associations."""
    ans = {}
    for a, users in snap["association"].items():
        users = list(users)
        if not users:
            ans[int(a)] = 0.0
            continue
        n = len(users)
        bw = env.cfg.access_bw_hz / n
        tx = env.cfg.access_tx_dbm - 10.0 * math.log10(n)
        uav_xy = env._cell_to_xy(snap["uav_positions"][int(a)])
        offer = 0.0
        for u in users:
            ue_xy = env._cell_to_xy(snap["ue_positions"][int(u)])
            pl = env._a2g_path_loss_db(
                ue_xy,
                env.cfg.ue_height_m,
                uav_xy,
                env.cfg.uav_altitude_m,
                env.cfg.access_fc_hz,
            )
            rate = env._shannon_rate_mbps(tx - pl, bw)
            offer += min(rate, env.cfg.ue_demand_mbps)
        ans[int(a)] = offer
    return ans


def backhaul_hop_breakdown(env, snap, offer):
    """Recreate one-hop relay second-hop shares as implemented in V4 scorer.

    Sequential loss decomposition: access offer -> A2A first hop -> shared
    Relay-to-BS second hop. Direct Access uses direct-to-BS loss only.
    The decomposition is descriptive and exact for current routing, NOT a
    causal statement about a counterfactual path without interference.
    """
    info = {}
    caps = snap["backhaul_caps_mbps"]
    routes = snap["routes"]
    relay_ids = set(rid for rid in routes.values() if rid is not None)
    relay_bs_capacity = {}
    first_capacity = {}
    for rid in relay_ids:
        rxy = env._cell_to_xy(snap["uav_positions"][rid])
        candidates = [
            env._bh_a2g_rate(rxy, env.cfg.uav_altitude_m, bsxy, env.cfg.bs_height_m)
            for bsxy in env.surviving_bs_xy
            if env._xy_within_radius_m(rxy, bsxy, env.cfg.backhaul_radius_cells)
        ]
        relay_bs_capacity[rid] = max(candidates) if candidates else 0.0
    for aid, rid in routes.items():
        if rid is None:
            continue
        axy = env._cell_to_xy(snap["uav_positions"][aid])
        rxy = env._cell_to_xy(snap["uav_positions"][rid])
        first_capacity[aid] = env._bh_a2a_rate(axy, rxy)
    for aid, rid in routes.items():
        offered = offer.get(aid, 0.0)
        assigned = caps.get(aid, 0.0)
        if rid is None:
            info[aid] = {
                "a2a_cap_mbps": "",
                "relay_bs_total_cap_mbps": "",
                "relay_bs_share_mbps": "",
                "direct_bs_loss_mbps": max(0.0, offered - assigned),
                "a2a_loss_mbps": 0.0,
                "relay_bs_loss_mbps": 0.0,
            }
            continue
        clients = [a for a, linked_rid in routes.items() if linked_rid == rid]
        weights = {a: max(1e-6, min(offer.get(a, 0.0), first_capacity[a])) for a in clients}
        second_share = relay_bs_capacity[rid] * weights[aid] / sum(weights.values())
        first_limited = min(offered, first_capacity[aid])
        info[aid] = {
            "a2a_cap_mbps": float(first_capacity[aid]),
            "relay_bs_total_cap_mbps": float(relay_bs_capacity[rid]),
            "relay_bs_share_mbps": float(second_share),
            "direct_bs_loss_mbps": 0.0,
            "a2a_loss_mbps": max(0.0, offered - first_limited),
            "relay_bs_loss_mbps": max(0.0, first_limited - min(first_limited, second_share)),
        }
        if abs(min(first_capacity[aid], second_share) - assigned) > 1e-6:
            raise AssertionError("Hop-share reconstruction != network scorer")
    return info


def slot_metrics(env, snap, direct_cells):
    rates = snap["ue_rate_mbps"]
    demand = env.cfg.ue_demand_mbps
    roles, status = snap["roles"], snap["status"]
    active_access_ids = np.flatnonzero((roles == ACCESS) & (status == ACTIVE))
    active_relay_ids = np.flatnonzero((roles == RELAY) & (status == ACTIVE))
    required = relay_required_ue_mask(env, snap["ue_positions"], direct_cells)
    covered = np.zeros(env.cfg.num_ues, dtype=bool)
    for aid in active_access_ids:
        distance = np.linalg.norm(snap["ue_positions"] - snap["uav_positions"][aid], axis=1)
        covered |= distance <= env.cfg.access_radius_cells + EPS
    assigned = np.zeros(env.cfg.num_ues, dtype=bool)
    relay_users = np.zeros(env.cfg.num_ues, dtype=bool)
    for aid, users in snap["association"].items():
        assigned[np.asarray(users, dtype=np.int64)] = True
        if snap["routes"].get(int(aid)) is not None:
            relay_users[np.asarray(users, dtype=np.int64)] = True
    offer = radio_offer(env, snap)
    total_offer = sum(offer.values())
    hop_losses = backhaul_hop_breakdown(env, snap, offer)
    caps = snap["backhaul_caps_mbps"]
    bh_loss = sum(max(0.0, v - max(0.0, caps.get(k, 0.0))) for k, v in offer.items())
    no_bh = [aid for aid, v in offer.items() if v > EPS and caps.get(aid, 0.0) <= EPS]
    limited = [aid for aid, v in offer.items() if v > caps.get(aid, 0.0) + EPS]
    routes = snap["routes"]
    direct_ues = assigned & ~relay_users & (rates > EPS)
    relay_ues = assigned & relay_users & (rates > EPS)
    req_served = required & (rates > EPS)
    req_fraction = (
        np.sum(np.clip(rates[required] / demand, 0, 1)) / int(required.sum())
        if required.any()
        else 0.0
    )
    any_deep = 0
    deep_count = 0
    for aid in active_access_ids:
        p = env._cell_to_xy(snap["uav_positions"][aid])
        dist = min(np.linalg.norm(p - bs) for bs in env.surviving_bs_xy)
        if dist > env.cfg.backhaul_radius_cells * env.cfg.cell_size_m + EPS:
            deep_count += 1
    any_deep = int(deep_count > 0)
    return (
        {
            "utility": float(snap["utility"]),
            "served_ues": int((rates > EPS).sum()),
            "full_ues": int((rates >= demand - EPS).sum()),
            "required_ues": int(required.sum()),
            "required_covered": int((required & covered).sum()),
            "required_associated": int((required & assigned).sum()),
            "required_served": int(req_served.sum()),
            "required_score": float(req_fraction),
            "required_uncovered": int((required & ~covered).sum()),
            "required_associated_zero_rate": int((required & assigned & (rates <= EPS)).sum()),
            "uncovered_ues": int((~covered).sum()),
            "covered_unassociated_ues": int((covered & ~assigned).sum()),
            "associated_zero_rate_ues": int((assigned & (rates <= EPS)).sum()),
            "direct_served_ues": int(direct_ues.sum()),
            "relay_served_ues": int(relay_ues.sum()),
            "required_relay_served_ues": int((required & relay_ues).sum()),
            "total_throughput_mbps": float(rates.sum()),
            "offered_access_mbps": float(total_offer),
            "backhaul_reduction_mbps": float(bh_loss),
            "direct_bs_lost_mbps": float(
                sum(v["direct_bs_loss_mbps"] for v in hop_losses.values())
            ),
            "access_to_relay_lost_mbps": float(
                sum(v["a2a_loss_mbps"] for v in hop_losses.values())
            ),
            "relay_to_bs_lost_mbps": float(
                sum(v["relay_bs_loss_mbps"] for v in hop_losses.values())
            ),
            "associated_ue_count": int(assigned.sum()),
            "total_access_capacity_slots": int(len(active_access_ids) * env.cfg.max_ues_per_access),
            "offered_access_uavs_with_zero_bh": len(no_bh),
            "backhaul_limited_access_uavs": len(limited),
            "effective_relay_access_routes": sum(
                1
                for a, rid in routes.items()
                if rid is not None and offer.get(a, 0) > EPS and caps.get(a, 0) > EPS
            ),
            "active_access_uavs": int(len(active_access_ids)),
            "active_relay_uavs": int(len(active_relay_ids)),
            "deep_access_uavs": int(deep_count),
            "any_deep_access": any_deep,
        },
        offer,
        hop_losses,
    )


def radio_with_roles_exact(env, snap, newroles):
    """Exact counterfactual for same SLOT (pre-mobility); restores every field."""
    saved = {
        name: getattr(env, name) for name in ("uav_positions", "ue_positions", "roles", "status")
    }
    try:
        env.uav_positions = snap["uav_positions"].copy()
        env.ue_positions = snap["ue_positions"].copy()
        env.roles = np.asarray(newroles, dtype=np.int64).copy()
        env.status = snap["status"].copy()
        return V4FinalFixedMapEnv.compute_network(env)
    finally:
        for name, v in saved.items():
            setattr(env, name, v)


def audit_oracle(env, snap):
    active = np.flatnonzero(snap["status"] == ACTIVE)
    current_roles = snap["roles"].copy()
    original = radio_with_roles_exact(env, snap, current_roles)
    if abs(float(original["utility"]) - float(snap["utility"])) > 1e-8:
        raise AssertionError("Exact same-slot replay diverged; likely snapshot/mobility mismatch")
    all_access = current_roles.copy()
    all_access[active] = ACCESS
    all_u = float(radio_with_roles_exact(env, snap, all_access)["utility"])
    best_score = -1.0
    best_nrelay = 0
    for flags in itertools.product((ACCESS, RELAY), repeat=len(active)):
        candidate = current_roles.copy()
        candidate[active] = flags
        score = float(radio_with_roles_exact(env, snap, candidate)["utility"])
        if score > best_score + 1e-11:
            best_score = score
            best_nrelay = int(sum(v == RELAY for v in flags))
    return {
        "oracle_all_access_utility": all_u,
        "oracle_best_role_utility": best_score,
        "oracle_actual_utility": float(snap["utility"]),
        "oracle_role_gap": max(0.0, best_score - float(snap["utility"])),
        "oracle_value_over_allaccess": best_score - all_u,
        "oracle_best_num_relay": best_nrelay,
    }


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def run_one(model, checkpoint, seeds, horizon, audit_every):
    env0 = make_env(seeds[0], horizon)
    agent = make_learner(env0)
    agent.load(str(checkpoint))
    agent.agent.eval()
    agent.mixer.eval()
    slots, access_rows, oracle_rows, episode_rows = [], [], [], []
    # Reproduce training-run heldout utility on each exact scenario before
    # trusting derived capacity diagnostics. This catches wrong checkpoints,
    # mismatched environment versions or seed/horizon mistakes.
    stored = {}
    summary_path = checkpoint.parent / f"seed{int(checkpoint.name.split('_')[0][4:])}_summary.json"
    if horizon == 150 and summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        stored = {
            int(x["seed"]): float(x["mean_utility"])
            for x in summary.get("heldout_test", {}).get("rows", [])
        }
    for scenario_seed in seeds:
        env = make_env(scenario_seed, horizon)
        direct_cells = direct_access_cells(env)
        out = env.reset(seed=scenario_seed)
        rng = np.random.default_rng(900000 + int(scenario_seed))
        scenario_slots = []
        for step in range(1, horizon + 1):
            actions = agent.act(out["obs"], out["action_mask"], 0.0, rng)
            env.last_network = None
            out = env.step(actions)
            snap = env.last_network
            if snap is None or abs(snap["utility"] - float(out["reward"])) > 1e-8:
                raise AssertionError("Exact captured utility != step reward")
            metrics, offers, hops = slot_metrics(env, snap, direct_cells)
            row = {
                "model": model,
                "scenario_seed": scenario_seed,
                "step": step,
                **metrics,
            }
            slots.append(row)
            scenario_slots.append(row)
            for aid, users in snap["association"].items():
                rate_sum = float(snap["ue_rate_mbps"][users].sum())
                offer = offers.get(aid, 0.0)
                cap = snap["backhaul_caps_mbps"].get(aid, 0.0)
                access_rows.append(
                    {
                        "model": model,
                        "scenario_seed": scenario_seed,
                        "step": step,
                        "access_id": aid,
                        "relay_id": snap["routes"].get(aid),
                        "role_route": "relay"
                        if snap["routes"].get(aid) is not None
                        else "direct_or_unreachable",
                        "num_associated": len(users),
                        "num_served": int((snap["ue_rate_mbps"][users] > EPS).sum()),
                        "offer_mbps": offer,
                        "backhaul_cap_mbps": cap,
                        "actual_mbps": rate_sum,
                        "backhaul_scale": (min(1.0, cap / offer) if offer > EPS else 1.0),
                        "lost_mbps_due_to_bh": max(0.0, offer - cap),
                        **hops[int(aid)],
                        "at_access_ue_cap": int(len(users) == env.cfg.max_ues_per_access),
                    }
                )
            if audit_every > 0 and (step == 1 or step % audit_every == 0):
                oracle_rows.append(
                    {
                        "model": model,
                        "scenario_seed": scenario_seed,
                        "step": step,
                        **audit_oracle(env, snap),
                    }
                )
            if out["truncated"] or out["terminated"]:
                break
        scalars = [k for k in scenario_slots[0] if k not in ("model", "scenario_seed", "step")]
        episode_rows.append(
            {
                "model": model,
                "scenario_seed": scenario_seed,
                "steps": len(scenario_slots),
                **{"mean_" + k: float(np.mean([r[k] for r in scenario_slots])) for k in scalars},
            }
        )
        if int(scenario_seed) in stored:
            expected = stored[int(scenario_seed)]
            observed = episode_rows[-1]["mean_utility"]
            if not np.isclose(observed, expected, rtol=0, atol=5e-7):
                raise AssertionError(
                    f"{model} scenario={scenario_seed}: "
                    f"replayed utility={observed} != saved heldout={expected}"
                )
        print(
            "DONE",
            model,
            scenario_seed,
            "utility=",
            round(episode_rows[-1]["mean_utility"], 6),
            "deep_required_served=",
            round(episode_rows[-1]["mean_required_served"], 4),
            flush=True,
        )
    return slots, access_rows, oracle_rows, episode_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variants", nargs="+", choices=tuple(PATHS), default=list(PATHS))
    ap.add_argument("--train-seed", type=int, default=91)
    ap.add_argument("--scenarios", nargs="+", type=int, default=[3102, 3106, 3117])
    ap.add_argument("--horizon", type=int, default=150)
    ap.add_argument(
        "--oracle-every",
        type=int,
        default=25,
        help="Role-only exact enumeration every N steps; set 0 to disable (faster)",
    )
    ap.add_argument("--outdir", type=Path, default=Path("results/relay_stress_r4_qmix/cause_audit"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--torch-threads", type=int, default=1)
    args = ap.parse_args()
    paths = {n: checkpoint_path(n, args.train_seed) for n in args.variants}
    for name, path in paths.items():
        print("MODEL", name, "checkpoint=", path, "exists=", path.exists(), flush=True)
    print(
        "SCENARIOS",
        args.scenarios,
        "horizon",
        args.horizon,
        "oracle-every",
        args.oracle_every,
        flush=True,
    )
    if args.dry_run:
        return
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing trained R4 checkpoints (NOT in JSON KPIs):\n" + "\n".join(missing)
        )
    torch.set_num_threads(max(args.torch_threads, 1))
    outdir = args.outdir if args.outdir.is_absolute() else ROOT / args.outdir
    slots, accesses, oracles, episodes = [], [], [], []
    for name, ckpt in paths.items():
        s, a, o, e = run_one(name, ckpt, args.scenarios, args.horizon, args.oracle_every)
        slots.extend(s)
        accesses.extend(a)
        oracles.extend(o)
        episodes.extend(e)
    write_csv(outdir / "per_slot.csv", slots)
    write_csv(outdir / "per_access.csv", accesses)
    write_csv(outdir / "same_geometry_role_oracle.csv", oracles)
    write_csv(outdir / "by_scenario.csv", episodes)
    summary = {}
    for name in paths:
        rs = [r for r in episodes if r["model"] == name]
        summary[name] = {
            k: float(np.mean([r[k] for r in rs])) for k in rs[0] if k.startswith("mean_")
        }
        rows = [r for r in oracles if r["model"] == name]
        if rows:
            summary[name].update(
                {
                    k: float(np.mean([r[k] for r in rows]))
                    for k in (
                        "oracle_role_gap",
                        "oracle_value_over_allaccess",
                        "oracle_best_num_relay",
                    )
                }
            )
    (outdir / "summary.json").write_text(
        json.dumps(
            {
                "audit_type": "read-only same-scenario checkpoint replay; hypothetical role-oracle is NOT a deployable online policy",
                "seeds": args.scenarios,
                "horizon": args.horizon,
                "oracle_every": args.oracle_every,
                "model_summary": summary,
                "note": "Per-slot snapshot from compute_network BEFORE UE mobility / battery return trigger. "
                "Total offered Access rate is reconstructed from exact associated users, Shannon and bandwidth sharing. "
                "Same-geometry oracle changes ONLY active UAV roles, not movement, battery or UE positions.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("WROTE", outdir, flush=True)
    print("SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
