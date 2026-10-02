"""Operational evaluation without retraining: 10,000 executed simulator slots.

Every method sees the same fresh scenario seeds, each episode is 150 slots.
No 10k-step single, never-reset event is silently substituted for that model.
For an actual step, snapshot the network/UE state *during* compute_network,
because UE mobility happens later within env.step.
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.envs.v0_env import ACCESS, RELAY, MOVE_DELTAS
from src.envs.v4_fixed_mobile_battery_env import ACTIVE, RETURNING, CHARGING
from src.envs.thesis_cartesian_r6_curriculum_env import CartesianR6BatteryEnv, CartesianR6BatteryConfig
from src.rl.v5_relay_request_spatial_qmix import V5RelayRequestSpatialQMIXLearner
from src.baselines.thesis_exhaustive_cartesian_role_greedy import ExhaustiveCartesianRoleGreedy

TEST_SEED_START = 3101


class CapturedCartesianEnv(CartesianR6BatteryEnv):
    """Exact in-slot service state for the *executed* env.step, not inferred links."""
    def __init__(self, config):
        self._capture_network = False
        self.last_network_snapshot = None
        super().__init__(config)

    def compute_network(self):
        net = super().compute_network()
        if self._capture_network:
            self.last_network_snapshot = {
                "t_before_advance": int(self.t),
                "uav_cells": self.uav_positions.astype(float).tolist(),
                "uav_roles": self.roles.astype(int).tolist(),
                "uav_status": self.status.astype(int).tolist(),
                "battery_at_service": self.battery.astype(float).tolist(),
                "ue_cells": self.ue_positions.astype(float).tolist(),
                "ue_group": self.ue_hotspot.astype(int).tolist(),
                "ue_rate_mbps": net["ue_rate_mbps"].astype(float).tolist(),
                "association": {str(k): list(map(int, v)) for k, v in net["association"].items()},
                "routes": {str(k): (None if v is None else int(v)) for k, v in net["routes"].items()},
                "backhaul_caps_mbps": {str(k): float(v) for k, v in net["backhaul_caps_mbps"].items()},
                "utility": float(net["utility"]),
            }
        return net


def make_env(seed, horizon):
    return CapturedCartesianEnv(CartesianR6BatteryConfig(
        seed=int(seed), horizon=int(horizon),
        residual_initial_battery=30.0, randomize_residual_uav=True,
    ))


def make_learner(env, checkpoint, device):
    learner = V5RelayRequestSpatialQMIXLearner(
        env.cfg.num_uavs, env.obs_dim, env.state_dim, env.n_actions,
        grid_size=env.cfg.grid_size, coarse_size=env.cfg.coarse_obs_size,
        hidden_dim=64, mixing_embed_dim=32, device=device,
    )
    learner.load(str(checkpoint))
    learner.agent.eval()
    learner.mixer.eval()
    return learner


class LocalAllAccess:
    """New transparent online heuristic; NOT claimed to reproduce an older Fast5A file.

    No network what-if search. Select one local move per UAV; reward underserved
    currently reachable UEs and availability of any direct BS backhaul.
    It intentionally never uses Relay; the strong exhaustive benchmark is separate.
    """
    def __init__(self, env):
        self.env = env

    def act(self, mask):
        e = self.env
        claimed = set()
        actions = []
        for i in range(e.cfg.num_uavs):
            if e.status[i] != ACTIVE:
                actions.append(int(np.flatnonzero(mask[i])[0]))
                continue
            best = None
            best_move = 4
            for mv in range(5):
                act = e.encode_action(mv, ACCESS)
                if not bool(mask[i, act]):
                    continue
                cand = e.uav_positions[i] + MOVE_DELTAS[mv]
                if not e._inside_grid(cand):
                    continue
                d = np.linalg.norm(e.ue_positions - cand[None, :], axis=1)
                covered = np.flatnonzero(d <= e.cfg.access_radius_cells + 1e-9)
                covered = covered[np.argsort(d[covered])[:e.cfg.max_ues_per_access]]
                unique = float(sum(int(u) not in claimed for u in covered))
                access_xy = e._cell_to_xy(cand)
                feasible = [bs for bs in e.surviving_bs_xy if e._xy_within_radius_m(
                    access_xy, bs, e.cfg.backhaul_radius_cells)]
                direct_cap = max((e._bh_a2g_rate(access_xy, e.cfg.uav_altitude_m,
                             bs, e.cfg.bs_height_m) for bs in feasible), default=0.0)
                # One-step feasible throughput proxy, then coverage, then Stay.
                score = (min(direct_cap / e.cfg.ue_demand_mbps, unique),
                         unique, float(len(covered)), int(mv == 4))
                if best is None or score > best:
                    best = score
                    best_move = mv
            actions.append(int(e.encode_action(best_move, ACCESS)))
            cell = e.uav_positions[i] + MOVE_DELTAS[best_move]
            d = np.linalg.norm(e.ue_positions - cell[None, :], axis=1)
            covered = np.flatnonzero(d <= e.cfg.access_radius_cells + 1e-9)
            covered = covered[np.argsort(d[covered])[:e.cfg.max_ues_per_access]]
            claimed.update(map(int, covered))
        return actions


def counts_from_exact_snapshot(snap, demand):
    """UEs counted from *achieved* per-UE rates, not geometry alone."""
    rates = np.asarray(snap["ue_rate_mbps"], dtype=float)
    served = int(np.sum(rates > 1e-9))
    full = int(np.sum(rates >= float(demand) - 1e-9))
    via_relay = 0
    active_routes = 0
    for aid, rid in snap["routes"].items():
        if rid is None or snap["backhaul_caps_mbps"].get(aid, 0.0) <= 1e-9:
            continue
        users = snap["association"].get(aid, [])
        n = sum(rates[u] > 1e-9 for u in users)
        if n > 0:
            active_routes += 1
            via_relay += n
    return served, full, via_relay, active_routes


def episode_start_snapshot(env):
    return {
        "t_before_advance": 0,
        "uav_cells": env.uav_positions.astype(float).tolist(),
        "uav_roles": env.roles.astype(int).tolist(),
        "uav_status": env.status.astype(int).tolist(),
        "battery_at_service": env.battery.astype(float).tolist(),
        "ue_cells": env.ue_positions.astype(float).tolist(),
        "ue_group": env.ue_hotspot.astype(int).tolist(),
        "ue_rate_mbps": [0.0] * env.cfg.num_ues,
        "association": {}, "routes": {}, "backhaul_caps_mbps": {},
        "utility": 0.0,
        "is_initial_state_without_action": True,
    }


def run_method(args, label, policy_name, checkpoint=None):
    rng = np.random.default_rng(400000 + sum(map(ord, label)))
    env = make_env(args.seed_start, args.horizon)
    learner = make_learner(env, checkpoint, args.device) if policy_name == "rl" else None
    if policy_name == "exhaustive":
        policy = ExhaustiveCartesianRoleGreedy(env)
    elif policy_name == "local5a":
        policy = LocalAllAccess(env)
    else:
        policy = None

    rows, case, best_relay = [], {}, None
    best_relay_count = -1
    global_step = 0
    n_cases = 0
    while global_step < args.steps:
        seed = int(args.seed_start + n_cases)
        n_cases += 1
        out = env.reset(seed=seed)
        if policy_name == "rl" and label == f"rl_seed{args.case_training_seed}" and seed == args.case_seed:
            case["0"] = episode_start_snapshot(env)
        for t in range(1, args.horizon + 1):
            if global_step >= args.steps:
                break
            env._capture_network = False
            tick = time.perf_counter()
            if learner is not None:
                actions = learner.act(out["obs"], out["action_mask"], 0.0, rng)
            elif policy_name == "random":
                actions = [int(rng.choice(np.flatnonzero(mask))) for mask in out["action_mask"]]
            else:
                actions = policy.act(out["action_mask"])
            decision_ms = (time.perf_counter() - tick) * 1000.0

            env._capture_network = True
            out = env.step(actions)
            env._capture_network = False
            snap = env.last_network_snapshot
            if snap is None or abs(snap["utility"] - float(out["reward"])) > 1e-8:
                raise RuntimeError("Captured in-slot network result disagrees with executed reward")
            served, full, via_relay, n_links = counts_from_exact_snapshot(snap, env.cfg.ue_demand_mbps)
            global_step += 1
            status_after = np.asarray(out["info"]["status"], dtype=int)
            row = {
                "step": global_step, "scenario_seed": seed, "episode": n_cases,
                "episode_step": t, "episode_reset": int(t == 1),
                "utility": float(out["reward"]), "served_ues": served,
                "fully_satisfied_ues": full, "relay_served_ues": via_relay,
                "effective_relay_routes": n_links,
                "access": int(np.sum((np.asarray(snap["uav_roles"]) == ACCESS) &
                                     (np.asarray(snap["uav_status"]) == ACTIVE))),
                "relay": int(np.sum((np.asarray(snap["uav_roles"]) == RELAY) &
                                    (np.asarray(snap["uav_status"]) == ACTIVE))),
                "returning": int((status_after == RETURNING).sum()),
                "charging": int((status_after == CHARGING).sum()),
                "mean_battery": float(np.mean(out["info"]["battery"])),
                "residual_battery": float(out["info"]["battery"][env.residual_uav_idx]),
                "decision_ms": float(decision_ms),
            }
            rows.append(row)
            if policy_name == "rl" and label == f"rl_seed{args.case_training_seed}":
                snap["scenario_seed"] = seed
                snap["episode_step"] = t
                snap["step_global"] = global_step
                if seed == args.case_seed and t in (1, 20, 50, 75, 100, 125, args.horizon):
                    case[str(t)] = snap
                # Illustrative *observed* routed traffic; select best in 10k,
                # never use this selected snapshot for aggregate performance.
                if via_relay > best_relay_count:
                    best_relay_count = via_relay
                    best_relay = snap
            if out["truncated"]:
                break
        if n_cases % 10 == 0 or global_step == args.steps:
            print(f"{label}: {global_step}/{args.steps} slots; episodes={n_cases}", flush=True)

    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)
    filename = outdir / f"{label}_steps.csv"
    with filename.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    aux = {
        "method": label, "steps": global_step, "horizon": args.horizon,
        "scenario_seeds": [args.seed_start, args.seed_start + n_cases - 1],
        "scenario_count": n_cases, "partial_last_episode": bool(args.steps % args.horizon),
        "utility_mean_all_slots": float(np.mean([r["utility"] for r in rows])),
        "served_ues_mean": float(np.mean([r["served_ues"] for r in rows])),
        "mean_decision_ms": float(np.mean([r["decision_ms"] for r in rows])),
        "episodes_are_reset": True,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "labels_do_not_imply_same_implementation_as_prior_fast5a": True,
    }
    (outdir / f"{label}_summary.json").write_text(json.dumps(aux, indent=2), encoding="utf-8")
    if case:
        traces = {"method": label, "case_seed": args.case_seed,
                  "case_frames": case, "best_observed_relay": best_relay,
                  "best_relay_served_ues": int(best_relay_count),
                  "bs_xy_m": env.bs_xy.astype(float).tolist(),
                  "failed_bs_idx": int(env.cfg.failed_bs_idx),
                  "charging_station_cells": env.charging_stations.astype(float).tolist(),
                  "affected_mask": env.affected_mask.astype(int).tolist(),
                  "cell_size_m": float(env.cfg.cell_size_m),
                  "access_radius_m": float(env.cfg.access_radius_cells * env.cfg.cell_size_m),
                  "backhaul_radius_m": float(env.cfg.backhaul_radius_cells * env.cfg.cell_size_m),
                  "ue_demand_mbps": float(env.cfg.ue_demand_mbps),
                  "bs_height_m": float(env.cfg.bs_height_m),
                  "uav_altitude_m": float(env.cfg.uav_altitude_m)}
        (outdir / "rl_case_study.json").write_text(json.dumps(traces), encoding="utf-8")
    print("FINISHED", json.dumps(aux), flush=True)
    return aux


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--horizon", type=int, default=150)
    ap.add_argument("--seed-start", type=int, default=3101)
    ap.add_argument("--case-seed", type=int, default=3101)
    ap.add_argument("--case-training-seed", type=int, default=91)
    ap.add_argument("--rl-seeds", nargs="+", type=int, default=[91, 92, 93])
    ap.add_argument("--methods", nargs="+", choices=["random", "local5a", "rl", "exhaustive"],
                    default=["random", "local5a", "rl"])
    ap.add_argument("--checkpoint-root", default="results/thesis_curriculum_ablation/direct_full_task_randomized")
    ap.add_argument("--outdir", default="results/thesis_operational_eval")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--torch-threads", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.steps < 1 or args.horizon < 1:
        ap.error("steps and horizon must be positive")
    if args.case_seed < args.seed_start or args.case_seed >= args.seed_start + (args.steps + args.horizon - 1) // args.horizon:
        ap.error("case-seed must lie in the evaluated scenario range")
    torch.set_num_threads(max(1, args.torch_threads))
    # Check all requested RL files BEFORE running any expensive rollout.
    ckroot = Path(args.checkpoint_root)
    if not ckroot.is_absolute():
        ckroot = ROOT / ckroot
    checkpoints = {
        seed: ckroot / f"seed{seed}_best_final_block.pt" for seed in args.rl_seeds
    }
    if "rl" in args.methods:
        missing = [str(p) for p in checkpoints.values() if not p.is_file()]
        if missing:
            ap.error("Missing FINAL Cartesian no-curriculum checkpoint(s): " + ", ".join(missing))
    if "rl" in args.methods and args.case_training_seed not in args.rl_seeds:
        ap.error("case-training-seed must be one of rl-seeds")

    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)
    for method in args.methods:
        configs = [(f"rl_seed{s}", "rl", checkpoints[s]) for s in args.rl_seeds] if method == "rl" else [(method, method, None)]
        for label, policy_name, checkpoint in configs:
            csvpath = outdir / f"{label}_steps.csv"
            summary_path = outdir / f"{label}_summary.json"
            if csvpath.is_file() and summary_path.is_file() and not args.force:
                existing = json.loads(summary_path.read_text(encoding="utf-8"))
                if existing.get("steps") == args.steps and existing.get("horizon") == args.horizon and existing.get("scenario_seeds", [])[0] == args.seed_start:
                    print("SKIP completed:", label, flush=True)
                    continue
            run_method(args, label, policy_name, checkpoint)
    print("Now run: python scripts/plot_uav_operational_eval.py --input", outdir, flush=True)


if __name__ == "__main__":
    main()
