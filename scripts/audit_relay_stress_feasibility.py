"""Fast feasibility audit for the 200 m relay-stress thesis topology.

This is NOT RL training.  It runs privileged exact-state heuristic references
on the same held-out scenario seeds used by the reward ablation:

* 5A + Recovery: all ACTIVE UAVs are Access.
* Dynamic + Recovery: exact enumeration of every ACTIVE Access/Relay role
  vector (<= 2^5 = 32), followed by one-pass coordinate-greedy movement.

The two policies share the same zero-service recovery implementation.  The
script additionally computes an exact *same-geometry role oracle gap* during
Dynamic decisions: best current role vector utility minus all-Access utility,
with UAV positions held fixed.  This isolates whether Relay has immediate
value at the current geometry from trajectory differences.

No future rollout and no joint 5^N movement search are used.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_uav_operational_eval import CapturedCartesianEnv, counts_from_exact_snapshot
from src.baselines.recovery_aware_all_access_greedy import RecoveryAwareAllAccessGreedy
from src.baselines.recovery_aware_role_greedy import RecoveryAwareExhaustiveGreedy, RecoveryAwareFixedMixGreedy
from src.envs.thesis_cartesian_r4_relay_stress_env import CartesianR4RelayStressBatteryConfig
from src.envs.v0_env import ACCESS
from src.envs.v4_fixed_mobile_battery_env import ACTIVE

DEFAULT_TEST_SEEDS = tuple(range(3101, 3121))
EPS = 1e-9


def make_env(seed: int, horizon: int) -> CapturedCartesianEnv:
    cfg = CartesianR4RelayStressBatteryConfig(
        seed=int(seed), horizon=int(horizon),
        enable_ue_mobility=True, enable_battery_cycle=True,
        residual_initial_battery=30.0, randomize_residual_uav=True,
    )
    env = CapturedCartesianEnv(cfg)
    if abs(float(env.cfg.backhaul_radius_cells) - 4.0) > 1e-12:
        raise AssertionError("R4 audit accidentally instantiated a non-R4 environment")
    return env


class GeometryRelayAudit:
    """Precompute static R4 grid/backhaul geometry; evaluate moving UEs cheaply."""

    def __init__(self, env):
        g = int(env.cfg.grid_size)
        self.cells = np.indices((g, g)).reshape(2, -1).T.astype(np.float64)
        cell_xy = (self.cells[:, ::-1] + 0.5) * float(env.cfg.cell_size_m)
        gateways = np.asarray(env.surviving_bs_xy, dtype=np.float64)
        self.bh_m = float(env.cfg.backhaul_radius_cells * env.cfg.cell_size_m)
        self.access_radius_cells = float(env.cfg.access_radius_cells)
        nearest_bs = np.min(
            np.linalg.norm(cell_xy[:, None, :] - gateways[None, :, :], axis=2),
            axis=1,
        )
        self.relay_direct = nearest_bs <= self.bh_m + EPS
        direct_xy = cell_xy[self.relay_direct]
        if len(direct_xy):
            nearest_direct_relay = np.min(
                np.linalg.norm(cell_xy[:, None, :] - direct_xy[None, :, :], axis=2),
                axis=1,
            )
            self.access_can_relay = nearest_direct_relay <= self.bh_m + EPS
        else:
            self.access_can_relay = np.zeros(len(cell_xy), dtype=bool)

    def masks(self, ue_cells):
        ue = np.asarray(ue_cells, dtype=np.float64)
        cover = np.linalg.norm(
            self.cells[:, None, :] - ue[None, :, :], axis=2
        ) <= self.access_radius_cells + EPS
        direct_possible = np.any(cover & self.relay_direct[:, None], axis=0)
        relay_possible = np.any(cover & self.access_can_relay[:, None], axis=0)
        return ~direct_possible, relay_possible


def geometry_masks(env, ue_cells) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility helper used by tests; one precomputation per call."""
    return GeometryRelayAudit(env).masks(ue_cells)


def all_access_role_vector(env) -> np.ndarray:
    return np.full(int(env.cfg.num_uavs), ACCESS, dtype=np.int64)


def run_episode(policy_name: str, scenario_seed: int, horizon: int, patience: int):
    env = make_env(scenario_seed, horizon)
    out = env.reset(seed=int(scenario_seed))
    geom = GeometryRelayAudit(env)
    if policy_name == "5A":
        policy = RecoveryAwareAllAccessGreedy(env, zero_service_patience=patience)
    elif policy_name == "dynamic":
        policy = RecoveryAwareExhaustiveGreedy(env, zero_service_patience=patience)
    elif policy_name in ("4A1R", "3A2R"):
        policy = RecoveryAwareFixedMixGreedy(
            env, composition=policy_name, zero_service_patience=patience
        )
    else:
        raise ValueError(policy_name)

    totals = defaultdict(float)
    role_gain_positive = 0
    role_gain_slots = 0
    rows = []
    for step in range(1, horizon + 1):
        before_status = env.status.copy()

        # Same-geometry exact role benefit.  This is an audit only and does not
        # alter the selected action or environment state.
        all_access_score = best_role_score = None
        if policy_name == "dynamic":
            all_access_score = float(policy._score(all_access_role_vector(env)))

        env._capture_network = False
        tic = time.perf_counter()
        actions = policy.act(out["action_mask"])
        decision_ms = 1000.0 * (time.perf_counter() - tic)
        dbg = policy.last_stats.copy()
        if policy_name == "dynamic":
            best_role_score = float(dbg["role_only_utility"])
            gain = best_role_score - float(all_access_score)
            role_gain_slots += 1
            role_gain_positive += int(gain > 1e-10)
            totals["same_geometry_role_gain"] += gain
        else:
            gain = 0.0

        env._capture_network = True
        out = env.step(actions)
        env._capture_network = False
        snap = env.last_network_snapshot
        if snap is None or abs(float(snap["utility"]) - float(out["reward"])) > 1e-8:
            raise RuntimeError("Captured network and actual reward disagree")
        policy.observe_service(float(out["reward"]))

        served, full, relay_ues, routes = counts_from_exact_snapshot(snap, env.cfg.ue_demand_mbps)
        rates = np.asarray(snap["ue_rate_mbps"], dtype=float)
        required, relay_possible = geom.masks(np.asarray(snap["ue_cells"], dtype=float))
        required_count = int(np.sum(required))
        required_feasible_count = int(np.sum(required & relay_possible))
        required_served = int(np.sum(required & (rates > EPS)))

        service_roles = np.asarray(snap["uav_roles"], dtype=int)
        service_status = np.asarray(snap["uav_status"], dtype=int)
        n_relay = int(np.sum((service_roles != ACCESS) & (service_status == ACTIVE)))

        row = {
            "policy": policy_name,
            "scenario_seed": int(scenario_seed),
            "step": int(step),
            "utility": float(out["reward"]),
            "served_ues": int(served),
            "fully_satisfied_ues": int(full),
            "relay_served_ues": int(relay_ues),
            "effective_relay_routes": int(routes),
            "relay_required_ues_geometry": required_count,
            "relay_required_feasible_ues_geometry": required_feasible_count,
            "relay_required_served_ues": required_served,
            "active_relay_uavs": n_relay,
            "decision_ms": float(decision_ms),
            "recovery_active": int(bool(dbg.get("recovery_active", False))),
            "same_geometry_all_access_utility": all_access_score,
            "same_geometry_best_role_utility": best_role_score,
            "same_geometry_role_gain": float(gain),
        }
        rows.append(row)
        for k in (
            "utility", "served_ues", "fully_satisfied_ues", "relay_served_ues",
            "effective_relay_routes", "relay_required_ues_geometry",
            "relay_required_feasible_ues_geometry", "relay_required_served_ues",
            "active_relay_uavs", "decision_ms", "recovery_active",
        ):
            totals[k] += float(row[k])

        if out.get("terminated") or out.get("truncated"):
            if step != horizon:
                raise RuntimeError(f"Unexpected early episode end at step {step}")
            break

    n = len(rows)
    summary = {
        "policy": policy_name,
        "scenario_seed": int(scenario_seed),
        "steps": n,
        "backhaul_radius_cells": float(env.cfg.backhaul_radius_cells),
        "backhaul_radius_m": float(env.cfg.backhaul_radius_cells * env.cfg.cell_size_m),
        "access_radius_m": float(env.cfg.access_radius_cells * env.cfg.cell_size_m),
        "mean_utility": totals["utility"] / n,
        "mean_served_ues": totals["served_ues"] / n,
        "mean_fully_satisfied_ues": totals["fully_satisfied_ues"] / n,
        "mean_relay_served_ues": totals["relay_served_ues"] / n,
        "mean_effective_relay_routes": totals["effective_relay_routes"] / n,
        "mean_relay_required_ues_geometry": totals["relay_required_ues_geometry"] / n,
        "mean_relay_required_feasible_ues_geometry": totals["relay_required_feasible_ues_geometry"] / n,
        "mean_relay_required_served_ues": totals["relay_required_served_ues"] / n,
        "mean_active_relay_uavs": totals["active_relay_uavs"] / n,
        "mean_decision_ms": totals["decision_ms"] / n,
        "recovery_fraction": totals["recovery_active"] / n,
        "mean_same_geometry_role_gain": (
            totals["same_geometry_role_gain"] / role_gain_slots if role_gain_slots else None
        ),
        "positive_same_geometry_role_gain_fraction": (
            role_gain_positive / role_gain_slots if role_gain_slots else None
        ),
    }
    return summary, rows


def mean_std(rows, key):
    vals = np.asarray([float(r[key]) for r in rows], dtype=float)
    return float(np.mean(vals)), float(np.std(vals))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test-seeds", nargs="+", type=int, default=list(DEFAULT_TEST_SEEDS))
    ap.add_argument("--horizon", type=int, default=150)
    ap.add_argument("--policies", nargs="+", choices=["5A", "4A1R", "3A2R", "dynamic"], default=["5A", "dynamic"])
    ap.add_argument("--zero-service-patience", type=int, default=3)
    ap.add_argument("--outdir", default="results/relay_stress_r4_feasibility")
    args = ap.parse_args()
    if args.horizon < 1 or args.zero_service_patience < 1:
        ap.error("horizon and patience must be positive")

    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    episode_rows = []
    step_rows = []
    for policy in args.policies:
        for seed in args.test_seeds:
            summary, rows = run_episode(policy, seed, args.horizon, args.zero_service_patience)
            episode_rows.append(summary)
            step_rows.extend(rows)
            print(
                f"DONE {policy} seed={seed}: U={summary['mean_utility']:.4f} "
                f"served={summary['mean_served_ues']:.2f} relayUE={summary['mean_relay_served_ues']:.2f} "
                f"required={summary['mean_relay_required_ues_geometry']:.2f} "
                f"required-served={summary['mean_relay_required_served_ues']:.2f}",
                flush=True,
            )

    aggregate = []
    for policy in args.policies:
        rr = [r for r in episode_rows if r["policy"] == policy]
        item = {
            "policy": policy,
            "episodes": len(rr),
            "horizon": int(args.horizon),
            "backhaul_radius_m": 200.0,
            "uses_privileged_exact_state": True,
            "future_rollout": False,
            "joint_movement_enumeration": False,
        }
        for key in (
            "mean_utility", "mean_served_ues", "mean_fully_satisfied_ues",
            "mean_relay_served_ues", "mean_effective_relay_routes",
            "mean_relay_required_ues_geometry", "mean_relay_required_feasible_ues_geometry",
            "mean_relay_required_served_ues", "mean_active_relay_uavs",
            "mean_decision_ms", "recovery_fraction",
        ):
            m, sd = mean_std(rr, key)
            item[key] = m
            item[key + "_std"] = sd
        if policy == "dynamic":
            for key in ("mean_same_geometry_role_gain", "positive_same_geometry_role_gain_fraction"):
                m, sd = mean_std(rr, key)
                item[key] = m
                item[key + "_std"] = sd
        aggregate.append(item)

    def write_csv(path: Path, rows):
        if not rows:
            return
        keys = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key); keys.append(key)
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader(); w.writerows(rows)

    write_csv(outdir / "relay_stress_by_step.csv", step_rows)
    write_csv(outdir / "relay_stress_by_episode.csv", episode_rows)
    write_csv(outdir / "relay_stress_summary.csv", aggregate)
    report = {
        "experiment": "R4 relay-stress feasibility audit before RL retraining",
        "change_from_locked_r6": "backhaul radius only: 6 cells/300 m -> 4 cells/200 m",
        "unchanged": [
            "20x20 grid, 50 m/cell", "Access radius 4 cells/200 m", "5 UAV, 90 UE",
            "UE mobility", "battery lifecycle", "10 Cartesian movement x role actions",
            "one-hop opportunistic routing", "Shannon/pathloss network model",
        ],
        "test_seeds": args.test_seeds,
        "horizon": args.horizon,
        "zero_service_patience": args.zero_service_patience,
        "interpretation_guardrail": "Heuristics use privileged exact UE state; this is feasibility evidence, not a fair RL comparison.",
        "summary": aggregate,
    }
    (outdir / "relay_stress_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("SUMMARY")
    for row in aggregate:
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
