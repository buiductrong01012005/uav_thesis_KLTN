"""Diagnostic one-reset rollout with recovery-aware greedy navigation.

Run in project root after copying the source patch:
    python scripts/evaluation/run_role_recovery_diagnostic.py --steps 1200 --policies 5A 4A1R 3A2R dynamic
    python scripts/evaluation/run_role_recovery_diagnostic.py --steps 20000 --policies 4A1R 3A2R dynamic --force

Keeps old baseline output untouched. Uses exact current UE positions: a
privileged diagnostic, not the proposed method and not an information-matched
online baseline. No future rollout, no RL training, no automatic episode reset.
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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluation.run_uav_continuous_eval import FIELDNAMES, make_continuous_env
from scripts.evaluation.run_uav_operational_eval import counts_from_exact_snapshot
from src.baselines.recovery_aware_role_greedy import (
    RecoveryAwareFixedMixGreedy,
    RecoveryAwareExhaustiveGreedy,
)
from src.baselines.recovery_aware_all_access_greedy import RecoveryAwareAllAccessGreedy
from src.envs.v4_fixed_mobile_battery_env import ACTIVE, RETURNING, CHARGING
from src.envs.v0_env import ACCESS, RELAY

EXTRA = (
    "recovery_active",
    "zero_service_run_before_decision",
    "active_at_decision",
    "role_vectors_evaluated",
    "movement_candidates_evaluated",
    "selected_composition",
)


def run_one(
    label: str,
    steps: int,
    seed: int,
    dest: Path,
    patience: int,
    log_every: int,
    force: bool,
):
    stem = f"recovery_{label.lower()}"
    p_csv = dest / f"{stem}_steps.csv"
    p_json = dest / f"{stem}_summary.json"
    if not force and steps and p_csv.exists() and p_json.exists():
        s = json.loads(p_json.read_text(encoding="utf-8"))
        if (
            s.get("steps") == steps
            and s.get("scenario_seed") == seed
            and s.get("zero_service_patience") == patience
        ):
            print("SKIP completed:", stem, flush=True)
            return
    env = make_continuous_env(seed, steps + 1 if steps else 2**31 - 1)
    obs = env.reset(seed=seed)
    if label == "dynamic":
        policy = RecoveryAwareExhaustiveGreedy(env, zero_service_patience=patience)
    elif label == "5A":
        policy = RecoveryAwareAllAccessGreedy(env, zero_service_patience=patience)
    else:
        policy = RecoveryAwareFixedMixGreedy(env, composition=label, zero_service_patience=patience)
    totals = defaultdict(float)
    total_steps = 0
    first_return = None
    first_rejoin = None
    start = time.perf_counter()
    dest.mkdir(parents=True, exist_ok=True)
    print("START", stem, "steps", steps, "seed", seed, "patience", patience, flush=True)
    try:
        with p_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDNAMES + list(EXTRA))
            writer.writeheader()
            while steps == 0 or total_steps < steps:
                before = env.status.copy()
                env._capture_network = False
                tic = time.perf_counter()
                a = policy.act(obs["action_mask"])
                ms = (time.perf_counter() - tic) * 1000.0
                dbg = policy.last_stats.copy()
                env._capture_network = True
                obs = env.step(a)
                env._capture_network = False
                if obs["terminated"] or obs["truncated"]:
                    raise RuntimeError("Unexpected end of one-reset evaluation")
                snap = env.last_network_snapshot
                if snap is None or abs(snap["utility"] - float(obs["reward"])) > 1e-8:
                    raise RuntimeError("Captured network and actual utility disagree")
                policy.observe_service(float(obs["reward"]))
                served, full, relay_ues, links = counts_from_exact_snapshot(
                    snap, env.cfg.ue_demand_mbps
                )
                after = env.status.copy()
                returns = int(np.sum((before == ACTIVE) & (after == RETURNING)))
                rejoins = int(np.sum((before == CHARGING) & (after == ACTIVE)))
                total_steps += 1
                if returns and first_return is None:
                    first_return = total_steps
                if rejoins and first_rejoin is None:
                    first_rejoin = total_steps
                roles = np.asarray(snap["uav_roles"], dtype=int)
                at_service = np.asarray(snap["uav_status"], dtype=int)
                n_access = int(np.sum((roles == ACCESS) & (at_service == ACTIVE)))
                n_relay = int(np.sum((roles == RELAY) & (at_service == ACTIVE)))
                row = {
                    "step": total_steps,
                    "utility": float(obs["reward"]),
                    "served_ues": served,
                    "fully_satisfied_ues": full,
                    "relay_served_ues": relay_ues,
                    "effective_relay_routes": links,
                    "access": n_access,
                    "relay": n_relay,
                    "returning": int(np.sum(after == RETURNING)),
                    "charging": int(np.sum(after == CHARGING)),
                    "mean_battery": float(np.mean(obs["info"]["battery"])),
                    "residual_battery": float(obs["info"]["battery"][env.residual_uav_idx]),
                    "decision_ms": ms,
                    "return_events_this_slot": returns,
                    "rejoin_events_this_slot": rejoins,
                    "recovery_active": int(bool(dbg["recovery_active"])),
                    "zero_service_run_before_decision": int(
                        dbg["zero_service_run_before_decision"]
                    ),
                    "active_at_decision": int(np.sum(before == ACTIVE)),
                    "role_vectors_evaluated": int(dbg["role_vectors_evaluated"]),
                    "movement_candidates_evaluated": int(dbg["movement_candidates_evaluated"]),
                    "selected_composition": f"{n_access}A{n_relay}R",
                }
                writer.writerow(row)
                for k in (
                    "utility",
                    "served_ues",
                    "relay_served_ues",
                    "decision_ms",
                    "return_events_this_slot",
                    "rejoin_events_this_slot",
                    "recovery_active",
                ):
                    totals[k] += row[k]
                if total_steps % log_every == 0:
                    fh.flush()
                    print(
                        f"{stem} step={total_steps} avgU={totals['utility'] / total_steps:.4f} "
                        f"recovery={int(totals['recovery_active'])}",
                        flush=True,
                    )
    except KeyboardInterrupt:
        print("Stopped by user, saving partial:", stem, total_steps, flush=True)
    summary = {
        "method": stem,
        "algorithm": "original role selection plus zero-service geometry navigation",
        "scenario_seed": seed,
        "steps": total_steps,
        "zero_service_patience": patience,
        "number_of_environment_resets": 1,
        "mean_utility": totals["utility"] / max(1, total_steps),
        "mean_served_ues": totals["served_ues"] / max(1, total_steps),
        "mean_relay_served_ues": totals["relay_served_ues"] / max(1, total_steps),
        "mean_decision_ms": totals["decision_ms"] / max(1, total_steps),
        "recovery_steps": int(totals["recovery_active"]),
        "return_events": int(totals["return_events_this_slot"]),
        "rejoin_events": int(totals["rejoin_events_this_slot"]),
        "first_return_step": first_return,
        "first_rejoin_step": first_rejoin,
        "uses_current_exact_ue_positions": True,
        "uses_future_network_rollout": False,
        "elapsed_s": time.perf_counter() - start,
    }
    p_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("DONE", json.dumps(summary), flush=True)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--scenario-seed", type=int, default=3101)
    p.add_argument(
        "--policies",
        nargs="+",
        choices=["5A", "4A1R", "3A2R", "dynamic"],
        default=["5A", "4A1R", "3A2R", "dynamic"],
    )
    p.add_argument("--outdir", default="results/thesis_continuous_eval/recovery_audit")
    p.add_argument("--zero-service-patience", type=int, default=3)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    if args.steps < 0 or args.zero_service_patience < 1 or args.log_every < 1:
        p.error("invalid steps/patience/log interval")
    if args.steps == 0 and len(args.policies) != 1:
        p.error("--steps 0 needs exactly one policy")
    dest = Path(args.outdir)
    if not dest.is_absolute():
        dest = ROOT / dest
    for policy in args.policies:
        run_one(
            policy,
            args.steps,
            args.scenario_seed,
            dest,
            args.zero_service_patience,
            args.log_every,
            args.force,
        )


if __name__ == "__main__":
    main()
