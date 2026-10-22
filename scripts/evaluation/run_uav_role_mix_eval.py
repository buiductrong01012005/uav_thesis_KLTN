"""One-reset 4A1R/3A2R evaluation with measured relay traffic.

Runs the same CartesianR6BatteryEnv/scenario seed as the existing 20k QMIX
continuous logs. Supports selecting a single policy or both. This is a
privileged role-search reference, NOT a cheap information-matched RL baseline.
The 3A1R four-active-UAV phase is recorded separately for each policy.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluation.run_uav_continuous_eval import FIELDNAMES, make_continuous_env
from scripts.evaluation.run_uav_operational_eval import counts_from_exact_snapshot
from src.baselines.fixed_mix_cartesian_greedy import FixedMixCartesianGreedy
from src.envs.v4_fixed_mobile_battery_env import ACTIVE, RETURNING, CHARGING
from src.envs.v0_env import ACCESS, RELAY


EXTRA_FIELDS = [
    "active_at_decision",
    "selected_composition",
    "role_vectors_evaluated",
    "movement_candidates_evaluated",
    "role_search_privileged",
]


def run_policy(
    *,
    composition: str,
    steps: int,
    seed: int,
    outdir: Path,
    log_every: int = 1000,
    force: bool = False,
):
    label = f"fixed_{composition.lower()}"
    summary_path = outdir / f"{label}_summary.json"
    csv_path = outdir / f"{label}_steps.csv"
    if steps > 0 and not force and summary_path.is_file() and csv_path.is_file():
        old = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            old.get("steps") == steps
            and old.get("scenario_seed") == seed
            and old.get("evaluation_type") == "one continuous simulator run without episode reset"
            and old.get("algorithm") == "restricted current-state role search + coordinate greedy"
        ):
            print(f"SKIP {label}: matching result already exists", flush=True)
            return old

    env = make_continuous_env(seed, steps + 1 if steps else 2**31 - 1)
    output = env.reset(seed=seed)  # one reset for this policy, never another
    policy = FixedMixCartesianGreedy(env, composition=composition)
    aggregate = defaultdict(float)
    composition_counts = Counter()
    n_active_counts = Counter()
    four_active = defaultdict(float)
    n_four = 0
    n_rows = n_returns = n_rejoins = 0
    first_return_step = first_rejoin_step = None
    t0 = time.perf_counter()
    outdir.mkdir(parents=True, exist_ok=True)
    fields = FIELDNAMES + EXTRA_FIELDS
    print(
        f"START {composition}, seed={seed}, steps={steps or 'infinite'}, one reset",
        flush=True,
    )

    try:
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            while steps == 0 or n_rows < steps:
                env._capture_network = False
                statuses_before = env.status.copy()
                n_active = int(np.sum(statuses_before == ACTIVE))
                tic = time.perf_counter()
                actions = policy.act(output["action_mask"])
                dt_ms = 1000.0 * (time.perf_counter() - tic)
                stats = policy.last_stats.copy()
                # No current-state candidate evaluation should leak into the
                # executed network snapshot. Capture only actual env.step().
                env._capture_network = True
                output = env.step(actions)
                env._capture_network = False
                if output.get("terminated") or output.get("truncated"):
                    raise RuntimeError("Unexpected horizon end in continuous evaluation")
                snapshot = env.last_network_snapshot
                if (
                    snapshot is None
                    or abs(float(snapshot["utility"]) - float(output["reward"])) > 1e-8
                ):
                    raise RuntimeError("In-slot captured network differs from executed utility")
                served, fully, relayed, n_paths = counts_from_exact_snapshot(
                    snapshot, env.cfg.ue_demand_mbps
                )
                after = env.status.copy()
                new_returns = int(np.sum((statuses_before == ACTIVE) & (after == RETURNING)))
                new_rejoins = int(np.sum((statuses_before == CHARGING) & (after == ACTIVE)))
                n_returns += new_returns
                n_rejoins += new_rejoins
                n_rows += 1
                if first_return_step is None and new_returns:
                    first_return_step = n_rows
                if first_rejoin_step is None and new_rejoins:
                    first_rejoin_step = n_rows
                roles = np.asarray(snapshot["uav_roles"], dtype=int)
                status_at_service = np.asarray(snapshot["uav_status"], dtype=int)
                used_access = int(np.sum((roles == ACCESS) & (status_at_service == ACTIVE)))
                used_relay = int(np.sum((roles == RELAY) & (status_at_service == ACTIVE)))
                row = {
                    "step": n_rows,
                    "utility": float(output["reward"]),
                    "served_ues": int(served),
                    "fully_satisfied_ues": int(fully),
                    "relay_served_ues": int(relayed),
                    "effective_relay_routes": int(n_paths),
                    "access": used_access,
                    "relay": used_relay,
                    "returning": int(np.sum(after == RETURNING)),
                    "charging": int(np.sum(after == CHARGING)),
                    "mean_battery": float(np.mean(output["info"]["battery"])),
                    "residual_battery": float(output["info"]["battery"][env.residual_uav_idx]),
                    "decision_ms": float(dt_ms),
                    "return_events_this_slot": new_returns,
                    "rejoin_events_this_slot": new_rejoins,
                    "active_at_decision": n_active,
                    "selected_composition": stats["effective_composition"],
                    "role_vectors_evaluated": int(stats["role_vectors_evaluated"]),
                    "movement_candidates_evaluated": int(stats["movement_candidates_evaluated"]),
                    "role_search_privileged": True,
                }
                writer.writerow(row)
                for key in (
                    "utility",
                    "served_ues",
                    "fully_satisfied_ues",
                    "relay_served_ues",
                    "decision_ms",
                    "role_vectors_evaluated",
                    "movement_candidates_evaluated",
                ):
                    aggregate[key] += float(row[key])
                composition_counts[stats["effective_composition"]] += 1
                n_active_counts[n_active] += 1
                if stats["effective_composition"] == "3A1R":
                    n_four += 1
                    for key in (
                        "utility",
                        "served_ues",
                        "relay_served_ues",
                        "decision_ms",
                    ):
                        four_active[key] += float(row[key])
                if n_rows % log_every == 0:
                    fh.flush()
                    print(
                        f"{composition}: {n_rows}/{steps or '∞'}; mean utility={aggregate['utility'] / n_rows:.4f}; avg decision={aggregate['decision_ms'] / n_rows:.1f}ms",
                        flush=True,
                    )
    except KeyboardInterrupt:
        print(
            f"INTERRUPTED {composition} at step {n_rows}; saving partial run",
            flush=True,
        )

    summary = {
        "method": label,
        "algorithm": "restricted current-state role search + coordinate greedy",
        "requested_composition": composition,
        "evaluation_type": "one continuous simulator run without episode reset",
        "number_of_environment_resets": 1,
        "scenario_seed": seed,
        "steps": n_rows,
        "evaluation_horizon": int(env.cfg.horizon),
        "same_ue_population_and_failed_bs_throughout": True,
        "mobility_and_battery_continuous": True,
        "policy_training_horizon": None,
        "mean_utility": aggregate["utility"] / max(1, n_rows),
        "mean_served_ues": aggregate["served_ues"] / max(1, n_rows),
        "mean_fully_satisfied_ues": aggregate["fully_satisfied_ues"] / max(1, n_rows),
        "mean_relay_served_ues": aggregate["relay_served_ues"] / max(1, n_rows),
        "mean_decision_ms": aggregate["decision_ms"] / max(1, n_rows),
        "avg_role_vectors_evaluated": aggregate["role_vectors_evaluated"] / max(1, n_rows),
        "avg_movement_candidates_evaluated": aggregate["movement_candidates_evaluated"]
        / max(1, n_rows),
        "composition_steps": dict(sorted(composition_counts.items())),
        "active_uav_steps": {str(k): int(v) for k, v in sorted(n_active_counts.items())},
        "three_access_one_relay_window": {
            "steps": n_four,
            "mean_utility": four_active["utility"] / n_four if n_four else None,
            "mean_served_ues": four_active["served_ues"] / n_four if n_four else None,
            "mean_relay_served_ues": four_active["relay_served_ues"] / n_four if n_four else None,
            "mean_decision_ms": four_active["decision_ms"] / n_four if n_four else None,
        },
        "return_events": n_returns,
        "rejoin_events": n_rejoins,
        "first_return_step": first_return_step,
        "first_rejoin_step": first_rejoin_step,
        "privileged_exact_ue_positions": True,
        "uses_exact_network_what_if_at_current_slot": True,
        "uses_future_state_rollout": False,
        "uses_joint_movement_brute_force": False,
        "selection_rule_for_four_active": "3A1R under both 4A1R and 3A2R policies",
        "elapsed_s": float(time.perf_counter() - t0),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("FINISHED", json.dumps(summary), flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--steps",
        type=int,
        default=20000,
        help="One run without resets; 0 means Ctrl+C to stop (one policy only)",
    )
    ap.add_argument("--scenario-seed", type=int, default=3101)
    ap.add_argument("--policies", nargs="+", choices=["4A1R", "3A2R"], default=["4A1R", "3A2R"])
    ap.add_argument("--outdir", default="results/thesis_continuous_eval")
    ap.add_argument("--log-every", type=int, default=1000)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.steps < 0 or args.log_every <= 0:
        ap.error("steps must be >=0 and log-every >0")
    if args.steps == 0 and len(args.policies) != 1:
        ap.error("For infinite evaluation choose just one --policies item")
    dest = Path(args.outdir)
    if not dest.is_absolute():
        dest = ROOT / dest
    for composition in args.policies:
        run_policy(
            composition=composition,
            steps=args.steps,
            seed=args.scenario_seed,
            outdir=dest,
            log_every=args.log_every,
            force=args.force,
        )


if __name__ == "__main__":
    main()
