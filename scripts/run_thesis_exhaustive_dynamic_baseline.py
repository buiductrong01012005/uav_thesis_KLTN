from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.envs.thesis_cartesian_r6_curriculum_env import CartesianR6BatteryConfig, CartesianR6BatteryEnv
from src.baselines.thesis_exhaustive_cartesian_role_greedy import ExhaustiveCartesianRoleGreedy

TEST_SEEDS = tuple(range(3101, 3121))


def run(seed: int, horizon: int):
    env = CartesianR6BatteryEnv(CartesianR6BatteryConfig(
        seed=seed,
        horizon=horizon,
        residual_initial_battery=30.0,
        randomize_residual_uav=True,
    ))
    out = env.reset(seed=seed)
    policy = ExhaustiveCartesianRoleGreedy(env)

    rewards = []
    times = []
    role_vecs = []
    move_evals = []
    relay_counts = []
    selected_compositions = Counter()
    family_checks = Counter()

    while True:
        actions = policy.act(out["action_mask"])
        st = policy.last_stats
        times.append(st["decision_ms"])
        role_vecs.append(st["role_vectors_evaluated"])
        move_evals.append(st["movement_candidates_evaluated"])
        relay_counts.append(st["selected_relay_count"])
        selected_compositions[f"{st['selected_access_count']}A{st['selected_relay_count']}R"] += 1
        for relay_count, n_vectors in st["role_family_vector_counts"].items():
            family_checks[f"{relay_count}R"] += int(n_vectors)

        out = env.step(actions)
        rewards.append(float(out["reward"]))
        if out["truncated"]:
            break

    return {
        "seed": int(seed),
        "mean_utility": float(np.mean(rewards)),
        "final20": float(np.mean(rewards[-20:])),
        "decision_ms_per_slot": float(np.mean(times)),
        "mean_role_vectors_evaluated": float(np.mean(role_vecs)),
        "mean_movement_candidates_evaluated": float(np.mean(move_evals)),
        "mean_selected_relays": float(np.mean(relay_counts)),
        "selected_composition_counts": dict(selected_compositions),
        "role_family_vector_checks": dict(family_checks),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=list(TEST_SEEDS))
    ap.add_argument("--horizon", type=int, default=150)
    ap.add_argument("--out", default="results/thesis_exhaustive_dynamic_baseline.json")
    args = ap.parse_args()

    rows = []
    for s in args.seeds:
        row = run(int(s), args.horizon)
        rows.append(row)
        print("EVAL", json.dumps(row), flush=True)

    out = {
        "name": "ExhaustiveCartesianRoleGreedy",
        "description": (
            "checks every 2^N Access/Relay role vector at the current geometry, "
            "including all-Access and all-Relay, then performs one-pass coordinate-"
            "greedy movement only for the selected role vector; no future rollout"
        ),
        "privileged_exact_current_simulator_state": True,
        "joint_movement_enumeration": False,
        "role_search_complexity": "O(2^N) role vectors; intended as a strong small-fleet benchmark",
        "rows": rows,
        "mean_utility": float(np.mean([r["mean_utility"] for r in rows])),
        "std_utility": float(np.std([r["mean_utility"] for r in rows])),
        "decision_ms_per_slot": float(np.mean([r["decision_ms_per_slot"] for r in rows])),
        "mean_role_vectors_evaluated": float(np.mean([r["mean_role_vectors_evaluated"] for r in rows])),
        "mean_movement_candidates_evaluated": float(np.mean([r["mean_movement_candidates_evaluated"] for r in rows])),
    }
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("DONE", json.dumps({k: v for k, v in out.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    main()
