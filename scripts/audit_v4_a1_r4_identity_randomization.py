from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.envs.v0_env import ACCESS, RELAY
from src.envs.v4_final_fixed_map_env import ACTIVE
from scripts.run_v4_a1_r4_100k import make_env, build_learner

LAYOUT_SEED = 1101
from src.envs.v4_a1_identity_randomization import (
    RELAY_ANCHOR_CELL as RELAY_ANCHOR,
    randomized_identity_layout as _layout,
    reset_identity_randomized,
)


def randomized_identity_layout(residual_uav_idx: int, permutation_seed: int) -> np.ndarray:
    env = make_env()
    return _layout(env, residual_uav_idx, permutation_seed)

def rollout(learner, residual_uav_idx: int, permutation_seed: int, residual_battery: float) -> dict:
    env = make_env()
    out = reset_identity_randomized(env, residual_uav_idx, permutation_seed, residual_battery)
    rng = np.random.default_rng(90_000 + int(permutation_seed) + 1000 * int(residual_uav_idx))

    rid = int(residual_uav_idx)
    initial_layout = env.uav_positions.astype(int).tolist()
    utilities = []
    relay_counts = []
    relay_ids_by_t = []
    residual_roles = []
    return_time = None
    handover_time = None
    other_relay_at_departure = None
    reactive_replacement_time = None
    role_changes = 0

    while True:
        actions = learner.act(out["obs"], out["action_mask"], 0.0, rng)
        nxt = env.step(actions)
        info = nxt["info"]
        t = int(info["t"])
        roles = np.asarray(info["roles"], dtype=np.int64)
        status = np.asarray(info["status"], dtype=np.int64)
        active_relays = [i for i in range(env.cfg.num_uavs) if status[i] == ACTIVE and roles[i] == RELAY]
        other_relays = [i for i in active_relays if i != rid]

        utilities.append(float(nxt["reward"]))
        relay_counts.append(len(active_relays))
        relay_ids_by_t.append(active_relays)
        residual_roles.append(int(roles[rid]))
        role_changes += int(info["role_switches_this_slot"])

        # A true handover means original relay becomes Access while another active Relay exists.
        if handover_time is None and status[rid] == ACTIVE and roles[rid] == ACCESS and other_relays:
            handover_time = t

        new_return = int(info["return_start_time"][rid])
        if return_time is None and new_return >= 0:
            return_time = new_return
            other_relay_at_departure = bool(other_relays)

        if (
            return_time is not None
            and other_relay_at_departure is False
            and reactive_replacement_time is None
            and t > return_time
            and other_relays
        ):
            reactive_replacement_time = t

        out = nxt
        if nxt["truncated"]:
            break

    relay_at_5 = relay_ids_by_t[4] if len(relay_ids_by_t) >= 5 else relay_ids_by_t[-1]
    relay_at_20 = relay_ids_by_t[19] if len(relay_ids_by_t) >= 20 else relay_ids_by_t[-1]
    return {
        "residual_uav_idx": rid,
        "permutation_seed": int(permutation_seed),
        "residual_battery": float(residual_battery),
        "initial_layout": initial_layout,
        "mean_utility": float(np.mean(utilities)),
        "final20": float(np.mean(utilities[-20:])),
        "mean_active_relays": float(np.mean(relay_counts)),
        "role_changes": int(role_changes),
        "return_time": return_time,
        "handover_time": handover_time,
        "handover_before_return": bool(handover_time is not None and (return_time is None or handover_time < return_time)),
        "other_relay_at_departure": other_relay_at_departure,
        "reactive_replacement_time": reactive_replacement_time,
        "reactive_replacement_delay": (
            int(reactive_replacement_time - return_time)
            if reactive_replacement_time is not None and return_time is not None else None
        ),
        "relay_ids_at_t5": [int(x) for x in relay_at_5],
        "relay_ids_at_t20": [int(x) for x in relay_at_20],
        "residual_is_relay_at_t5": bool(rid in relay_at_5),
        "residual_is_relay_at_t20": bool(rid in relay_at_20),
    }


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    t5 = Counter()
    t20 = Counter()
    for r in rows:
        for i in r["relay_ids_at_t5"]:
            t5[int(i)] += 1
        for i in r["relay_ids_at_t20"]:
            t20[int(i)] += 1
    return {
        "n_cases": n,
        "mean_utility": float(np.mean([r["mean_utility"] for r in rows])),
        "mean_active_relays": float(np.mean([r["mean_active_relays"] for r in rows])),
        "handover_fraction": float(np.mean([r["handover_time"] is not None for r in rows])),
        "handover_before_return_fraction": float(np.mean([r["handover_before_return"] for r in rows])),
        "residual_relay_at_t5_fraction": float(np.mean([r["residual_is_relay_at_t5"] for r in rows])),
        "residual_relay_at_t20_fraction": float(np.mean([r["residual_is_relay_at_t20"] for r in rows])),
        "return_fraction": float(np.mean([r["return_time"] is not None for r in rows])),
        "other_relay_at_departure_fraction": float(np.mean([bool(r["other_relay_at_departure"]) for r in rows if r["return_time"] is not None])) if any(r["return_time"] is not None for r in rows) else None,
        "reactive_replacement_fraction": float(np.mean([r["reactive_replacement_time"] is not None for r in rows])),
        "relay_identity_counts_t5": {str(k): int(v) for k, v in sorted(t5.items())},
        "relay_identity_counts_t20": {str(k): int(v) for k, v in sorted(t20.items())},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--permutations-per-residual", type=int, default=4)
    ap.add_argument("--out", default="results/a1_r4_identity_randomization_sanity.json")
    args = ap.parse_args()

    env = make_env()
    learner = build_learner(env, "cpu")
    learner.load(str(Path(args.checkpoint).resolve()))

    low_rows = []
    full_rows = []
    for rid in range(env.cfg.num_uavs):
        for k in range(args.permutations_per_residual):
            pseed = 50_000 + rid * 100 + k
            low_rows.append(rollout(learner, rid, pseed, env.cfg.residual_initial_battery))
            full_rows.append(rollout(learner, rid, pseed, env.cfg.max_battery))

    low = summarize(low_rows)
    full = summarize(full_rows)
    battery_handover_gap = low["handover_before_return_fraction"] - full["handover_before_return_fraction"]
    battery_relay_retention_gap_t20 = full["residual_relay_at_t20_fraction"] - low["residual_relay_at_t20_fraction"]

    # Predeclared sanity gate: low battery must induce substantially more handover
    # and full battery must retain the original relay substantially more often.
    passed = bool(
        low["handover_before_return_fraction"] >= 0.60
        and full["handover_before_return_fraction"] <= 0.40
        and battery_handover_gap >= 0.30
        and battery_relay_retention_gap_t20 >= 0.30
    )

    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "backhaul_radius_cells": float(env.cfg.backhaul_radius_cells),
        "relay_anchor": list(RELAY_ANCHOR),
        "permutations_per_residual": int(args.permutations_per_residual),
        "low_battery": low,
        "full_battery": full,
        "battery_handover_gap": float(battery_handover_gap),
        "battery_relay_retention_gap_t20": float(battery_relay_retention_gap_t20),
        "sanity_gate": {
            "criteria": {
                "low_battery_handover_before_return_fraction_min": 0.60,
                "full_battery_handover_before_return_fraction_max": 0.40,
                "handover_gap_min": 0.30,
                "relay_retention_gap_t20_min": 0.30,
            },
            "passed": passed,
        },
        "low_battery_cases": low_rows,
        "full_battery_cases": full_rows,
    }
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("low_battery_cases", "full_battery_cases")}, indent=2))
    print(f"OUTPUT={out}")
    print("SANITY_PASS" if passed else "SANITY_FAIL")


if __name__ == "__main__":
    main()
