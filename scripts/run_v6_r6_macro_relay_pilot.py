from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.envs.v6_r6_macro_relay_mobility_env import (
    V6R6MacroRelayMobilityConfig,
    V6R6MacroRelayMobilityEnv,
)
from src.rl.qmix import ReplayBuffer, Transition
from src.rl.v5_relay_request_spatial_qmix import V5RelayRequestSpatialQMIXLearner

TRAIN_SEEDS = tuple(range(1101, 1121))
VALIDATION_SEEDS = (2101, 2102, 2103, 2104, 2105)
TEST_SEEDS = tuple(range(3101, 3121))


def make_env(seed: int, horizon: int = 150):
    return V6R6MacroRelayMobilityEnv(
        V6R6MacroRelayMobilityConfig(seed=seed, horizon=horizon)
    )


def movement_only_mask(env):
    mask = env.get_action_mask().copy()
    mask[:, 5:] = 0
    return mask


def evaluate(learner, seeds, horizon=150, *, movement_only=False):
    rows = []
    for seed in seeds:
        env = make_env(seed, horizon)
        out = env.reset(seed=seed)
        rng = np.random.default_rng(900000 + int(seed))
        rewards = []
        relays = []
        macro_count = conflicts = role_changes = 0
        target_counts = Counter()
        assisted_move_counts = Counter()
        prev_roles = env.roles.copy()

        while True:
            mask = movement_only_mask(env) if movement_only else out["action_mask"]
            actions = learner.act(out["obs"], mask, 0.0, rng)
            out = env.step(actions)
            info = out["info"]
            rewards.append(float(out["reward"]))
            relays.append(int(info["num_relay"]))
            role_changes += int(np.sum(env.roles != prev_roles))
            prev_roles = env.roles.copy()
            accepted = info.get("accepted_macro_relays_this_slot", [])
            macro_count += len(accepted)
            conflicts += int(info.get("relay_claim_conflicts_this_slot", 0))
            for actor, target, move in accepted:
                target_counts[int(target)] += 1
                assisted_move_counts[int(move)] += 1
            if out["truncated"]:
                break

        rows.append({
            "seed": int(seed),
            "mean_utility": float(np.mean(rewards)),
            "final20": float(np.mean(rewards[-20:])),
            "mean_relays": float(np.mean(relays)),
            "relay_count_switches": int(sum(a != b for a, b in zip(relays, relays[1:]))),
            "role_changes": int(role_changes),
            "macro_relay_count": int(macro_count),
            "claim_conflicts": int(conflicts),
            "target_counts": {str(k): int(v) for k, v in sorted(target_counts.items())},
            "assisted_move_counts": {str(k): int(v) for k, v in sorted(assisted_move_counts.items())},
        })

    return {
        "mean_utility": float(np.mean([r["mean_utility"] for r in rows])),
        "std_utility": float(np.std([r["mean_utility"] for r in rows])),
        "mean_relays": float(np.mean([r["mean_relays"] for r in rows])),
        "mean_role_changes": float(np.mean([r["role_changes"] for r in rows])),
        "mean_macro_relay_count": float(np.mean([r["macro_relay_count"] for r in rows])),
        "mean_claim_conflicts": float(np.mean([r["claim_conflicts"] for r in rows])),
        "rows": rows,
    }


def compare(dynamic, movement_only):
    paired = []
    wins = 0
    for d, m in zip(dynamic["rows"], movement_only["rows"]):
        if d["seed"] != m["seed"]:
            raise ValueError("Evaluation seed order mismatch")
        gain = float(d["mean_utility"] - m["mean_utility"])
        wins += int(gain > 0.0)
        paired.append({
            "seed": int(d["seed"]),
            "macro_dynamic": float(d["mean_utility"]),
            "movement_only_5A": float(m["mean_utility"]),
            "gain_pp": float(100.0 * gain),
        })
    delta = float(dynamic["mean_utility"] - movement_only["mean_utility"])
    return {
        "dynamic_minus_5A_pp": float(100.0 * delta),
        "win_fraction": float(wins / max(1, len(paired))),
        "wins": int(wins),
        "n": int(len(paired)),
        "paired_rows": paired,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total-transitions", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=91)
    ap.add_argument("--horizon", type=int, default=150)
    ap.add_argument("--eval-every", type=int, default=5000)
    ap.add_argument("--buffer-capacity", type=int, default=30000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=512)
    ap.add_argument("--update-every", type=int, default=4)
    ap.add_argument("--target-every", type=int, default=200)
    ap.add_argument("--eps-decay-steps", type=int, default=10000)
    ap.add_argument("--outdir", default="results/v6_r6_macro_relay_pilot")
    ap.add_argument("--torch-threads", type=int, default=1)
    args = ap.parse_args()

    torch.set_num_threads(max(1, args.torch_threads))
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)

    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "total_transitions": int(args.total_transitions),
        "seed": int(args.seed),
        "horizon": int(args.horizon),
        "eval_every": int(args.eval_every),
        "buffer_capacity": int(args.buffer_capacity),
        "batch_size": int(args.batch_size),
        "warmup": int(args.warmup),
        "update_every": int(args.update_every),
        "target_every": int(args.target_every),
        "eps_decay_steps": int(args.eps_decay_steps),
        "train_seeds": list(TRAIN_SEEDS),
        "validation_seeds": list(VALIDATION_SEEDS),
        "test_seeds": list(TEST_SEEDS),
        "fresh_from_scratch": True,
        "checkpoint_selection": "highest macro-dynamic mean utility on validation seeds only",
        "test_policy": "held-out test seeds evaluated only after checkpoint selection",
        "action_design": "0-4 Access movement; 5-8 RelayTo(target) + deterministic assisted one-step Relay movement",
        "relay_persistence": False,
        "release_action": False,
        "routing": "target-exclusive designated Relay only",
    }
    (outdir / f"seed{args.seed}_run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    env = make_env(TRAIN_SEEDS[0], args.horizon)
    learner = V5RelayRequestSpatialQMIXLearner(
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
        device=device,
    )
    replay = ReplayBuffer(args.buffer_capacity, args.seed)
    steps = episodes = 0
    best = -1.0
    best_step = None
    history = []
    t0 = time.time()

    print("SCENARIO=V4 map + R6 + natural UE mobility + V6 macro RelayTo(target); battery off", flush=True)
    print(f"DEVICE={device}", flush=True)

    while steps < args.total_transitions:
        episodes += 1
        scenario_seed = int(rng.choice(TRAIN_SEEDS))
        out = env.reset(seed=scenario_seed)
        while steps < args.total_transitions:
            frac = min(steps / max(1, args.eps_decay_steps), 1.0)
            eps = 1.0 + frac * (0.05 - 1.0)
            actions = learner.act(out["obs"], out["action_mask"], eps, rng)
            nxt = env.step(actions)
            done = float(nxt["truncated"])
            replay.add(Transition(
                out["obs"], out["state"], np.asarray(actions, dtype=np.int64),
                float(nxt["reward"]), nxt["obs"], nxt["state"], nxt["action_mask"], done,
            ))
            steps += 1
            if len(replay) >= max(args.warmup, args.batch_size) and steps % args.update_every == 0:
                learner.train_batch(replay.sample(args.batch_size))
            if steps % args.target_every == 0:
                learner.update_targets()
            out = nxt

            if steps % args.eval_every == 0:
                dyn = evaluate(learner, VALIDATION_SEEDS, args.horizon, movement_only=False)
                no_relay = evaluate(learner, VALIDATION_SEEDS, args.horizon, movement_only=True)
                cmp = compare(dyn, no_relay)
                row = {
                    "steps": int(steps),
                    "episodes": int(episodes),
                    "epsilon": float(eps),
                    "validation_dynamic_mean_utility": dyn["mean_utility"],
                    "validation_5A_mean_utility": no_relay["mean_utility"],
                    "validation_dynamic_minus_5A_pp": cmp["dynamic_minus_5A_pp"],
                    "validation_win_fraction": cmp["win_fraction"],
                    "dynamic_mean_relays": dyn["mean_relays"],
                    "dynamic_mean_role_changes": dyn["mean_role_changes"],
                    "dynamic_mean_macro_relay_count": dyn["mean_macro_relay_count"],
                    "dynamic_mean_claim_conflicts": dyn["mean_claim_conflicts"],
                    "runtime_s": float(time.time() - t0),
                }
                history.append(row)
                print("VAL", json.dumps(row), flush=True)
                if dyn["mean_utility"] > best:
                    best = dyn["mean_utility"]
                    best_step = int(steps)
                    learner.save(str(outdir / f"seed{args.seed}_best.pt"))
                (outdir / f"seed{args.seed}_history.json").write_text(
                    json.dumps({"best": best, "best_step": best_step, "history": history}, indent=2),
                    encoding="utf-8",
                )
            if done:
                break

    best_path = outdir / f"seed{args.seed}_best.pt"
    if not best_path.exists():
        raise RuntimeError("No best checkpoint was saved")
    learner.load(str(best_path))

    val_dyn = evaluate(learner, VALIDATION_SEEDS, args.horizon, movement_only=False)
    val_5a = evaluate(learner, VALIDATION_SEEDS, args.horizon, movement_only=True)
    val_cmp = compare(val_dyn, val_5a)

    test_dyn = evaluate(learner, TEST_SEEDS, args.horizon, movement_only=False)
    test_5a = evaluate(learner, TEST_SEEDS, args.horizon, movement_only=True)
    test_cmp = compare(test_dyn, test_5a)

    gate = {
        "positive_test_mean_gain": bool(test_cmp["dynamic_minus_5A_pp"] > 0.0),
        "majority_test_seeds_won": bool(test_cmp["win_fraction"] > 0.5),
        "macro_relay_actually_used": bool(test_dyn["mean_relays"] > 0.1 and test_dyn["mean_macro_relay_count"] > 0.0),
        "test_gain_at_least_1pp": bool(test_cmp["dynamic_minus_5A_pp"] >= 1.0),
    }
    gate["pass_core_gate"] = bool(
        gate["positive_test_mean_gain"]
        and gate["majority_test_seeds_won"]
        and gate["macro_relay_actually_used"]
    )

    summary = {
        "seed": int(args.seed),
        "steps": int(steps),
        "episodes": int(episodes),
        "best_validation": float(best),
        "best_step": best_step,
        "validation": {"dynamic": val_dyn, "movement_only_5A": val_5a, "comparison": val_cmp},
        "heldout_test": {"dynamic": test_dyn, "movement_only_5A": test_5a, "comparison": test_cmp},
        "pilot_gate": gate,
        "scenario": {
            "map": "canonical V4 topology",
            "backhaul_radius_cells": 6.0,
            "ue_mobility": "canonical stochastic correlated-group mobility",
            "battery": False,
            "action_design": "0-4 Access movement; 5-8 RelayTo(target) with assisted one-step movement",
            "relay_persistence": False,
            "release_action": False,
            "routing": "target-exclusive designated Relay; no opportunistic reuse",
            "train_seeds": list(TRAIN_SEEDS),
            "validation_seeds": list(VALIDATION_SEEDS),
            "heldout_test_seeds": list(TEST_SEEDS),
        },
    }
    (outdir / f"seed{args.seed}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("DONE", json.dumps({
        "best_validation": best,
        "best_step": best_step,
        "validation_gain_pp": val_cmp["dynamic_minus_5A_pp"],
        "test_gain_pp": test_cmp["dynamic_minus_5A_pp"],
        "test_win_fraction": test_cmp["win_fraction"],
        "pass_core_gate": gate["pass_core_gate"],
    }), flush=True)


if __name__ == "__main__":
    main()
