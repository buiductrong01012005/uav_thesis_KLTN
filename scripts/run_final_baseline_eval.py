from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.envs.v4_fixed_mobile_battery_env import ACTIVE, CHARGING, RETURNING
from src.envs.v9_r6_macro_relay_battery_focus_env import (
    V9R6MacroRelayBatteryFocusConfig,
    V9R6MacroRelayBatteryFocusEnv,
)
from src.envs.final_movement_only_env import FinalMovementOnlyBatteryFocusEnv
from src.rl.v5_relay_request_spatial_qmix import V5RelayRequestSpatialQMIXLearner
from src.baselines.final_fast_greedy import FinalFastGreedyPolicy

TEST_SEEDS = tuple(range(3101, 3121))


def make_v9_env(seed: int, horizon: int):
    return V9R6MacroRelayBatteryFocusEnv(
        V9R6MacroRelayBatteryFocusConfig(
            seed=seed, horizon=horizon, residual_initial_battery=30.0,
            randomize_residual_uav=True, randomize_residual_battery=False,
        )
    )


def make_move_env(seed: int, horizon: int):
    return FinalMovementOnlyBatteryFocusEnv(
        V9R6MacroRelayBatteryFocusConfig(
            seed=seed, horizon=horizon, residual_initial_battery=30.0,
            randomize_residual_uav=True, randomize_residual_battery=False,
        )
    )


def make_learner(env, checkpoint: Path, n_actions: int, device: str):
    learner = V5RelayRequestSpatialQMIXLearner(
        env.cfg.num_uavs, env.obs_dim, env.state_dim, n_actions,
        grid_size=env.cfg.grid_size, coarse_size=env.cfg.coarse_obs_size,
        lr=5e-5, gamma=0.99, hidden_dim=64, mixing_embed_dim=32, device=device,
    )
    learner.load(str(checkpoint))
    return learner


def _sync(device: str):
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def evaluate_policy(name, env_factory, act_builder, seeds, horizon, device="cpu"):
    rows = []
    total_decision_ns = 0
    total_decisions = 0
    for seed in seeds:
        env = env_factory(int(seed), horizon)
        out = env.reset(seed=int(seed))
        act = act_builder(env, int(seed))
        rewards, pre, unavail, post = [], [], [], []
        seen_unavailable = False
        seen_rejoin = False
        returns = charges = rejoins = 0
        while True:
            before = env.status.copy()
            n_active = int(np.sum(before == ACTIVE))
            if n_active < env.cfg.num_uavs:
                seen_unavailable = True
            elif seen_unavailable:
                seen_rejoin = True

            _sync(device)
            t0 = time.perf_counter_ns()
            actions = act(out)
            _sync(device)
            total_decision_ns += time.perf_counter_ns() - t0
            total_decisions += 1

            nxt = env.step(actions)
            reward = float(nxt["reward"])
            rewards.append(reward)
            if not seen_unavailable:
                pre.append(reward)
            elif n_active < env.cfg.num_uavs:
                unavail.append(reward)
            elif seen_rejoin:
                post.append(reward)

            after = env.status.copy()
            returns += int(np.sum((before == ACTIVE) & (after == RETURNING)))
            charges += int(np.sum((before == RETURNING) & (after == CHARGING)))
            rejoins += int(np.sum((before == CHARGING) & (after == ACTIVE)))
            out = nxt
            if nxt["truncated"]:
                break
        rows.append({
            "seed": int(seed),
            "mean_utility": float(np.mean(rewards)),
            "final20": float(np.mean(rewards[-20:])),
            "pre_return_mean_utility": float(np.mean(pre)) if pre else None,
            "unavailable_mean_utility": float(np.mean(unavail)) if unavail else None,
            "post_rejoin_mean_utility": float(np.mean(post)) if post else None,
            "return_events": int(returns), "charge_arrivals": int(charges), "rejoin_events": int(rejoins),
        })

    def opt_mean(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return float(np.mean(vals)) if vals else None

    return {
        "name": name,
        "mean_utility": float(np.mean([r["mean_utility"] for r in rows])),
        "std_utility": float(np.std([r["mean_utility"] for r in rows])),
        "pre_return_mean_utility": opt_mean("pre_return_mean_utility"),
        "unavailable_mean_utility": opt_mean("unavailable_mean_utility"),
        "post_rejoin_mean_utility": opt_mean("post_rejoin_mean_utility"),
        "decision_ms_per_slot": float(total_decision_ns / max(1, total_decisions) / 1e6),
        "mean_return_events": float(np.mean([r["return_events"] for r in rows])),
        "mean_charge_arrivals": float(np.mean([r["charge_arrivals"] for r in rows])),
        "mean_rejoin_events": float(np.mean([r["rejoin_events"] for r in rows])),
        "rows": rows,
    }


def paired_compare(a, b, label_a="a", label_b="b"):
    brow = {r["seed"]: r for r in b["rows"]}
    paired = []
    wins = 0
    for ra in a["rows"]:
        rb = brow[ra["seed"]]
        gain = float(ra["mean_utility"] - rb["mean_utility"])
        wins += int(gain > 0)
        paired.append({
            "seed": int(ra["seed"]), label_a: float(ra["mean_utility"]),
            label_b: float(rb["mean_utility"]), "gain_pp": float(100 * gain),
        })
    return {
        "mean_gain_pp": float(100 * (a["mean_utility"] - b["mean_utility"])),
        "win_fraction": float(wins / len(paired)), "wins": int(wins), "n": len(paired),
        "paired_rows": paired,
    }


def discover(root: Path, pattern: str, seed: int):
    p = root / pattern.format(seed=seed)
    if not p.exists():
        raise FileNotFoundError(f"Missing required checkpoint: {p}")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[91, 92, 93])
    ap.add_argument("--horizon", type=int, default=150)
    ap.add_argument("--outdir", default="results/final_evaluation")
    ap.add_argument("--torch-threads", type=int, default=1)
    ap.add_argument("--v9-pattern", default="results/v9_r6_battery_transition_curriculum/seed{seed}_best.pt")
    ap.add_argument("--move-pattern", default="results/final_movement_only_spatial_qmix/seed{seed}_stage3_best.pt")
    args = ap.parse_args()

    torch.set_num_threads(max(1, args.torch_threads))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    outdir = Path(args.outdir)
    if not outdir.is_absolute(): outdir = ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    learned_rows = []
    for train_seed in args.seeds:
        v9_ck = discover(ROOT, args.v9_pattern, train_seed)
        mv_ck = discover(ROOT, args.move_pattern, train_seed)

        v9_probe = make_v9_env(TEST_SEEDS[0], args.horizon)
        v9_learner = make_learner(v9_probe, v9_ck, 10, device)
        v9_rngs = {}
        def build_v9(env, seed):
            rng = np.random.default_rng(880000 + seed + train_seed * 1000)
            return lambda out: v9_learner.act(out["obs"], out["action_mask"], 0.0, rng)
        v9_eval = evaluate_policy(f"V9_seed{train_seed}", make_v9_env, build_v9, TEST_SEEDS, args.horizon, device)

        mv_probe = make_move_env(TEST_SEEDS[0], args.horizon)
        mv_learner = make_learner(mv_probe, mv_ck, 5, device)
        def build_mv(env, seed):
            rng = np.random.default_rng(890000 + seed + train_seed * 1000)
            return lambda out: mv_learner.act(out["obs"], out["action_mask"], 0.0, rng)
        mv_eval = evaluate_policy(f"Move5_seed{train_seed}", make_move_env, build_mv, TEST_SEEDS, args.horizon, device)

        cmp = paired_compare(v9_eval, mv_eval, "v9", "movement_only")
        learned_rows.append({
            "training_seed": int(train_seed), "v9": v9_eval,
            "movement_only_spatial_qmix": mv_eval, "comparison": cmp,
        })
        print("LEARNED", json.dumps({
            "seed": train_seed, "v9": v9_eval["mean_utility"], "move5": mv_eval["mean_utility"],
            "gain_pp": cmp["mean_gain_pp"], "win_fraction": cmp["win_fraction"],
        }), flush=True)

    def build_random10(env, seed):
        rng = np.random.default_rng(910000 + seed)
        def act(out):
            return [int(rng.choice(np.flatnonzero(out["action_mask"][i]))) for i in range(env.cfg.num_uavs)]
        return act
    random10 = evaluate_policy("Random10", make_v9_env, build_random10, TEST_SEEDS, args.horizon, device="cpu")

    def build_random5(env, seed):
        rng = np.random.default_rng(920000 + seed)
        def act(out):
            return [int(rng.choice(np.flatnonzero(out["action_mask"][i]))) for i in range(env.cfg.num_uavs)]
        return act
    random5 = evaluate_policy("Random5", make_move_env, build_random5, TEST_SEEDS, args.horizon, device="cpu")

    def build_greedy(env, seed):
        policy = FinalFastGreedyPolicy(env, allow_relay=True)
        return lambda out: policy.act(out["action_mask"])
    greedy = evaluate_policy("FastGreedy", make_v9_env, build_greedy, TEST_SEEDS, args.horizon, device="cpu")

    gains = [r["comparison"]["mean_gain_pp"] for r in learned_rows]
    wins = [r["comparison"]["win_fraction"] for r in learned_rows]
    v9_means = [r["v9"]["mean_utility"] for r in learned_rows]
    move_means = [r["movement_only_spatial_qmix"]["mean_utility"] for r in learned_rows]
    summary = {
        "protocol": "final held-out baseline evaluation; test seeds 3101-3120; independent 5-action RL baseline",
        "training_seeds": list(args.seeds),
        "learned": learned_rows,
        "nonlearned": {"random10": random10, "random5": random5, "fast_greedy": greedy},
        "across_training_seeds": {
            "v9_mean": float(np.mean(v9_means)),
            "v9_std_across_training_seeds": float(np.std(v9_means)),
            "movement_only_mean": float(np.mean(move_means)),
            "movement_only_std_across_training_seeds": float(np.std(move_means)),
            "v9_minus_movement_only_pp_mean": float(np.mean(gains)),
            "v9_minus_movement_only_pp_std": float(np.std(gains)),
            "paired_win_fraction_mean": float(np.mean(wins)),
            "positive_gain_training_seed_fraction": float(np.mean(np.asarray(gains) > 0.0)),
        },
        "runtime_note": "decision_ms_per_slot measures policy action computation only, excluding env.step/network simulation",
    }
    (outdir / "final_baseline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with (outdir / "final_baseline_table.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "training_seed", "mean_utility", "std_utility", "decision_ms_per_slot"])
        for r in learned_rows:
            w.writerow(["V9", r["training_seed"], r["v9"]["mean_utility"], r["v9"]["std_utility"], r["v9"]["decision_ms_per_slot"]])
            w.writerow(["MovementOnlySpatialQMIX", r["training_seed"], r["movement_only_spatial_qmix"]["mean_utility"], r["movement_only_spatial_qmix"]["std_utility"], r["movement_only_spatial_qmix"]["decision_ms_per_slot"]])
        for key, r in (("Random10", random10), ("Random5", random5), ("FastGreedy", greedy)):
            w.writerow([key, "-", r["mean_utility"], r["std_utility"], r["decision_ms_per_slot"]])

    print("FINAL_SUMMARY", json.dumps(summary["across_training_seeds"], indent=2), flush=True)
    print("NONLEARNED", json.dumps({
        "random10": random10["mean_utility"], "random5": random5["mean_utility"],
        "fast_greedy": greedy["mean_utility"], "fast_greedy_ms": greedy["decision_ms_per_slot"],
    }), flush=True)

if __name__ == "__main__":
    main()
