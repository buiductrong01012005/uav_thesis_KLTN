from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.envs.v4_fixed_mobile_battery_env import ACTIVE, CHARGING, RETURNING
from src.envs.v7_r6_macro_relay_stable_env import V7R6MacroRelayStableConfig
from src.envs.v8_r6_macro_relay_battery_env import V8R6MacroRelayBatteryConfig
from src.envs.v9_r6_macro_relay_battery_focus_env import V9R6MacroRelayBatteryFocusConfig
from src.envs.final_movement_only_env import (
    FinalMovementOnlyNoBatteryEnv,
    FinalMovementOnlyBatteryEnv,
    FinalMovementOnlyBatteryFocusEnv,
)
from src.rl.qmix import ReplayBuffer, Transition
from src.rl.v5_relay_request_spatial_qmix import V5RelayRequestSpatialQMIXLearner

TRAIN_SEEDS = tuple(range(1101, 1121))
VALIDATION_SEEDS = (2101, 2102, 2103, 2104, 2105)
TEST_SEEDS = tuple(range(3101, 3121))


def make_env(stage: int, seed: int, horizon: int, *, randomized=False, battery_min=24.0, battery_max=36.0):
    if stage == 1:
        return FinalMovementOnlyNoBatteryEnv(
            V7R6MacroRelayStableConfig(seed=seed, horizon=horizon, enable_battery_cycle=False)
        )
    if stage == 2:
        return FinalMovementOnlyBatteryEnv(
            V8R6MacroRelayBatteryConfig(
                seed=seed, horizon=horizon, residual_initial_battery=30.0,
                randomize_residual_uav=True,
            )
        )
    if stage == 3:
        return FinalMovementOnlyBatteryFocusEnv(
            V9R6MacroRelayBatteryFocusConfig(
                seed=seed, horizon=horizon,
                residual_initial_battery=30.0,
                randomize_residual_uav=True,
                randomize_residual_battery=bool(randomized),
                residual_battery_min=float(battery_min),
                residual_battery_max=float(battery_max),
            )
        )
    raise ValueError("stage must be 1, 2, or 3")


def make_learner(env, lr: float, device: str):
    return V5RelayRequestSpatialQMIXLearner(
        env.cfg.num_uavs, env.obs_dim, env.state_dim, env.n_actions,
        grid_size=env.cfg.grid_size, coarse_size=env.cfg.coarse_obs_size,
        lr=float(lr), gamma=0.99, hidden_dim=64, mixing_embed_dim=32,
        device=device,
    )


def _return_focus_state(env, focus_margin: float) -> bool:
    if np.any(env.status != ACTIVE):
        return True
    if not hasattr(env, "min_active_return_margin"):
        return False
    margin = env.min_active_return_margin()
    return bool(np.isfinite(margin) and margin <= float(focus_margin))


def _concat_batches(*batches: dict) -> dict:
    batches = tuple(b for b in batches if b is not None)
    return {k: np.concatenate([b[k] for b in batches], axis=0) for k in batches[0]}


def sample_focus_batch(main_replay, focus_replay, batch_size, focus_fraction):
    bs = int(batch_size)
    n_focus = min(int(round(bs * float(focus_fraction))), len(focus_replay))
    n_main = bs - n_focus
    if n_main > len(main_replay):
        n_main = len(main_replay)
        n_focus = min(bs - n_main, len(focus_replay))
    if n_main + n_focus < bs:
        n_main = min(bs, len(main_replay)); n_focus = 0
    main = main_replay.sample(n_main)
    if n_focus <= 0:
        return main, 0.0
    batch = _concat_batches(main, focus_replay.sample(n_focus))
    order = np.random.permutation(batch["reward"].shape[0])
    return {k: v[order] for k, v in batch.items()}, float(n_focus / len(order))


def evaluate(learner, stage: int, seeds, horizon=150, *, focus_margin=10.0, measure_time=False):
    rows = []
    decision_ns = 0
    decision_calls = 0
    for seed in seeds:
        env = make_env(stage, seed, horizon, randomized=False)
        out = env.reset(seed=seed)
        rng = np.random.default_rng(970000 + int(seed))
        rewards, pre, unavailable, post = [], [], [], []
        active_counts = []
        seen_unavailable = False
        seen_rejoin = False
        return_events = charge_arrivals = rejoin_events = 0
        while True:
            status_before = env.status.copy()
            active_before = int(np.sum(status_before == ACTIVE))
            if active_before < env.cfg.num_uavs:
                seen_unavailable = True
            elif seen_unavailable:
                seen_rejoin = True

            t0 = time.perf_counter_ns()
            actions = learner.act(out["obs"], out["action_mask"], 0.0, rng)
            if measure_time:
                decision_ns += time.perf_counter_ns() - t0
                decision_calls += 1
            nxt = env.step(actions)
            reward = float(nxt["reward"])
            rewards.append(reward)
            active_counts.append(active_before)
            if not seen_unavailable:
                pre.append(reward)
            elif active_before < env.cfg.num_uavs:
                unavailable.append(reward)
            elif seen_rejoin:
                post.append(reward)

            status_after = env.status.copy()
            return_events += int(np.sum((status_before == ACTIVE) & (status_after == RETURNING)))
            charge_arrivals += int(np.sum((status_before == RETURNING) & (status_after == CHARGING)))
            rejoin_events += int(np.sum((status_before == CHARGING) & (status_after == ACTIVE)))
            out = nxt
            if nxt["truncated"]:
                break
        rows.append({
            "seed": int(seed),
            "mean_utility": float(np.mean(rewards)),
            "final20": float(np.mean(rewards[-20:])),
            "pre_return_mean_utility": float(np.mean(pre)) if pre else None,
            "unavailable_mean_utility": float(np.mean(unavailable)) if unavailable else None,
            "post_rejoin_mean_utility": float(np.mean(post)) if post else None,
            "mean_active_uavs": float(np.mean(active_counts)),
            "return_events": int(return_events),
            "charge_arrivals": int(charge_arrivals),
            "rejoin_events": int(rejoin_events),
        })

    def mean_optional(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return float(np.mean(vals)) if vals else None

    return {
        "mean_utility": float(np.mean([r["mean_utility"] for r in rows])),
        "std_utility": float(np.std([r["mean_utility"] for r in rows])),
        "pre_return_mean_utility": mean_optional("pre_return_mean_utility"),
        "unavailable_mean_utility": mean_optional("unavailable_mean_utility"),
        "post_rejoin_mean_utility": mean_optional("post_rejoin_mean_utility"),
        "mean_active_uavs": float(np.mean([r["mean_active_uavs"] for r in rows])),
        "mean_return_events": float(np.mean([r["return_events"] for r in rows])),
        "mean_charge_arrivals": float(np.mean([r["charge_arrivals"] for r in rows])),
        "mean_rejoin_events": float(np.mean([r["rejoin_events"] for r in rows])),
        "decision_ms_per_slot": float(decision_ns / max(1, decision_calls) / 1e6) if measure_time else None,
        "rows": rows,
    }


def train_stage(*, stage, seed, total_transitions, horizon, eval_every, outdir, init_checkpoint,
                lr, eps_start, eps_end, eps_decay_steps, warmup, batch_size, buffer_capacity,
                update_every, target_every, device, rng, focus_buffer_capacity=15000,
                focus_fraction=0.5, focus_margin=10.0, post_rejoin_focus_slots=12,
                residual_battery_min=24.0, residual_battery_max=36.0):
    train_env = make_env(
        stage, TRAIN_SEEDS[0], horizon,
        randomized=(stage == 3), battery_min=residual_battery_min, battery_max=residual_battery_max,
    )
    learner = make_learner(train_env, lr, device)
    if init_checkpoint is not None:
        learner.load(str(init_checkpoint))

    replay = ReplayBuffer(buffer_capacity, seed + stage * 1000)
    focus_replay = ReplayBuffer(focus_buffer_capacity, seed + stage * 1000 + 77) if stage == 3 else None
    steps = episodes = 0
    best = -1.0
    best_step = None
    history = []
    interval_losses = []
    interval_focus_batches = []
    t_start = time.time()
    checkpoint = outdir / f"seed{seed}_stage{stage}_best.pt"

    while steps < total_transitions:
        episodes += 1
        scenario_seed = int(rng.choice(TRAIN_SEEDS))
        env = make_env(
            stage, scenario_seed, horizon,
            randomized=(stage == 3), battery_min=residual_battery_min, battery_max=residual_battery_max,
        )
        out = env.reset(seed=scenario_seed)
        post_focus_left = 0
        while steps < total_transitions:
            frac = min(steps / max(1, eps_decay_steps), 1.0)
            eps = float(eps_start + frac * (eps_end - eps_start))
            focus_before = _return_focus_state(env, focus_margin) if stage == 3 else False
            status_before = env.status.copy()
            actions = learner.act(out["obs"], out["action_mask"], eps, rng)
            nxt = env.step(actions)
            status_after = env.status.copy()
            n_return = int(np.sum((status_before == ACTIVE) & (status_after == RETURNING)))
            n_rejoin = int(np.sum((status_before == CHARGING) & (status_after == ACTIVE)))
            if n_rejoin:
                post_focus_left = int(post_rejoin_focus_slots)

            tr = Transition(
                out["obs"], out["state"], np.asarray(actions, dtype=np.int64),
                float(nxt["reward"]), nxt["obs"], nxt["state"], nxt["action_mask"],
                float(nxt["truncated"]),
            )
            replay.add(tr)
            if stage == 3:
                focus_after = _return_focus_state(env, focus_margin)
                if focus_before or focus_after or n_return or n_rejoin or post_focus_left > 0:
                    focus_replay.add(tr)
                if post_focus_left > 0:
                    post_focus_left -= 1

            steps += 1
            if len(replay) >= max(warmup, batch_size) and steps % update_every == 0:
                if stage == 3:
                    batch, ff = sample_focus_batch(replay, focus_replay, batch_size, focus_fraction)
                    interval_focus_batches.append(ff)
                else:
                    batch = replay.sample(batch_size)
                interval_losses.append(float(learner.train_batch(batch)))
            if steps % target_every == 0:
                learner.update_targets()
            out = nxt

            if steps % eval_every == 0:
                val = evaluate(learner, stage, VALIDATION_SEEDS, horizon, focus_margin=focus_margin)
                row = {
                    "steps": int(steps), "episodes": int(episodes), "epsilon": eps,
                    "validation_mean_utility": val["mean_utility"],
                    "validation_pre_return": val["pre_return_mean_utility"],
                    "validation_unavailable": val["unavailable_mean_utility"],
                    "validation_post_rejoin": val["post_rejoin_mean_utility"],
                    "train_td_loss_mean": float(np.mean(interval_losses)) if interval_losses else None,
                    "train_batch_focus_fraction": float(np.mean(interval_focus_batches)) if interval_focus_batches else 0.0,
                    "runtime_s": float(time.time() - t_start),
                }
                history.append(row)
                print(f"STAGE{stage}_VAL", json.dumps(row), flush=True)
                if val["mean_utility"] > best:
                    best = float(val["mean_utility"])
                    best_step = int(steps)
                    learner.save(str(checkpoint))
                interval_losses.clear(); interval_focus_batches.clear()
            if nxt["truncated"]:
                break

    if not checkpoint.exists():
        raise RuntimeError(f"No checkpoint saved for stage {stage}")
    learner.load(str(checkpoint))
    final_val = evaluate(learner, stage, VALIDATION_SEEDS, horizon, focus_margin=focus_margin)
    return learner, checkpoint, {
        "stage": int(stage),
        "best_validation": float(best),
        "best_step": int(best_step),
        "history": history,
        "validation": final_val,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=91)
    ap.add_argument("--horizon", type=int, default=150)
    ap.add_argument("--stage1-transitions", type=int, default=30000)
    ap.add_argument("--stage2-transitions", type=int, default=30000)
    ap.add_argument("--stage3-transitions", type=int, default=20000)
    ap.add_argument("--eval-every", type=int, default=5000)
    ap.add_argument("--buffer-capacity", type=int, default=30000)
    ap.add_argument("--focus-buffer-capacity", type=int, default=15000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=3000)
    ap.add_argument("--update-every", type=int, default=4)
    ap.add_argument("--target-every", type=int, default=1000)
    ap.add_argument("--focus-fraction", type=float, default=0.5)
    ap.add_argument("--focus-margin", type=float, default=10.0)
    ap.add_argument("--post-rejoin-focus-slots", type=int, default=12)
    ap.add_argument("--residual-battery-min", type=float, default=24.0)
    ap.add_argument("--residual-battery-max", type=float, default=36.0)
    ap.add_argument("--outdir", default="results/final_movement_only_spatial_qmix")
    ap.add_argument("--torch-threads", type=int, default=1)
    args = ap.parse_args()

    torch.set_num_threads(max(1, args.torch_threads))
    np.random.seed(args.seed); random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)

    outdir = Path(args.outdir)
    if not outdir.is_absolute(): outdir = ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    common = dict(
        seed=args.seed, horizon=args.horizon, eval_every=args.eval_every, outdir=outdir,
        warmup=args.warmup, batch_size=args.batch_size, buffer_capacity=args.buffer_capacity,
        update_every=args.update_every, target_every=args.target_every, device=device, rng=rng,
        focus_buffer_capacity=args.focus_buffer_capacity, focus_fraction=args.focus_fraction,
        focus_margin=args.focus_margin, post_rejoin_focus_slots=args.post_rejoin_focus_slots,
        residual_battery_min=args.residual_battery_min, residual_battery_max=args.residual_battery_max,
    )

    _, ck1, s1 = train_stage(
        stage=1, total_transitions=args.stage1_transitions, init_checkpoint=None,
        lr=3e-4, eps_start=1.0, eps_end=0.05, eps_decay_steps=25000, **common,
    )
    _, ck2, s2 = train_stage(
        stage=2, total_transitions=args.stage2_transitions, init_checkpoint=ck1,
        lr=1e-4, eps_start=0.35, eps_end=0.05, eps_decay_steps=20000, **common,
    )
    learner, ck3, s3 = train_stage(
        stage=3, total_transitions=args.stage3_transitions, init_checkpoint=ck2,
        lr=5e-5, eps_start=0.2, eps_end=0.05, eps_decay_steps=15000, **common,
    )

    test = evaluate(learner, 3, TEST_SEEDS, args.horizon, focus_margin=args.focus_margin, measure_time=True)
    summary = {
        "protocol": "independent 5-action movement-only Spatial-QMIX with matched V7->V8->V9 curriculum budget",
        "seed": int(args.seed),
        "n_actions": 5,
        "reward_changed": False,
        "scenario_changed": False,
        "total_training_transitions": int(args.stage1_transitions + args.stage2_transitions + args.stage3_transitions),
        "stages": [s1, s2, s3],
        "heldout_test": test,
        "final_checkpoint": str(ck3),
        "train_seeds": list(TRAIN_SEEDS),
        "validation_seeds": list(VALIDATION_SEEDS),
        "test_seeds": list(TEST_SEEDS),
    }
    (outdir / f"seed{args.seed}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (outdir / f"seed{args.seed}_run_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print("FINAL", json.dumps({
        "seed": args.seed,
        "test_mean": test["mean_utility"],
        "test_std": test["std_utility"],
        "decision_ms_per_slot": test["decision_ms_per_slot"],
        "checkpoint": str(ck3),
    }), flush=True)


if __name__ == "__main__":
    main()
