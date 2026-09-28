from __future__ import annotations

import argparse
import json
import os
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

from src.envs.v5_dynamic_tradeoff_env import V5DynamicTradeoffConfig, V5DynamicTradeoffEnv
from src.rl.qmix import ReplayBuffer, Transition
from src.rl.v5_relay_request_spatial_qmix import V5RelayRequestSpatialQMIXLearner


def make_env(seed: int, horizon: int, *, randomize_phase: bool) -> V5DynamicTradeoffEnv:
    return V5DynamicTradeoffEnv(
        V5DynamicTradeoffConfig(
            seed=seed,
            horizon=horizon,
            enable_ue_mobility=True,
            enable_battery_cycle=True,
            backhaul_radius_cells=4.0,
            initial_battery_min=55.0,
            initial_battery_max=100.0,
            randomize_start_identity=True,
            randomize_tradeoff_phase=randomize_phase,
            tradeoff_segment_slots=160,
            relay_request_on_ratio=0.75,
            relay_request_off_ratio=0.90,
            relay_min_dwell_slots=2,
        )
    )


def evaluate(learner, seeds: list[int], horizon: int) -> dict:
    rows = []
    for seed in seeds:
        env = make_env(seed, horizon, randomize_phase=False)
        out = env.reset(seed=seed)
        rng = np.random.default_rng(900000 + seed)
        rewards = []
        relay_counts = []
        access_counts = []
        open_requests = []
        role_changes = 0
        accepts = 0
        releases = 0
        return_events = 0
        rejoin_events = 0
        movement_steps = 0
        composition = Counter()
        regime_rewards = {"direct_friendly": [], "relay_stressed": [], "transition": []}
        regime_relays = {"direct_friendly": [], "relay_stressed": [], "transition": []}
        prev_status = env.status.copy()
        prev_pos = env.uav_positions.copy()

        while True:
            actions = learner.act(out["obs"], out["action_mask"], 0.0, rng)
            nxt = env.step(actions)
            info = nxt["info"]
            rewards.append(float(nxt["reward"]))
            relay_counts.append(int(info["num_relay"]))
            access_counts.append(int(info["num_access"]))
            open_requests.append(int(np.sum(info["request_open"])))
            role_changes += int(info["role_switches_this_slot"])
            accepts += len(info["accepted_relay_targets_this_slot"])
            releases += len(info["released_relay_ids_this_slot"])

            status = np.asarray(info["status"], dtype=int)
            pos = np.asarray(info["uav_positions"], dtype=int)
            return_events += int(np.sum((prev_status == 0) & (status == 1)))
            rejoin_events += int(np.sum((prev_status != 0) & (status == 0)))
            movement_steps += int(np.sum(np.any(pos != prev_pos, axis=1) & (prev_status == 0)))
            prev_status = status.copy()
            prev_pos = pos.copy()

            a, r = int(info["num_access"]), int(info["num_relay"])
            composition[f"{a}A{r}R"] += 1
            regime = str(info["tradeoff_regime"])
            regime_rewards[regime].append(float(nxt["reward"]))
            regime_relays[regime].append(int(info["num_relay"]))

            out = nxt
            if out["truncated"]:
                break

        T = len(rewards)
        rows.append({
            "seed": int(seed),
            "mean_utility": float(np.mean(rewards)),
            "final100": float(np.mean(rewards[-min(100, T):])),
            "mean_active_access": float(np.mean(access_counts)),
            "mean_active_relays": float(np.mean(relay_counts)),
            "mean_open_requests": float(np.mean(open_requests)),
            "role_changes": int(role_changes),
            "accept_count": int(accepts),
            "release_count": int(releases),
            "return_events": int(return_events),
            "rejoin_events": int(rejoin_events),
            "movement_steps": int(movement_steps),
            "composition_fraction": {k: float(v / T) for k, v in composition.items()},
            "regime_mean_utility": {
                k: (float(np.mean(v)) if v else None) for k, v in regime_rewards.items()
            },
            "regime_mean_relays": {
                k: (float(np.mean(v)) if v else None) for k, v in regime_relays.items()
            },
        })

    def avg(key: str) -> float:
        return float(np.mean([r[key] for r in rows]))

    regime_summary = {}
    for regime in ("direct_friendly", "relay_stressed", "transition"):
        us = [r["regime_mean_utility"][regime] for r in rows if r["regime_mean_utility"][regime] is not None]
        rs = [r["regime_mean_relays"][regime] for r in rows if r["regime_mean_relays"][regime] is not None]
        regime_summary[regime] = {
            "mean_utility": float(np.mean(us)) if us else None,
            "mean_relays": float(np.mean(rs)) if rs else None,
        }

    return {
        "seeds": [int(x) for x in seeds],
        "mean_utility": avg("mean_utility"),
        "std_utility_across_seeds": float(np.std([r["mean_utility"] for r in rows])),
        "final100": avg("final100"),
        "mean_active_access": avg("mean_active_access"),
        "mean_active_relays": avg("mean_active_relays"),
        "mean_open_requests": avg("mean_open_requests"),
        "mean_role_changes": avg("role_changes"),
        "mean_accept_count": avg("accept_count"),
        "mean_release_count": avg("release_count"),
        "mean_return_events": avg("return_events"),
        "mean_rejoin_events": avg("rejoin_events"),
        "mean_movement_steps": avg("movement_steps"),
        "regime_summary": regime_summary,
        "rows": rows,
    }


def save_full(path: Path, learner, steps: int, episodes: int, best: float, history: list[dict], rng: np.random.Generator):
    payload = {
        "agent": learner.agent.state_dict(),
        "mixer": learner.mixer.state_dict(),
        "target_agent": learner.target_agent.state_dict(),
        "target_mixer": learner.target_mixer.state_dict(),
        "optimizer": learner.optimizer.state_dict(),
        "steps": int(steps),
        "episodes": int(episodes),
        "best": float(best),
        "history": history,
        "numpy_state": np.random.get_state(),
        "python_state": random.getstate(),
        "torch_state": torch.get_rng_state(),
        "generator_state": rng.bit_generator.state,
    }
    if torch.cuda.is_available():
        payload["cuda_state"] = torch.cuda.get_rng_state_all()
    torch.save(payload, path)


def load_full(path: Path, learner, rng: np.random.Generator):
    payload = torch.load(path, map_location=learner.device, weights_only=False)
    learner.agent.load_state_dict(payload["agent"])
    learner.mixer.load_state_dict(payload["mixer"])
    learner.target_agent.load_state_dict(payload.get("target_agent", payload["agent"]))
    learner.target_mixer.load_state_dict(payload.get("target_mixer", payload["mixer"]))
    if "optimizer" in payload:
        learner.optimizer.load_state_dict(payload["optimizer"])
    if "numpy_state" in payload:
        np.random.set_state(payload["numpy_state"])
    if "python_state" in payload:
        random.setstate(payload["python_state"])
    if "torch_state" in payload:
        torch.set_rng_state(payload["torch_state"])
    if torch.cuda.is_available() and "cuda_state" in payload:
        torch.cuda.set_rng_state_all(payload["cuda_state"])
    if "generator_state" in payload:
        rng.bit_generator.state = payload["generator_state"]
    return (
        int(payload.get("steps", 0)),
        int(payload.get("episodes", 0)),
        float(payload.get("best", -1.0)),
        list(payload.get("history", [])),
    )


def default_write_root() -> Path:
    override = os.environ.get("UAV_OUTPUT_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    kaggle_work = Path("/kaggle/working")
    if kaggle_work.is_dir() and os.access(kaggle_work, os.W_OK):
        return kaggle_work
    if ROOT.is_dir() and os.access(ROOT, os.W_OK):
        return ROOT
    return Path.cwd().resolve()


def resolve_output_dir(value: str) -> Path:
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (default_write_root() / p).resolve()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total-transitions", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=81)
    ap.add_argument("--horizon", type=int, default=600)
    ap.add_argument("--eval-every", type=int, default=10000)
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--buffer-capacity", type=int, default=60000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=1024)
    ap.add_argument("--update-every", type=int, default=4)
    ap.add_argument("--target-every", type=int, default=200)
    ap.add_argument("--eps-decay-steps", type=int, default=60000)
    ap.add_argument("--train-seed-start", type=int, default=1101)
    ap.add_argument("--train-seed-count", type=int, default=20)
    ap.add_argument("--val-seeds", nargs="+", type=int, default=[2101, 2102, 2103, 2104, 2105])
    ap.add_argument("--outdir", type=str, default="results/v5_dynamic_tradeoff")
    ap.add_argument("--torch-threads", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--resume-from", type=str, default=None)
    args = ap.parse_args()

    torch.set_num_threads(max(1, args.torch_threads))
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)

    train_seeds = list(range(args.train_seed_start, args.train_seed_start + args.train_seed_count))
    outdir = resolve_output_dir(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    best_path = outdir / f"seed{args.seed}_best.pt"
    latest = outdir / f"seed{args.seed}_latest_full.pt"
    history_path = outdir / f"seed{args.seed}_history.json"

    env = make_env(train_seeds[0], args.horizon, randomize_phase=True)
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

    run_cfg = {
        **vars(args),
        "project_root": str(ROOT),
        "output_dir": str(outdir),
        "device": device,
        "train_seeds": train_seeds,
        "initialization": "scratch",
        "fixed_bs_topology": True,
        "backhaul_radius_cells": 4.0,
        "action_design": "0-4 movement preserve role; 5-8 AcceptRelay(other target); 9 ReleaseRelay",
        "relay_needed": "automatic direct-backhaul/access-offer hysteresis 0.75 on, 0.90 off",
        "reward": "mean user satisfaction only; unchanged",
        "mobility_benchmark": "same-map smooth traffic path alternates direct-friendly and relay-stressed regimes; per-UE local motion remains stochastic",
        "battery": "automatic Return/Charging/Rejoin; staggered 55..100 initial battery levels shuffled across UAV IDs each episode",
        "initial_roles": "5 Access; RL creates/releases Relay assignments",
    }
    (outdir / f"seed{args.seed}_run_config.json").write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")

    replay = ReplayBuffer(args.buffer_capacity, args.seed)
    steps = 0
    episodes = 0
    best = -1.0
    history = []
    resume_path = None
    if args.resume_from:
        rp = Path(args.resume_from).expanduser()
        resume_path = rp.resolve() if rp.is_absolute() else (Path.cwd() / rp).resolve()
    elif args.resume and latest.exists():
        resume_path = latest
    if resume_path is not None:
        steps, episodes, best, history = load_full(resume_path, learner, rng)
        print(f"RESUMED from={resume_path} steps={steps} episodes={episodes} best={best:.4f}", flush=True)
    next_eval = ((steps // args.eval_every) + 1) * args.eval_every
    next_save = ((steps // args.save_every) + 1) * args.save_every
    t0 = time.time()

    print(f"PROJECT_ROOT={ROOT}", flush=True)
    print(f"OUTPUT_DIR={outdir}", flush=True)
    print(f"DEVICE={device}", flush=True)
    print(f"TRAIN_SEEDS={train_seeds[0]}..{train_seeds[-1]}", flush=True)
    print(f"VAL_SEEDS={args.val_seeds}", flush=True)

    while steps < args.total_transitions:
        episodes += 1
        episode_seed = int(rng.choice(train_seeds))
        out = env.reset(seed=episode_seed)
        while steps < args.total_transitions:
            frac = min(steps / max(1, args.eps_decay_steps), 1.0)
            eps = 1.0 + frac * (0.05 - 1.0)
            actions = learner.act(out["obs"], out["action_mask"], eps, rng)
            nxt = env.step(actions)
            done = float(nxt["truncated"])
            replay.add(Transition(
                out["obs"], out["state"], np.asarray(actions, dtype=np.int64),
                float(nxt["reward"]), nxt["obs"], nxt["state"],
                nxt["action_mask"], done,
            ))
            steps += 1

            if len(replay) >= max(args.warmup, args.batch_size) and steps % args.update_every == 0:
                learner.train_batch(replay.sample(args.batch_size))
            if steps % args.target_every == 0:
                learner.update_targets()
            out = nxt

            if steps >= next_eval:
                ev = evaluate(learner, list(args.val_seeds), args.horizon)
                row = {
                    "steps": int(steps),
                    "episodes": int(episodes),
                    "epsilon": float(eps),
                    "mean_utility": ev["mean_utility"],
                    "std_utility_across_seeds": ev["std_utility_across_seeds"],
                    "final100": ev["final100"],
                    "mean_active_relays": ev["mean_active_relays"],
                    "mean_role_changes": ev["mean_role_changes"],
                    "mean_return_events": ev["mean_return_events"],
                    "mean_rejoin_events": ev["mean_rejoin_events"],
                    "direct_mean_relays": ev["regime_summary"]["direct_friendly"]["mean_relays"],
                    "relay_stressed_mean_relays": ev["regime_summary"]["relay_stressed"]["mean_relays"],
                    "runtime_s": float(time.time() - t0),
                }
                history.append(row)
                print("EVAL", json.dumps(row), flush=True)
                if ev["mean_utility"] > best:
                    best = ev["mean_utility"]
                    learner.save(str(best_path))
                history_path.write_text(json.dumps({"best": best, "history": history}, indent=2), encoding="utf-8")
                next_eval += args.eval_every

            if steps >= next_save:
                save_full(latest, learner, steps, episodes, best, history, rng)
                next_save += args.save_every

            if done:
                break

    save_full(latest, learner, steps, episodes, best, history, rng)
    if best_path.exists():
        learner.load(str(best_path))
    final_eval = evaluate(learner, list(args.val_seeds), args.horizon)
    summary = {
        "seed": int(args.seed),
        "steps": int(steps),
        "episodes": int(episodes),
        "best_validation": float(best),
        "validation": final_eval,
        "history": history,
    }
    (outdir / f"seed{args.seed}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("DONE", json.dumps({
        "best_validation": best,
        "mean_utility": final_eval["mean_utility"],
        "direct_mean_relays": final_eval["regime_summary"]["direct_friendly"]["mean_relays"],
        "relay_stressed_mean_relays": final_eval["regime_summary"]["relay_stressed"]["mean_relays"],
        "mean_return_events": final_eval["mean_return_events"],
        "mean_rejoin_events": final_eval["mean_rejoin_events"],
    }), flush=True)


if __name__ == "__main__":
    main()
