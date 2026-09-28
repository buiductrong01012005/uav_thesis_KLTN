from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.envs.v0_env import ACCESS, RELAY
from src.envs.v4_fixed_mobile_battery_env import ACTIVE, RETURNING, CHARGING
from src.envs.v4_toggle_role_env import (
    STAY,
    TOGGLE_ROLE,
    V4ToggleRoleA1Config,
    V4ToggleRoleA1Env,
)
from src.rl.qmix import ReplayBuffer, Transition
from src.rl.v4_hardradius_spatial_qmix import V4HardRadiusSpatialQMIXLearner

LAYOUT_SEED = 1101


def make_env(seed: int, scenario_seed: int) -> V4ToggleRoleA1Env:
    return V4ToggleRoleA1Env(
        V4ToggleRoleA1Config(
            seed=seed,
            scenario_seed=scenario_seed,
            coarse_obs_size=5,
            enable_ue_mobility=False,
            enable_battery_cycle=True,
            backhaul_radius_cells=4.0,
            residual_initial_battery=30.0,
            randomize_residual_uav=False,
        )
    )


def disable_toggle(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[:, TOGGLE_ROLE] = 0
    return out


def evaluate(
    learner,
    *,
    residual_battery: float = 30.0,
    allow_toggle: bool = True,
    layout_seed: int = LAYOUT_SEED,
    permutation_seed_base: int = 70000,
) -> dict:
    """Evaluate one deterministic identity/permutation case per residual UAV."""
    rows = []
    for residual in range(5):
        env = make_env(layout_seed, scenario_seed=90000 + residual)
        out = env.reset_with_scenario(
            residual,
            permutation_seed_base + residual,
            layout_seed=layout_seed,
            residual_battery=residual_battery,
        )
        if not allow_toggle:
            out["action_mask"] = disable_toggle(out["action_mask"])

        rng = np.random.default_rng(10000 + residual)
        rewards = []
        relay_counts = []
        role_changes = 0
        first_other_relay = None
        first_handover = None
        roles0 = env.roles.copy()

        t = 0
        while True:
            actions = learner.act(out["obs"], out["action_mask"], 0.0, rng)
            nxt = env.step(actions)
            t += 1
            if not allow_toggle:
                nxt["action_mask"] = disable_toggle(nxt["action_mask"])

            rewards.append(float(nxt["reward"]))
            relay_counts.append(int(nxt["info"]["num_relay"]))
            role_changes += int(nxt["info"]["role_switches_this_slot"])

            active_other_relays = [
                i for i in range(env.cfg.num_uavs)
                if i != residual and env.status[i] == ACTIVE and env.roles[i] == RELAY
            ]
            if first_other_relay is None and active_other_relays:
                first_other_relay = t
            if first_handover is None and env.roles[residual] != RELAY and active_other_relays:
                first_handover = t

            out = nxt
            if out["truncated"]:
                break

        ret = int(env.return_start_time[residual]) if env.return_start_time[residual] >= 0 else None
        rejoin = int(env.rejoin_time[residual]) if env.rejoin_time[residual] >= 0 else None

        if ret is None:
            pre = float(np.mean(rewards))
            unavailable = None
            post = None
        else:
            # Stored return time is 1-based time of transition to Returning;
            # reward index ret-1 is the final service slot before unavailability.
            pre_slice = rewards[:ret]
            end_unavailable = (rejoin - 1) if rejoin is not None else len(rewards)
            unavailable_slice = rewards[ret:end_unavailable]
            post_slice = rewards[end_unavailable:] if rejoin is not None else []
            pre = float(np.mean(pre_slice)) if pre_slice else None
            unavailable = float(np.mean(unavailable_slice)) if unavailable_slice else None
            post = float(np.mean(post_slice)) if post_slice else None

        rows.append({
            "residual_uav_idx": residual,
            "residual_battery": float(residual_battery),
            "mean_utility": float(np.mean(rewards)),
            "final20": float(np.mean(rewards[-20:])),
            "max_instantaneous": float(np.max(rewards)),
            "pre_return_utility": pre,
            "unavailable_utility": unavailable,
            "post_rejoin_utility": post,
            "mean_active_relays": float(np.mean(relay_counts)),
            "role_changes": int(role_changes),
            "return_time": ret,
            "rejoin_time": rejoin,
            "first_other_relay_time": first_other_relay,
            "first_handover_time": first_handover,
            "other_relay_before_return": bool(
                first_other_relay is not None and (ret is None or first_other_relay <= ret)
            ),
            "initial_roles": roles0.astype(int).tolist(),
        })

    def mean_optional(key: str):
        xs = [r[key] for r in rows if r[key] is not None]
        return float(np.mean(xs)) if xs else None

    return {
        "residual_battery": float(residual_battery),
        "allow_toggle": bool(allow_toggle),
        "mean_utility": float(np.mean([r["mean_utility"] for r in rows])),
        "std_utility_across_residual_uav": float(np.std([r["mean_utility"] for r in rows])),
        "final20": float(np.mean([r["final20"] for r in rows])),
        "max_instantaneous": float(np.mean([r["max_instantaneous"] for r in rows])),
        "pre_return_utility": mean_optional("pre_return_utility"),
        "unavailable_utility": mean_optional("unavailable_utility"),
        "post_rejoin_utility": mean_optional("post_rejoin_utility"),
        "mean_active_relays": float(np.mean([r["mean_active_relays"] for r in rows])),
        "mean_role_changes": float(np.mean([r["role_changes"] for r in rows])),
        "other_relay_before_return_fraction": float(np.mean([r["other_relay_before_return"] for r in rows])),
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
    ap.add_argument("--total-transitions", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=81)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--buffer-capacity", type=int, default=30000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=512)
    ap.add_argument("--update-every", type=int, default=4)
    ap.add_argument("--target-every", type=int, default=200)
    ap.add_argument("--eps-decay-steps", type=int, default=30000)
    ap.add_argument("--outdir", type=str, default="results/v4_a1_toggle6_100k")
    ap.add_argument("--torch-threads", type=int, default=1, help="CPU torch threads; 1 is much faster for this small network/evaluation.")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--resume-from", type=str, default=None)
    args = ap.parse_args()

    if args.torch_threads < 1:
        raise ValueError("--torch-threads must be >= 1")
    torch.set_num_threads(args.torch_threads)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)

    outdir = resolve_output_dir(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    latest = outdir / f"seed{args.seed}_latest_full.pt"
    best_path = outdir / f"seed{args.seed}_best.pt"
    history_path = outdir / f"seed{args.seed}_history.json"

    env = make_env(LAYOUT_SEED, scenario_seed=args.seed + 1234)
    learner = V4HardRadiusSpatialQMIXLearner(
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

    steps = 0
    episodes = 0
    best = -1.0
    history: list[dict] = []
    resume_path = None
    if args.resume_from:
        p = Path(args.resume_from).expanduser()
        resume_path = p if p.is_absolute() else (Path.cwd() / p).resolve()
    elif args.resume and latest.exists():
        resume_path = latest
    if resume_path is not None:
        steps, episodes, best, history = load_full(resume_path, learner, rng)
        print(f"RESUMED from={resume_path} steps={steps} episodes={episodes} best={best:.4f}", flush=True)

    run_cfg = {
        **vars(args),
        "project_root": str(ROOT),
        "output_dir": str(outdir),
        "device": device,
        "torch_threads": int(args.torch_threads),
        "layout_seed": LAYOUT_SEED,
        "initialization": "scratch",
        "n_actions": 6,
        "action_design": "N/S/E/W/Stay preserve role; ToggleRole switches Access<->Relay without movement",
        "reward": "mean user satisfaction only; unchanged",
        "static_ue_layout": True,
        "battery_cycle": True,
        "dynamic_roles": True,
        "backhaul_radius_cells": 4.0,
        "initial_role_rule": "residual UAV at [11,6] is sole Relay; all other UAVs Access",
        "identity_position_randomization": "residual ID pinned to relay anchor; other IDs randomly permuted over remaining start cells",
        "residual_initial_battery": 30.0,
    }
    (outdir / f"seed{args.seed}_run_config.json").write_text(json.dumps(run_cfg, indent=2))

    print(f"PROJECT_ROOT={ROOT}", flush=True)
    print(f"OUTPUT_DIR={outdir}", flush=True)
    print(f"DEVICE={device}", flush=True)
    print("ACTION_DESIGN=6 actions (5 movement preserve role + ToggleRole)", flush=True)

    replay = ReplayBuffer(args.buffer_capacity, args.seed + steps)
    t0 = time.time()
    next_eval = ((steps // args.eval_every) + 1) * args.eval_every
    next_save = ((steps // args.save_every) + 1) * args.save_every

    while steps < args.total_transitions:
        episodes += 1
        out = env.reset(seed=LAYOUT_SEED)
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
                low = evaluate(learner, residual_battery=30.0, allow_toggle=True)
                fixed = evaluate(learner, residual_battery=30.0, allow_toggle=False)
                row = {
                    "steps": int(steps),
                    "episodes": int(episodes),
                    "epsilon": float(eps),
                    "mean_utility": low["mean_utility"],
                    "final20": low["final20"],
                    "mean_active_relays": low["mean_active_relays"],
                    "mean_role_changes": low["mean_role_changes"],
                    "other_relay_before_return_fraction": low["other_relay_before_return_fraction"],
                    "fixed_role_same_policy_utility": fixed["mean_utility"],
                    "dynamic_minus_fixed_pp": 100.0 * (low["mean_utility"] - fixed["mean_utility"]),
                    "runtime_s": float(time.time() - t0),
                }
                history.append(row)
                print("EVAL", json.dumps(row), flush=True)
                if low["mean_utility"] > best:
                    best = low["mean_utility"]
                    learner.save(str(best_path))
                history_path.write_text(json.dumps({"best": best, "history": history}, indent=2))
                next_eval += args.eval_every

            if steps >= next_save:
                save_full(latest, learner, steps, episodes, best, history, rng)
                next_save += args.save_every

            if done:
                break

    save_full(latest, learner, steps, episodes, best, history, rng)
    if best_path.exists():
        learner.load(str(best_path))

    low = evaluate(learner, residual_battery=30.0, allow_toggle=True)
    full = evaluate(learner, residual_battery=100.0, allow_toggle=True)
    fixed = evaluate(learner, residual_battery=30.0, allow_toggle=False)
    summary = {
        "seed": args.seed,
        "steps": steps,
        "episodes": episodes,
        "device": device,
        "best_validation": best,
        "action_design": run_cfg["action_design"],
        "reward": run_cfg["reward"],
        "best_checkpoint_eval_low_battery": low,
        "matched_full_battery_eval": full,
        "same_policy_toggle_disabled_eval": fixed,
        "dynamic_minus_fixed_pp": 100.0 * (low["mean_utility"] - fixed["mean_utility"]),
        "battery_specific_early_relay_gap": (
            low["other_relay_before_return_fraction"] - full["other_relay_before_return_fraction"]
        ),
        "history": history,
    }
    (outdir / f"seed{args.seed}_summary.json").write_text(json.dumps(summary, indent=2))
    print("DONE", json.dumps({
        "best_validation": best,
        "low_battery_mean": low["mean_utility"],
        "full_battery_mean": full["mean_utility"],
        "toggle_disabled_mean": fixed["mean_utility"],
        "dynamic_minus_fixed_pp": summary["dynamic_minus_fixed_pp"],
        "battery_specific_early_relay_gap": summary["battery_specific_early_relay_gap"],
    }), flush=True)


if __name__ == "__main__":
    main()
