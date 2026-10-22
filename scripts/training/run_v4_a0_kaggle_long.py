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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.envs.v0_env import ACCESS, RELAY
from src.envs.v4_final_fixed_map_env import V4FinalFixedMapConfig, V4FinalFixedMapEnv
from src.rl.qmix import ReplayBuffer, Transition
from src.rl.v4_hardradius_spatial_qmix import V4HardRadiusSpatialQMIXLearner

ROLES = np.asarray([ACCESS, ACCESS, ACCESS, ACCESS, RELAY], dtype=np.int64)
LAYOUT_SEED = 1101


def make_env(seed: int = LAYOUT_SEED) -> V4FinalFixedMapEnv:
    env = V4FinalFixedMapEnv(
        V4FinalFixedMapConfig(
            seed=seed,
            coarse_obs_size=5,
            enable_ue_mobility=False,
            enable_battery_cycle=False,
        )
    )
    env.reset(seed=seed)
    env.roles[:] = ROLES
    return env


def fixed_role_mask(env: V4FinalFixedMapEnv) -> np.ndarray:
    base = env.get_action_mask()
    out = np.zeros_like(base)
    for i in range(env.cfg.num_uavs):
        for move in range(5):
            a = env.encode_action(move, int(ROLES[i]))
            out[i, a] = base[i, a]
    return out


def reset_env(env: V4FinalFixedMapEnv):
    out = env.reset(seed=LAYOUT_SEED)
    env.roles[:] = ROLES
    out["obs"] = env.get_obs()
    out["state"] = env.get_state()
    out["action_mask"] = fixed_role_mask(env)
    return out


def step_env(env: V4FinalFixedMapEnv, actions):
    out = env.step(actions)
    env.roles[:] = ROLES
    out["obs"] = env.get_obs()
    out["state"] = env.get_state()
    out["action_mask"] = fixed_role_mask(env)
    return out


def evaluate(learner, episodes: int = 3) -> dict:
    vals = []
    finals = []
    maxima = []
    for ep in range(episodes):
        env = make_env()
        out = reset_env(env)
        rng = np.random.default_rng(10000 + ep)
        us = []
        while True:
            actions = learner.act(out["obs"], out["action_mask"], 0.0, rng)
            out = step_env(env, actions)
            us.append(float(out["reward"]))
            if out["truncated"]:
                break
        vals.append(float(np.mean(us)))
        finals.append(float(np.mean(us[-20:])))
        maxima.append(float(np.max(us)))
    return {
        "mean_utility": float(np.mean(vals)),
        "std_utility": float(np.std(vals)),
        "final20": float(np.mean(finals)),
        "max_instantaneous": float(np.mean(maxima)),
    }


def save_full(
    path: Path,
    learner,
    steps: int,
    episodes: int,
    best: float,
    history: list[dict],
    rng: np.random.Generator,
):
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
    """Return a writable output root on Kaggle or a normal local machine.

    Kaggle datasets under /kaggle/input are read-only, so results must go to
    /kaggle/working. Locally, keep the familiar project-relative results path.
    UAV_OUTPUT_ROOT can override either behavior.
    """
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
    if p.is_absolute():
        return p.resolve()
    return (default_write_root() / p).resolve()


def resolve_resume_path(value: str | None, latest: Path) -> Path | None:
    if value:
        p = Path(value).expanduser()
        if not p.is_absolute():
            p = Path.cwd() / p
        return p.resolve()
    return latest if latest.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total-transitions", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=81)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--buffer-capacity", type=int, default=30000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=512)
    ap.add_argument("--update-every", type=int, default=4)
    ap.add_argument("--target-every", type=int, default=200)
    ap.add_argument("--eps-decay-steps", type=int, default=15000)
    ap.add_argument("--outdir", type=str, default="results/v4_a0_kaggle_long")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Optional explicit full-checkpoint path (useful when a previous Kaggle output is mounted as a dataset).",
    )
    args = ap.parse_args()

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

    env = make_env()
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
    resume_path = (
        resolve_resume_path(args.resume_from, latest) if (args.resume or args.resume_from) else None
    )
    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        steps, episodes, best, history = load_full(resume_path, learner, rng)
        print(
            f"RESUMED from={resume_path} steps={steps} episodes={episodes} best={best:.4f}",
            flush=True,
        )

    print(f"PROJECT_ROOT={ROOT}", flush=True)
    print(f"OUTPUT_DIR={outdir}", flush=True)
    print(f"DEVICE={device}", flush=True)
    (outdir / f"seed{args.seed}_run_config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "project_root": str(ROOT),
                "output_dir": str(outdir),
                "device": device,
                "layout_seed": LAYOUT_SEED,
            },
            indent=2,
        )
    )

    # Replay is intentionally rebuilt after process restart. This keeps the
    # checkpoint compact/reliable; epsilon resumes from global step and the
    # buffer is rewarmed before updates continue.
    replay = ReplayBuffer(args.buffer_capacity, args.seed + steps)
    t0 = time.time()
    next_eval = ((steps // args.eval_every) + 1) * args.eval_every
    next_save = ((steps // args.save_every) + 1) * args.save_every

    while steps < args.total_transitions:
        episodes += 1
        out = reset_env(env)
        while steps < args.total_transitions:
            decay_steps = max(1, int(args.eps_decay_steps))
            frac = min(steps / decay_steps, 1.0)
            eps = 1.0 + frac * (0.05 - 1.0)
            actions = learner.act(out["obs"], out["action_mask"], eps, rng)
            nxt = step_env(env, actions)
            done = float(nxt["truncated"])
            replay.add(
                Transition(
                    out["obs"],
                    out["state"],
                    np.asarray(actions, dtype=np.int64),
                    float(nxt["reward"]),
                    nxt["obs"],
                    nxt["state"],
                    nxt["action_mask"],
                    done,
                )
            )
            steps += 1

            if len(replay) >= max(args.warmup, args.batch_size) and steps % args.update_every == 0:
                learner.train_batch(replay.sample(args.batch_size))
            if steps % args.target_every == 0:
                learner.update_targets()

            out = nxt

            if steps >= next_eval:
                ev = evaluate(learner, episodes=3)
                row = {
                    "steps": int(steps),
                    "episodes": int(episodes),
                    "epsilon": float(eps),
                    **ev,
                    "runtime_s": float(time.time() - t0),
                }
                history.append(row)
                print("EVAL", json.dumps(row), flush=True)
                if ev["mean_utility"] > best:
                    best = ev["mean_utility"]
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
    final = evaluate(learner, episodes=5)
    summary = {
        "seed": args.seed,
        "steps": steps,
        "episodes": episodes,
        "device": device,
        "best_validation": best,
        "final_best_checkpoint_eval": final,
        "solution_reference": {
            "reachable_mean_episode_utility": 0.7109667128576073,
            "steady_state_utility": 0.7207358478740035,
            "target_reached_in_steps": 8,
        },
        "history": history,
    }
    (outdir / f"seed{args.seed}_summary.json").write_text(json.dumps(summary, indent=2))
    print("DONE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
