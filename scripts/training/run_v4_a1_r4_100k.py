from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.envs.v0_env import ACCESS, RELAY
from src.envs.v4_final_fixed_map_env import (
    ACTIVE,
    V4FinalFixedMapConfig,
    V4FinalFixedMapEnv,
)
from src.rl.qmix import ReplayBuffer, Transition
from src.rl.v4_hardradius_spatial_qmix import V4HardRadiusSpatialQMIXLearner

LAYOUT_SEED = 1101
DEFAULT_TOTAL_TRANSITIONS = 100_000


def initial_roles_for_residual(residual_uav_idx: int, num_uavs: int = 5) -> np.ndarray:
    """A1 relay-dropout audit: the low-battery UAV starts as the sole Relay."""
    idx = int(residual_uav_idx)
    if not 0 <= idx < int(num_uavs):
        raise ValueError(f"residual_uav_idx out of range: {idx}")
    roles = np.full(int(num_uavs), ACCESS, dtype=np.int64)
    roles[idx] = RELAY
    return roles


def make_env() -> V4FinalFixedMapEnv:
    """A1-R4: identical to relay-dropout A1 except backhaul radius is 4 cells."""
    return V4FinalFixedMapEnv(
        V4FinalFixedMapConfig(
            seed=LAYOUT_SEED,
            coarse_obs_size=5,
            backhaul_radius_cells=4.0,
            enable_ue_mobility=False,
            enable_battery_cycle=True,
            residual_initial_battery=30.0,
            randomize_residual_uav=False,  # runner chooses it with an independent RNG
        )
    )


def reset_env(
    env: V4FinalFixedMapEnv,
    residual_uav_idx: int,
) -> dict:
    """Reset to exactly the same static UE layout, varying only residual UAV identity."""
    out = env.reset(seed=LAYOUT_SEED)
    env.residual_uav_idx = int(residual_uav_idx)
    env.roles[:] = initial_roles_for_residual(env.residual_uav_idx, env.cfg.num_uavs)
    env.battery[:] = env.cfg.max_battery
    env.battery[env.residual_uav_idx] = env.cfg.residual_initial_battery

    # Refresh tensors after runner-controlled roles/battery are applied.
    out["obs"] = env.get_obs()
    out["state"] = env.get_state()
    out["action_mask"] = env.get_action_mask()
    out["episode_meta"] = env._episode_meta()
    return out


def discover_a0_checkpoint(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit).expanduser()
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"A0 checkpoint not found: {p}")
        return p

    # First try the common layouts used by the handoff bundle and by a
    # project folder copied directly onto Windows.
    candidates = [
        # User's normal local project layout.
        ROOT / "results" / "v4_a0_kaggle_long" / "seed81_best.pt",
        ROOT / "checkpoints" / "seed81_best.pt",
        ROOT / "results_current" / "seed81_best.pt",
        ROOT.parent / "results_current" / "seed81_best.pt",
        ROOT.parent / "checkpoints" / "seed81_best.pt",
        ROOT / "seed81_best.pt",
        Path.cwd() / "seed81_best.pt",
        Path.cwd() / "checkpoints" / "seed81_best.pt",
        Path.cwd() / "results_current" / "seed81_best.pt",
    ]
    seen = set()
    for p in candidates:
        p = p.resolve()
        if p in seen:
            continue
        seen.add(p)
        if p.is_file():
            return p

    # Last-resort bounded recursive search. This makes the runner robust to
    # extracting the ZIP one level deeper/shallower than expected without
    # requiring the user to move the checkpoint manually.
    search_roots = [ROOT, ROOT.parent, Path.cwd()]
    for base in search_roots:
        try:
            base = base.resolve()
            if not base.exists():
                continue
            for p in base.glob("**/seed81_best.pt"):
                if p.is_file():
                    return p.resolve()
        except (OSError, PermissionError):
            pass

    tried = "\n  - ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "Could not auto-find seed81_best.pt. Tried common locations and a recursive search.\n"
        "Common locations were:\n  - "
        + tried
        + "\nEither place seed81_best.pt under ./checkpoints or pass it explicitly with --init-from PATH."
    )


def build_learner(env: V4FinalFixedMapEnv, device: str) -> V4HardRadiusSpatialQMIXLearner:
    return V4HardRadiusSpatialQMIXLearner(
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


def save_full(
    path: Path,
    learner: V4HardRadiusSpatialQMIXLearner,
    steps: int,
    episodes: int,
    best: float,
    history: list[dict],
    action_rng: np.random.Generator,
    episode_rng: np.random.Generator,
) -> None:
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
        "action_generator_state": action_rng.bit_generator.state,
        "episode_generator_state": episode_rng.bit_generator.state,
    }
    if torch.cuda.is_available():
        payload["cuda_state"] = torch.cuda.get_rng_state_all()
    torch.save(payload, path)


def load_full(
    path: Path,
    learner: V4HardRadiusSpatialQMIXLearner,
    action_rng: np.random.Generator,
    episode_rng: np.random.Generator,
):
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
    if "action_generator_state" in payload:
        action_rng.bit_generator.state = payload["action_generator_state"]
    if "episode_generator_state" in payload:
        episode_rng.bit_generator.state = payload["episode_generator_state"]
    return (
        int(payload.get("steps", 0)),
        int(payload.get("episodes", 0)),
        float(payload.get("best", -1.0)),
        list(payload.get("history", [])),
    )


def _mean_or_none(xs: list[float]) -> float | None:
    return float(np.mean(xs)) if xs else None


def evaluate_one(
    learner: V4HardRadiusSpatialQMIXLearner,
    residual_uav_idx: int,
) -> dict:
    env = make_env()
    out = reset_env(env, residual_uav_idx)
    rng = np.random.default_rng(20_000 + residual_uav_idx)

    all_u: list[float] = []
    pre: list[float] = []
    unavailable: list[float] = []
    post: list[float] = []
    relay_counts: list[int] = []
    role_changes = 0
    return_time = None
    rejoin_time = None
    residual_role_at_departure = None
    proactive_backup_time = None
    proactive_handover_time = None
    other_relay_present_at_departure = None
    reactive_replacement_time = None
    had_rejoined = False

    while True:
        status_before = int(env.status[residual_uav_idx])
        actions = learner.act(out["obs"], out["action_mask"], 0.0, rng)
        nxt = env.step(actions)
        info = nxt["info"]
        t_now = int(info["t"])
        u = float(nxt["reward"])
        all_u.append(u)
        relay_counts.append(int(info["num_relay"]))
        role_changes += int(info["role_switches_this_slot"])

        # Reward of this slot used status_before.
        if had_rejoined:
            post.append(u)
        elif status_before == ACTIVE:
            pre.append(u)
        else:
            unavailable.append(u)

        roles_now = np.asarray(info["roles"], dtype=np.int64)
        status_now = np.asarray(info["status"], dtype=np.int64)
        other_active_relays = [
            i
            for i in range(env.cfg.num_uavs)
            if i != residual_uav_idx and int(status_now[i]) == ACTIVE and int(roles_now[i]) == RELAY
        ]

        # Before the low-battery UAV leaves, distinguish merely creating a backup
        # Relay from a true handover where the original Relay has switched to Access.
        if return_time is None and other_active_relays:
            if proactive_backup_time is None:
                proactive_backup_time = t_now
            if (
                int(status_now[residual_uav_idx]) == ACTIVE
                and int(roles_now[residual_uav_idx]) == ACCESS
            ):
                if proactive_handover_time is None:
                    proactive_handover_time = t_now

        new_return_time = int(info["return_start_time"][residual_uav_idx])
        if return_time is None and new_return_time >= 0:
            return_time = new_return_time
            residual_role_at_departure = int(roles_now[residual_uav_idx])
            other_relay_present_at_departure = bool(other_active_relays)

        # Reactive replacement is counted only if no backup Relay was already
        # active when the residual UAV departed. This avoids calling a pre-existing
        # Relay a one-step "replacement".
        if (
            return_time is not None
            and other_relay_present_at_departure is False
            and reactive_replacement_time is None
            and t_now > return_time
            and other_active_relays
        ):
            reactive_replacement_time = t_now

        new_rejoin_time = int(info["rejoin_time"][residual_uav_idx])
        if rejoin_time is None and new_rejoin_time >= 0:
            rejoin_time = new_rejoin_time
        if rejoin_time is not None and t_now > rejoin_time:
            had_rejoined = True

        out = nxt
        if nxt["truncated"]:
            break

    reactive_replacement_delay = None
    if return_time is not None and reactive_replacement_time is not None:
        reactive_replacement_delay = int(reactive_replacement_time - return_time)

    if other_relay_present_at_departure:
        relay_gap_slots = 0
    elif return_time is not None and reactive_replacement_time is not None:
        relay_gap_slots = reactive_replacement_delay
    elif return_time is not None:
        # No other UAV ever became Relay before the episode ended.
        relay_gap_slots = int(env.cfg.horizon - return_time)
    else:
        relay_gap_slots = None

    return {
        "residual_uav_idx": int(residual_uav_idx),
        "initial_relay_idx": int(residual_uav_idx),
        "mean_utility": float(np.mean(all_u)),
        "final20": float(np.mean(all_u[-20:])),
        "max_instantaneous": float(np.max(all_u)),
        "pre_drop_utility": _mean_or_none(pre),
        "unavailable_utility": _mean_or_none(unavailable),
        "post_rejoin_utility": _mean_or_none(post),
        "mean_active_relays": float(np.mean(relay_counts)),
        "role_changes": int(role_changes),
        "return_time": return_time,
        "rejoin_time": rejoin_time,
        "residual_role_at_departure": residual_role_at_departure,
        "residual_was_relay_at_departure": (
            bool(residual_role_at_departure == RELAY)
            if residual_role_at_departure is not None
            else None
        ),
        "proactive_backup_time": proactive_backup_time,
        "proactive_handover_time": proactive_handover_time,
        "other_relay_present_at_departure": other_relay_present_at_departure,
        "reactive_replacement_time": reactive_replacement_time,
        "reactive_replacement_delay": reactive_replacement_delay,
        "relay_gap_slots_after_departure": relay_gap_slots,
        "no_other_relay_before_episode_end": bool(
            return_time is not None
            and not other_relay_present_at_departure
            and reactive_replacement_time is None
        ),
    }


def evaluate(learner: V4HardRadiusSpatialQMIXLearner) -> dict:
    # Deterministically evaluate every possible residual-UAV identity so that
    # validation cannot be dominated by one easy/hard battery assignment.
    per_case = [evaluate_one(learner, i) for i in range(5)]

    def avg(key: str) -> float | None:
        vals = [row[key] for row in per_case if row[key] is not None]
        return float(np.mean(vals)) if vals else None

    return {
        "mean_utility": avg("mean_utility"),
        "std_utility_across_residual_uav": float(np.std([r["mean_utility"] for r in per_case])),
        "final20": avg("final20"),
        "max_instantaneous": avg("max_instantaneous"),
        "pre_drop_utility": avg("pre_drop_utility"),
        "unavailable_utility": avg("unavailable_utility"),
        "post_rejoin_utility": avg("post_rejoin_utility"),
        "mean_active_relays": avg("mean_active_relays"),
        "mean_role_changes": avg("role_changes"),
        "proactive_backup_fraction": float(
            np.mean([r["proactive_backup_time"] is not None for r in per_case])
        ),
        "proactive_handover_fraction": float(
            np.mean([r["proactive_handover_time"] is not None for r in per_case])
        ),
        "relay_present_at_departure_fraction": float(
            np.mean([bool(r["other_relay_present_at_departure"]) for r in per_case])
        ),
        "reactive_replacement_fraction": float(
            np.mean([r["reactive_replacement_time"] is not None for r in per_case])
        ),
        "mean_relay_gap_slots_after_departure": avg("relay_gap_slots_after_departure"),
        "no_other_relay_fraction": float(
            np.mean([r["no_other_relay_before_episode_end"] for r in per_case])
        ),
        "per_residual_uav": per_case,
    }


def resolve_outdir(value: str) -> Path:
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="A1-R4 0->100k: static UE + battery + dynamic Access/Relay + 4-cell backhaul"
    )
    ap.add_argument("--total-transitions", type=int, default=DEFAULT_TOTAL_TRANSITIONS)
    ap.add_argument("--seed", type=int, default=81)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--buffer-capacity", type=int, default=30000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=512)
    ap.add_argument("--update-every", type=int, default=4)
    ap.add_argument("--target-every", type=int, default=200)
    ap.add_argument("--eps-decay-steps", type=int, default=30000)
    ap.add_argument("--outdir", type=str, default="results/v4_a1_r4_100k")
    ap.add_argument(
        "--init-from",
        type=str,
        default=None,
        help="A0 best checkpoint; auto-detected if omitted",
    )
    ap.add_argument("--resume", action="store_true", help="Resume A1 from latest full checkpoint")
    ap.add_argument("--resume-from", type=str, default=None)
    args = ap.parse_args()

    if args.total_transitions <= 0:
        raise ValueError("--total-transitions must be positive")

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    action_rng = np.random.default_rng(args.seed)
    episode_rng = np.random.default_rng(args.seed + 1_000_003)

    outdir = resolve_outdir(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    latest = outdir / f"seed{args.seed}_latest_full.pt"
    best_path = outdir / f"seed{args.seed}_best.pt"
    history_path = outdir / f"seed{args.seed}_history.json"

    env = make_env()
    learner = build_learner(env, device)

    steps = 0
    episodes = 0
    best = -1.0
    history: list[dict] = []
    init_checkpoint = None

    resume_path = None
    if args.resume_from:
        p = Path(args.resume_from).expanduser()
        resume_path = (Path.cwd() / p).resolve() if not p.is_absolute() else p.resolve()
    elif args.resume:
        resume_path = latest

    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        steps, episodes, best, history = load_full(resume_path, learner, action_rng, episode_rng)
        print(
            f"RESUMED_A1_R4 from={resume_path} steps={steps} episodes={episodes} best={best:.6f}",
            flush=True,
        )
    else:
        init_checkpoint = discover_a0_checkpoint(args.init_from)
        learner.load(str(init_checkpoint))
        print(f"WARM_START_A0={init_checkpoint}", flush=True)

    if steps > args.total_transitions:
        raise ValueError(
            f"Checkpoint already has {steps} transitions, above requested target {args.total_transitions}."
        )

    print(f"PROJECT_ROOT={ROOT}", flush=True)
    print(f"DEVICE={device}", flush=True)
    print(f"OUTPUT_DIR={outdir}", flush=True)
    print(f"TARGET_TRANSITIONS={args.total_transitions}", flush=True)
    print(
        "TASK=A1_R4 static_UE=True battery=True dynamic_roles=True residual_starts_as_relay=True backhaul_radius_cells=4.0",
        flush=True,
    )

    run_config = {
        **vars(args),
        "project_root": str(ROOT),
        "output_dir": str(outdir),
        "device": device,
        "layout_seed": LAYOUT_SEED,
        "initial_role_rule": "residual UAV is sole Relay; all other UAVs are Access",
        "warm_start_checkpoint": str(init_checkpoint) if init_checkpoint else None,
        "static_ue_layout": True,
        "battery_cycle": True,
        "dynamic_roles": True,
        "residual_uav_sampling": "uniform independent RNG over 0..4 per episode",
        "relay_dropout_sampling": "the sampled residual UAV is always the sole initial Relay",
        "backhaul_radius_cells": float(env.cfg.backhaul_radius_cells),
        "a1_variant": "relay_dropout_identity_randomized_backhaul_r4",
    }
    (outdir / f"seed{args.seed}_run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    # Replay is rebuilt after restart, exactly like the A0 long runner.
    replay = ReplayBuffer(args.buffer_capacity, args.seed + steps)
    next_eval = ((steps // args.eval_every) + 1) * args.eval_every
    next_save = ((steps // args.save_every) + 1) * args.save_every
    t0 = time.time()

    while steps < args.total_transitions:
        episodes += 1
        residual_idx = int(episode_rng.integers(0, env.cfg.num_uavs))
        out = reset_env(env, residual_idx)

        while steps < args.total_transitions:
            decay_steps = max(1, int(args.eps_decay_steps))
            frac = min(steps / decay_steps, 1.0)
            epsilon = 1.0 + frac * (0.05 - 1.0)

            actions = learner.act(out["obs"], out["action_mask"], epsilon, action_rng)
            nxt = env.step(actions)
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
                ev = evaluate(learner)
                row = {
                    "steps": int(steps),
                    "episodes": int(episodes),
                    "epsilon": float(epsilon),
                    **ev,
                    "runtime_s": float(time.time() - t0),
                }
                history.append(row)
                print("EVAL", json.dumps(row), flush=True)
                if ev["mean_utility"] is not None and ev["mean_utility"] > best:
                    best = float(ev["mean_utility"])
                    learner.save(str(best_path))
                    print(f"NEW_BEST steps={steps} mean_utility={best:.6f}", flush=True)
                history_path.write_text(
                    json.dumps({"best": best, "history": history}, indent=2),
                    encoding="utf-8",
                )
                next_eval += args.eval_every

            if steps >= next_save:
                save_full(
                    latest,
                    learner,
                    steps,
                    episodes,
                    best,
                    history,
                    action_rng,
                    episode_rng,
                )
                print(f"CHECKPOINT steps={steps} path={latest}", flush=True)
                next_save += args.save_every

            if done:
                break

    # Exact target: the inner loop condition prevents overshoot.
    save_full(
        latest,
        learner,
        steps,
        episodes,
        best,
        history,
        action_rng,
        episode_rng,
    )

    if best_path.exists():
        learner.load(str(best_path))
    final = evaluate(learner)
    summary = {
        "seed": args.seed,
        "steps": int(steps),
        "episodes": int(episodes),
        "device": device,
        "best_validation": float(best),
        "final_best_checkpoint_eval": final,
        "warm_started_from_a0": str(init_checkpoint) if init_checkpoint else "resumed_A1",
        "history": history,
    }
    summary_path = outdir / f"seed{args.seed}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("DONE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
