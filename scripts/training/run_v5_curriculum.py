from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.envs.v5_curriculum import (
    DEFAULT_CURRICULUM,
    V5CurriculumStage,
    curriculum_stage_dict,
    make_curriculum_env,
)
from src.rl.qmix import ReplayBuffer, Transition
from src.rl.v5_relay_request_spatial_qmix import V5RelayRequestSpatialQMIXLearner


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


def make_learner(env, device: str) -> V5RelayRequestSpatialQMIXLearner:
    return V5RelayRequestSpatialQMIXLearner(
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


def evaluate(learner, stage: V5CurriculumStage, seeds: list[int]) -> dict:
    rows = []
    for seed in seeds:
        env = make_curriculum_env(stage, seed, training=False)
        out = env.reset(seed=seed)
        rng = np.random.default_rng(900000 + stage.index * 10000 + seed)

        rewards = []
        relay_counts = []
        access_counts = []
        open_requests = []
        role_changes = 0
        accepts = 0
        releases = 0
        claim_conflicts = 0
        rejected_claims = 0
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
            claim_conflicts += int(info.get("relay_claim_conflicts_this_slot", 0))
            rejected_claims += len(info.get("rejected_relay_claims_this_slot", []))

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
        rows.append(
            {
                "seed": int(seed),
                "mean_utility": float(np.mean(rewards)),
                "final100": float(np.mean(rewards[-min(100, T) :])),
                "mean_active_access": float(np.mean(access_counts)),
                "mean_active_relays": float(np.mean(relay_counts)),
                "mean_open_requests": float(np.mean(open_requests)),
                "role_changes": int(role_changes),
                "accept_count": int(accepts),
                "release_count": int(releases),
                "claim_conflict_count": int(claim_conflicts),
                "rejected_claim_count": int(rejected_claims),
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
            }
        )

    def avg(key: str) -> float:
        return float(np.mean([r[key] for r in rows]))

    regime_summary = {}
    for regime in ("direct_friendly", "relay_stressed", "transition"):
        us = [
            r["regime_mean_utility"][regime]
            for r in rows
            if r["regime_mean_utility"][regime] is not None
        ]
        rs = [
            r["regime_mean_relays"][regime]
            for r in rows
            if r["regime_mean_relays"][regime] is not None
        ]
        regime_summary[regime] = {
            "mean_utility": float(np.mean(us)) if us else None,
            "mean_relays": float(np.mean(rs)) if rs else None,
        }

    direct_relays = regime_summary["direct_friendly"]["mean_relays"]
    stressed_relays = regime_summary["relay_stressed"]["mean_relays"]
    relay_regime_gap = (
        float(stressed_relays - direct_relays)
        if stressed_relays is not None and direct_relays is not None
        else None
    )

    return {
        "stage": curriculum_stage_dict(stage),
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
        "mean_claim_conflict_count": avg("claim_conflict_count"),
        "mean_rejected_claim_count": avg("rejected_claim_count"),
        "mean_return_events": avg("return_events"),
        "mean_rejoin_events": avg("rejoin_events"),
        "mean_movement_steps": avg("movement_steps"),
        "regime_summary": regime_summary,
        "relay_stressed_minus_direct_mean_relays": relay_regime_gap,
        "rows": rows,
    }


def save_full(
    path: Path,
    learner,
    stage: V5CurriculumStage,
    local_steps: int,
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
        "stage": curriculum_stage_dict(stage),
        "local_steps": int(local_steps),
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
        int(payload.get("local_steps", 0)),
        int(payload.get("episodes", 0)),
        float(payload.get("best", -1.0)),
        list(payload.get("history", [])),
    )


def epsilon_for(stage: V5CurriculumStage, local_steps: int) -> float:
    frac = min(local_steps / max(1, stage.eps_decay_steps), 1.0)
    return float(stage.eps_start + frac * (stage.eps_end - stage.eps_start))


def stage_dir_name(stage: V5CurriculumStage) -> str:
    return f"stage{stage.index}_{stage.name}"


def with_budget_overrides(args) -> tuple[V5CurriculumStage, ...]:
    budgets = {
        1: args.stage1_transitions,
        2: args.stage2_transitions,
        3: args.stage3_transitions,
    }
    stages = []
    for stage in DEFAULT_CURRICULUM:
        budget = budgets[stage.index]
        stages.append(replace(stage, train_transitions=int(budget)))
    return tuple(stages)


def write_progress(path: Path, progress: dict) -> None:
    path.write_text(json.dumps(progress, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=81)
    ap.add_argument("--stage1-transitions", type=int, default=30000)
    ap.add_argument("--stage2-transitions", type=int, default=30000)
    ap.add_argument("--stage3-transitions", type=int, default=60000)
    ap.add_argument("--stop-after-stage", type=int, choices=[1, 2, 3], default=3)
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--buffer-capacity", type=int, default=50000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=1024)
    ap.add_argument("--update-every", type=int, default=4)
    ap.add_argument("--target-every", type=int, default=200)
    ap.add_argument("--train-seed-start", type=int, default=1101)
    ap.add_argument("--train-seed-count", type=int, default=20)
    ap.add_argument("--val-seeds", nargs="+", type=int, default=[2101, 2102, 2103, 2104, 2105])
    ap.add_argument("--outdir", type=str, default="results/v5_curriculum")
    ap.add_argument("--torch-threads", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if min(args.stage1_transitions, args.stage2_transitions, args.stage3_transitions) < 0:
        raise ValueError("stage transition budgets must be non-negative")

    torch.set_num_threads(max(1, args.torch_threads))
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)

    stages = with_budget_overrides(args)
    train_seeds = list(range(args.train_seed_start, args.train_seed_start + args.train_seed_count))
    outdir = resolve_output_dir(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    progress_path = outdir / f"seed{args.seed}_curriculum_progress.json"
    run_cfg_path = outdir / f"seed{args.seed}_curriculum_config.json"

    run_cfg = {
        **vars(args),
        "project_root": str(ROOT),
        "output_dir": str(outdir),
        "device": device,
        "train_seeds": train_seeds,
        "stages": [curriculum_stage_dict(s) for s in stages],
        "fixed_bs_topology": True,
        "action_design": "0-4 movement preserve role; 5-8 AcceptRelay(other target); 9 ReleaseRelay",
        "relay_needed": "automatic direct-backhaul/access-offer hysteresis 0.75 on, 0.90 off",
        "reward": "mean user satisfaction only; unchanged across all curriculum stages",
        "curriculum_rule": (
            "stage best checkpoint is selected by mean validation utility only; "
            "next stage loads that checkpoint into a fresh optimizer and fresh replay buffer"
        ),
    }
    run_cfg_path.write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")

    progress = {
        "seed": int(args.seed),
        "completed_stage": 0,
        "current_stage": None,
        "stage_summaries": [],
    }
    if args.resume and progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))

    print(f"PROJECT_ROOT={ROOT}", flush=True)
    print(f"OUTPUT_DIR={outdir}", flush=True)
    print(f"DEVICE={device}", flush=True)
    print(f"TRAIN_SEEDS={train_seeds[0]}..{train_seeds[-1]}", flush=True)
    print(f"VAL_SEEDS={args.val_seeds}", flush=True)

    previous_best_path: Path | None = None
    completed_stage = int(progress.get("completed_stage", 0))
    if completed_stage > 0:
        prev = stages[completed_stage - 1]
        previous_best_path = outdir / stage_dir_name(prev) / f"seed{args.seed}_best.pt"

    for stage in stages:
        if stage.index > args.stop_after_stage:
            break
        if stage.index <= completed_stage:
            previous_best_path = outdir / stage_dir_name(stage) / f"seed{args.seed}_best.pt"
            continue
        if stage.train_transitions == 0:
            progress["completed_stage"] = stage.index
            progress["current_stage"] = None
            write_progress(progress_path, progress)
            completed_stage = stage.index
            continue

        stage_dir = outdir / stage_dir_name(stage)
        stage_dir.mkdir(parents=True, exist_ok=True)
        best_path = stage_dir / f"seed{args.seed}_best.pt"
        latest_path = stage_dir / f"seed{args.seed}_latest_full.pt"
        history_path = stage_dir / f"seed{args.seed}_history.json"
        summary_path = stage_dir / f"seed{args.seed}_summary.json"

        env = make_curriculum_env(stage, train_seeds[0], training=True)
        learner = make_learner(env, device)
        replay = ReplayBuffer(args.buffer_capacity, args.seed + stage.index * 1000)

        local_steps = 0
        episodes = 0
        best = -1.0
        history: list[dict] = []
        resuming_this_stage = (
            args.resume
            and int(progress.get("current_stage") or -1) == stage.index
            and latest_path.exists()
        )

        if resuming_this_stage:
            local_steps, episodes, best, history = load_full(latest_path, learner, rng)
            print(
                f"RESUMED stage={stage.index} from={latest_path} "
                f"local_steps={local_steps} episodes={episodes} best={best:.4f}",
                flush=True,
            )
        else:
            if stage.index > 1:
                if previous_best_path is None or not previous_best_path.exists():
                    raise FileNotFoundError(
                        f"Previous-stage best checkpoint not found: {previous_best_path}"
                    )
                learner.load(str(previous_best_path))
                print(
                    f"INIT stage={stage.index} from previous best={previous_best_path}",
                    flush=True,
                )

            # Evaluate the handoff *before* new-stage updates.  This guarantees
            # that catastrophic fine-tuning cannot erase a stronger incoming
            # checkpoint: if training degrades, step-0 remains the stage best.
            ev0 = evaluate(learner, stage, list(args.val_seeds))
            best = float(ev0["mean_utility"])
            learner.save(str(best_path))
            history = [
                {
                    "stage_steps": 0,
                    "episodes": 0,
                    "epsilon": float(stage.eps_start),
                    "mean_utility": ev0["mean_utility"],
                    "std_utility_across_seeds": ev0["std_utility_across_seeds"],
                    "final100": ev0["final100"],
                    "mean_active_relays": ev0["mean_active_relays"],
                    "mean_role_changes": ev0["mean_role_changes"],
                    "mean_claim_conflicts": ev0["mean_claim_conflict_count"],
                    "mean_rejected_claims": ev0["mean_rejected_claim_count"],
                    "mean_return_events": ev0["mean_return_events"],
                    "mean_rejoin_events": ev0["mean_rejoin_events"],
                    "direct_mean_relays": ev0["regime_summary"]["direct_friendly"]["mean_relays"],
                    "relay_stressed_mean_relays": ev0["regime_summary"]["relay_stressed"][
                        "mean_relays"
                    ],
                    "relay_stressed_minus_direct_mean_relays": ev0[
                        "relay_stressed_minus_direct_mean_relays"
                    ],
                    "runtime_s": 0.0,
                }
            ]
            history_path.write_text(
                json.dumps({"best": best, "history": history}, indent=2),
                encoding="utf-8",
            )

        progress["current_stage"] = stage.index
        write_progress(progress_path, progress)

        next_eval = ((local_steps // stage.eval_every) + 1) * stage.eval_every
        next_save = ((local_steps // args.save_every) + 1) * args.save_every
        t0 = time.time()

        print(
            "STAGE_START "
            + json.dumps(
                {
                    **curriculum_stage_dict(stage),
                    "local_steps": local_steps,
                    "best": best,
                }
            ),
            flush=True,
        )

        while local_steps < stage.train_transitions:
            episodes += 1
            episode_seed = int(rng.choice(train_seeds))
            out = env.reset(seed=episode_seed)

            while local_steps < stage.train_transitions:
                eps = epsilon_for(stage, local_steps)
                actions = learner.act(out["obs"], out["action_mask"], eps, rng)
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
                local_steps += 1

                if (
                    len(replay) >= max(args.warmup, args.batch_size)
                    and local_steps % args.update_every == 0
                ):
                    learner.train_batch(replay.sample(args.batch_size))
                if local_steps % args.target_every == 0:
                    learner.update_targets()
                out = nxt

                if local_steps >= next_eval:
                    ev = evaluate(learner, stage, list(args.val_seeds))
                    row = {
                        "stage_steps": int(local_steps),
                        "episodes": int(episodes),
                        "epsilon": float(eps),
                        "mean_utility": ev["mean_utility"],
                        "std_utility_across_seeds": ev["std_utility_across_seeds"],
                        "final100": ev["final100"],
                        "mean_active_relays": ev["mean_active_relays"],
                        "mean_role_changes": ev["mean_role_changes"],
                        "mean_claim_conflicts": ev["mean_claim_conflict_count"],
                        "mean_rejected_claims": ev["mean_rejected_claim_count"],
                        "mean_return_events": ev["mean_return_events"],
                        "mean_rejoin_events": ev["mean_rejoin_events"],
                        "direct_mean_relays": ev["regime_summary"]["direct_friendly"][
                            "mean_relays"
                        ],
                        "relay_stressed_mean_relays": ev["regime_summary"]["relay_stressed"][
                            "mean_relays"
                        ],
                        "relay_stressed_minus_direct_mean_relays": ev[
                            "relay_stressed_minus_direct_mean_relays"
                        ],
                        "runtime_s": float(time.time() - t0),
                    }
                    history.append(row)
                    print(f"STAGE{stage.index}_EVAL", json.dumps(row), flush=True)
                    if ev["mean_utility"] > best:
                        best = float(ev["mean_utility"])
                        learner.save(str(best_path))
                    history_path.write_text(
                        json.dumps({"best": best, "history": history}, indent=2),
                        encoding="utf-8",
                    )
                    next_eval += stage.eval_every

                if local_steps >= next_save:
                    save_full(
                        latest_path,
                        learner,
                        stage,
                        local_steps,
                        episodes,
                        best,
                        history,
                        rng,
                    )
                    next_save += args.save_every

                if done:
                    break

        save_full(latest_path, learner, stage, local_steps, episodes, best, history, rng)

        # Stage handoff always uses the strongest validation-utility checkpoint,
        # never the latest weights.
        learner.load(str(best_path))
        final_eval = evaluate(learner, stage, list(args.val_seeds))
        stage_summary = {
            "stage": curriculum_stage_dict(stage),
            "seed": int(args.seed),
            "stage_steps": int(local_steps),
            "episodes": int(episodes),
            "best_validation": float(best),
            "validation": final_eval,
            "history": history,
            "handoff_checkpoint": str(best_path),
        }
        summary_path.write_text(json.dumps(stage_summary, indent=2), encoding="utf-8")

        progress["completed_stage"] = stage.index
        progress["current_stage"] = None
        old_summaries = [
            x
            for x in progress.get("stage_summaries", [])
            if int(x.get("stage_index", -1)) != stage.index
        ]
        old_summaries.append(
            {
                "stage_index": stage.index,
                "stage_name": stage.name,
                "best_validation": float(best),
                "relay_stressed_minus_direct_mean_relays": final_eval[
                    "relay_stressed_minus_direct_mean_relays"
                ],
                "checkpoint": str(best_path),
            }
        )
        progress["stage_summaries"] = sorted(old_summaries, key=lambda x: x["stage_index"])
        write_progress(progress_path, progress)

        previous_best_path = best_path
        completed_stage = stage.index
        print(
            f"STAGE{stage.index}_DONE "
            + json.dumps(
                {
                    "best_validation": best,
                    "relay_stressed_minus_direct_mean_relays": final_eval[
                        "relay_stressed_minus_direct_mean_relays"
                    ],
                    "mean_return_events": final_eval["mean_return_events"],
                    "mean_rejoin_events": final_eval["mean_rejoin_events"],
                }
            ),
            flush=True,
        )

    final_stage_idx = min(int(args.stop_after_stage), int(progress.get("completed_stage", 0)))
    print(
        "CURRICULUM_DONE "
        + json.dumps(
            {
                "completed_stage": final_stage_idx,
                "progress": str(progress_path),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
