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

from src.envs.thesis_cartesian_r4_relay_stress_env import (
    CartesianR4RelayStressBatteryConfig,
    CartesianR4RelayStressBatteryEnv,
)
from src.rl.qmix import ReplayBuffer, Transition
from src.rl.v5_relay_request_spatial_qmix import V5RelayRequestSpatialQMIXLearner


TRAIN_SEEDS = tuple(range(1101, 1121))
VALIDATION_SEEDS = (2101, 2102, 2103, 2104, 2105)
TEST_SEEDS = tuple(range(3101, 3121))

# These three 30k optimizer/exploration blocks intentionally match the
# Spatial-QMIX + Curriculum run. The environment, however, is the FULL final
# task from transition 0 to transition 90000.
BLOCK_SPECS = (
    {
        "id": 1,
        "name": "Full task block 1",
        "lr": 3e-4,
        "eps_start": 1.0,
        "eps_end": 0.05,
        "eps_decay": 25000,
    },
    {
        "id": 2,
        "name": "Full task block 2",
        "lr": 1e-4,
        "eps_start": 0.35,
        "eps_end": 0.05,
        "eps_decay": 20000,
    },
    {
        "id": 3,
        "name": "Full task block 3",
        "lr": 5e-5,
        "eps_start": 0.20,
        "eps_end": 0.05,
        "eps_decay": 15000,
    },
)


def make_env(seed: int, horizon: int, residual_battery: float = 30.0):
    return CartesianR4RelayStressBatteryEnv(
        CartesianR4RelayStressBatteryConfig(
            seed=int(seed),
            horizon=int(horizon),
            residual_initial_battery=float(residual_battery),
            randomize_residual_uav=True,
        )
    )


def make_learner(env, device: str, lr: float):
    return V5RelayRequestSpatialQMIXLearner(
        env.cfg.num_uavs,
        env.obs_dim,
        env.state_dim,
        env.n_actions,
        grid_size=env.cfg.grid_size,
        coarse_size=env.cfg.coarse_obs_size,
        lr=float(lr),
        gamma=0.99,
        hidden_dim=64,
        mixing_embed_dim=32,
        device=device,
    )


def reset_optimizer(learner, lr: float):
    params = list(learner.agent.parameters()) + list(learner.mixer.parameters())
    learner.optimizer = torch.optim.Adam(params, lr=float(lr))
    learner.update_targets()


def evaluate(learner, seeds, horizon: int):
    # Validation/test distribution remains fixed at battery=30, exactly like
    # the curriculum experiment.
    rows = []
    for seed in seeds:
        env = make_env(int(seed), int(horizon), residual_battery=30.0)
        out = env.reset(seed=int(seed))
        rng = np.random.default_rng(900000 + int(seed))
        rewards = []
        active_counts = []
        relay_counts = []

        while True:
            actions = learner.act(out["obs"], out["action_mask"], 0.0, rng)
            out = env.step(actions)
            rewards.append(float(out["reward"]))
            info = out.get("info", {})
            active_counts.append(int(info.get("active_count", env.cfg.num_uavs)))
            relay_counts.append(int(info.get("num_relay", 0)))
            if out["truncated"]:
                break

        rows.append({
            "seed": int(seed),
            "mean_utility": float(np.mean(rewards)),
            "final20": float(np.mean(rewards[-20:])),
            "mean_active_uavs": float(np.mean(active_counts)),
            "mean_relays": float(np.mean(relay_counts)),
        })

    return {
        "mean_utility": float(np.mean([r["mean_utility"] for r in rows])),
        "std_utility": float(np.std([r["mean_utility"] for r in rows])),
        "mean_relays": float(np.mean([r["mean_relays"] for r in rows])),
        "rows": rows,
    }


def write_json(path: Path, payload: dict):
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Relay-stress R4 Spatial-QMIX: train from random initialization on the "
            "200 m backhaul Battery-ON + mobility task for all 90k transitions."
        )
    )
    ap.add_argument("--seed", type=int, default=91)
    ap.add_argument("--block-transitions", type=int, default=30000)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--horizon", type=int, default=150)
    ap.add_argument("--buffer-capacity", type=int, default=30000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=3000)
    ap.add_argument("--update-every", type=int, default=4)
    ap.add_argument("--target-every", type=int, default=1000)
    ap.add_argument("--torch-threads", type=int, default=1)
    ap.add_argument("--train-battery-low", type=float, default=24.0)
    ap.add_argument("--train-battery-high", type=float, default=36.0)
    ap.add_argument(
        "--outdir",
        default="results/relay_stress_r4_qmix/original_reward",
    )
    args = ap.parse_args()

    if args.block_transitions <= 0:
        raise ValueError("block-transitions must be positive")
    if args.eval_every <= 0:
        raise ValueError("eval-every must be positive")
    if args.train_battery_low > args.train_battery_high:
        raise ValueError("train-battery-low must be <= train-battery-high")

    torch.set_num_threads(max(1, int(args.torch_threads)))
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

    # Random initialization; no prior checkpoint.
    initial_env = make_env(TRAIN_SEEDS[0], args.horizon, residual_battery=30.0)
    learner = make_learner(initial_env, device, BLOCK_SPECS[0]["lr"])

    history = []
    global_steps = 0
    best_final_block = -np.inf
    best_final_block_global_step = None
    t0 = time.time()

    # True x=0 point on the same fixed validation distribution.
    initial_eval = evaluate(learner, VALIDATION_SEEDS, args.horizon)
    history.append({
        "global_steps": 0,
        "block": 0,
        "block_name": "Random initialization",
        "block_steps": 0,
        "epsilon": None,
        "validation_mean_utility": initial_eval["mean_utility"],
        "validation_std_utility": initial_eval["std_utility"],
        "validation_mean_relays": initial_eval["mean_relays"],
        "train_td_loss_mean": None,
        "train_mean_sampled_residual_battery": None,
        "runtime_s": float(time.time() - t0),
    })
    print("INITIAL", json.dumps(history[-1]), flush=True)

    run_config = {
        "method_label": "Spatial-QMIX",
        "ablation_role": "relay-stress R4 topology pilot; original utility reward",
        "environment_profile": "thesis_cartesian_r4_relay_stress",
        "backhaul_radius_cells": 4.0,
        "backhaul_radius_m": 200.0,
        "reference_r6_backhaul_radius_m": 300.0,
        "seed": int(args.seed),
        "random_initialization": True,
        "total_transitions": int(3 * args.block_transitions),
        "block_transitions": int(args.block_transitions),
        "eval_every": int(args.eval_every),
        "full_environment_from_transition_zero": True,
        "ue_mobility": True,
        "battery": True,
        "training_residual_battery_range": [
            float(args.train_battery_low),
            float(args.train_battery_high),
        ],
        "validation_test_residual_battery": 30.0,
        "transition_focused_replay": False,
        "replay_type": "uniform only",
        "action_space": "Cartesian 5 movement x {Access, Relay}",
        "reward_changed": False,
        "reward_formula": "original environment utility U_t only",
        "train_seeds": list(TRAIN_SEEDS),
        "validation_seeds": list(VALIDATION_SEEDS),
        "heldout_test_seeds": list(TEST_SEEDS),
        "block_specs": BLOCK_SPECS,
        "optimizer_replay_reset_at_30k_60k": True,
        "why_three_blocks": (
            "match the LR/epsilon/replay-reset schedule used by "
            "Spatial-QMIX + Curriculum while keeping the environment "
            "fully Battery-ON for all 90k transitions"
        ),
        "checkpoint_selection": (
            "best validation utility in the final 30k block only"
        ),
    }
    write_json(outdir / f"seed{args.seed}_run_config.json", run_config)

    for spec in BLOCK_SPECS:
        block = int(spec["id"])

        # The environment is never simplified: Battery ON for all blocks.
        env = make_env(TRAIN_SEEDS[0], args.horizon, residual_battery=30.0)

        reset_optimizer(learner, spec["lr"])
        replay = ReplayBuffer(
            int(args.buffer_capacity),
            int(args.seed + block * 1000),
        )

        block_steps = 0
        interval_losses = []
        interval_batteries = []

        print(
            f"BLOCK {block}: FULL TASK | Battery ON | mobility ON | "
            f"residual battery U[{args.train_battery_low},{args.train_battery_high}]",
            flush=True,
        )

        while block_steps < args.block_transitions:
            scenario_seed = int(rng.choice(TRAIN_SEEDS))

            # Critical ablation detail:
            # randomize residual battery BEFORE reset so the returned observation
            # is consistent with the sampled battery.
            sampled_battery = float(
                rng.uniform(args.train_battery_low, args.train_battery_high)
            )
            env.cfg.residual_initial_battery = sampled_battery
            out = env.reset(seed=scenario_seed)
            interval_batteries.append(sampled_battery)

            while block_steps < args.block_transitions:
                frac = min(
                    block_steps / max(1, int(spec["eps_decay"])),
                    1.0,
                )
                eps = float(
                    spec["eps_start"]
                    + frac * (spec["eps_end"] - spec["eps_start"])
                )

                actions = learner.act(
                    out["obs"],
                    out["action_mask"],
                    eps,
                    rng,
                )
                nxt = env.step(actions)

                replay.add(
                    Transition(
                        out["obs"],
                        out["state"],
                        np.asarray(actions, dtype=np.int64),
                        float(nxt["reward"]),
                        nxt["obs"],
                        nxt["state"],
                        nxt["action_mask"],
                        float(nxt["truncated"]),
                    )
                )

                block_steps += 1
                global_steps += 1

                if (
                    len(replay) >= max(args.warmup, args.batch_size)
                    and block_steps % args.update_every == 0
                ):
                    batch = replay.sample(args.batch_size)
                    interval_losses.append(
                        float(learner.train_batch(batch))
                    )

                if block_steps % args.target_every == 0:
                    learner.update_targets()

                out = nxt

                if (
                    block_steps % args.eval_every == 0
                    or block_steps == args.block_transitions
                ):
                    ev = evaluate(
                        learner,
                        VALIDATION_SEEDS,
                        args.horizon,
                    )
                    row = {
                        "global_steps": int(global_steps),
                        "block": int(block),
                        "block_name": str(spec["name"]),
                        "block_steps": int(block_steps),
                        "epsilon": float(eps),
                        "validation_mean_utility": ev["mean_utility"],
                        "validation_std_utility": ev["std_utility"],
                        "validation_mean_relays": ev["mean_relays"],
                        "train_td_loss_mean": (
                            float(np.mean(interval_losses))
                            if interval_losses
                            else None
                        ),
                        "train_mean_sampled_residual_battery": (
                            float(np.mean(interval_batteries))
                            if interval_batteries
                            else None
                        ),
                        "runtime_s": float(time.time() - t0),
                    }
                    history.append(row)
                    print("VAL", json.dumps(row), flush=True)

                    write_json(
                        outdir / f"seed{args.seed}_history.json",
                        {
                            "method": "Spatial-QMIX",
                            "training": "relay-stress R4; direct full task; original utility reward; no curriculum",
                            "seed": int(args.seed),
                            "boundaries": [
                                int(args.block_transitions),
                                int(2 * args.block_transitions),
                            ],
                            "history": history,
                        },
                    )

                    interval_losses.clear()
                    interval_batteries.clear()

                    # Same model-selection window as curriculum.
                    if block == 3 and ev["mean_utility"] > best_final_block:
                        best_final_block = float(ev["mean_utility"])
                        best_final_block_global_step = int(global_steps)
                        learner.save(
                            str(outdir / f"seed{args.seed}_best_final_block.pt")
                        )

                if nxt["truncated"]:
                    break

        learner.save(
            str(outdir / f"seed{args.seed}_block{block}_end.pt")
        )

    best_path = outdir / f"seed{args.seed}_best_final_block.pt"
    if not best_path.exists():
        learner.save(str(best_path))
        best_final_block_global_step = int(global_steps)
        best_final_block = evaluate(
            learner,
            VALIDATION_SEEDS,
            args.horizon,
        )["mean_utility"]

    learner.load(str(best_path))
    final_val = evaluate(
        learner,
        VALIDATION_SEEDS,
        args.horizon,
    )
    final_test = evaluate(
        learner,
        TEST_SEEDS,
        args.horizon,
    )

    summary = {
        "method": "Spatial-QMIX",
        "training": (
            "relay-stress R4 (200 m backhaul); no curriculum; original utility reward; "
            "full Battery-ON + mobility task from scratch; training residual battery Uniform[24,36]"
        ),
        "seed": int(args.seed),
        "total_transitions": int(global_steps),
        "best_final_block_validation": float(best_final_block),
        "best_final_block_global_step": int(best_final_block_global_step),
        "final_validation": final_val,
        "heldout_test": final_test,
    }
    write_json(outdir / f"seed{args.seed}_summary.json", summary)

    print(
        "DONE",
        json.dumps({
            "method": "Spatial-QMIX R4 relay-stress",
            "seed": int(args.seed),
            "backhaul_radius_m": 200.0,
            "heldout_test_mean": final_test["mean_utility"],
        }),
        flush=True,
    )


if __name__ == "__main__":
    main()
