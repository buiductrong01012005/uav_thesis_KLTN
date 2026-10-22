from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.envs.thesis_cartesian_r6_curriculum_env import (
    CartesianR6BatteryConfig,
    CartesianR6BatteryEnv,
)
from src.rl.qmix import ReplayBuffer, Transition
from src.envs.v4_fixed_mobile_battery_env import ACTIVE
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
    return CartesianR6BatteryEnv(
        CartesianR6BatteryConfig(
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


def active_pairwise_spatial_metrics(
    uav_positions, active_mask, cell_size_m: float, min_separation_m: float
):
    """Return normalized closeness penalty and interpretable distance KPIs.

    Only UAVs that were ACTIVE in the serving slot are counted.  Exact
    same-cell collisions are already blocked by the environment; this term is
    therefore a *dispersion* shaping signal, not a replacement for collision
    avoidance.
    """
    pos = np.asarray(uav_positions, dtype=np.float64)
    active = np.asarray(active_mask, dtype=bool)
    if pos.ndim != 2 or pos.shape[1] != 2 or active.shape != (pos.shape[0],):
        raise ValueError("Invalid UAV position/active-mask shape")
    if cell_size_m <= 0 or min_separation_m <= 0:
        raise ValueError("cell_size_m and min_separation_m must be positive")
    ids = np.flatnonzero(active)
    if len(ids) < 2:
        return 0.0, 0.0, 0.0
    distances = []
    penalties = []
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            d_m = float(np.linalg.norm(pos[ids[a]] - pos[ids[b]]) * cell_size_m)
            distances.append(d_m)
            penalties.append(max(0.0, (min_separation_m - d_m) / min_separation_m))
    distances = np.asarray(distances, dtype=np.float64)
    penalties = np.asarray(penalties, dtype=np.float64)
    return (
        float(np.mean(penalties)),
        float(np.mean(distances)),
        float(np.mean(distances < min_separation_m)),
    )


def evaluate(learner, seeds, horizon: int, min_separation_m: float = 100.0):
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
        role_switch_counts = []
        energy_used = []
        spatial_penalties = []
        pairwise_distances = []
        close_pair_fractions = []

        while True:
            actions = learner.act(out["obs"], out["action_mask"], 0.0, rng)
            battery_before = np.asarray(env.battery, dtype=np.float64).copy()
            active_before = np.asarray(env.status == ACTIVE, dtype=bool).copy()
            out = env.step(actions)
            # Absolute drain is the same KPI for all reward variants.
            energy_used.append(float(np.maximum(0.0, battery_before - env.battery).sum()))
            spatial, mean_dist, close_frac = active_pairwise_spatial_metrics(
                env.uav_positions, active_before, env.cfg.cell_size_m, min_separation_m
            )
            spatial_penalties.append(spatial)
            pairwise_distances.append(mean_dist)
            close_pair_fractions.append(close_frac)
            rewards.append(float(out["reward"]))
            info = out.get("info", {})
            active_counts.append(int(info.get("active_count", env.cfg.num_uavs)))
            relay_counts.append(int(info.get("num_relay", 0)))
            role_switch_counts.append(int(info.get("role_switches_this_slot", 0)))
            if out["truncated"]:
                break

        rows.append(
            {
                "seed": int(seed),
                "mean_utility": float(np.mean(rewards)),
                "final20": float(np.mean(rewards[-20:])),
                "mean_active_uavs": float(np.mean(active_counts)),
                "mean_relays": float(np.mean(relay_counts)),
                "mean_role_switches_per_step": float(np.mean(role_switch_counts)),
                "mean_battery_drain_per_step": float(np.mean(energy_used)),
                "mean_spatial_closeness_penalty": float(np.mean(spatial_penalties)),
                "mean_pairwise_active_distance_m": float(np.mean(pairwise_distances)),
                "mean_close_pair_fraction": float(np.mean(close_pair_fractions)),
            }
        )

    return {
        "mean_utility": float(np.mean([r["mean_utility"] for r in rows])),
        "std_utility": float(np.std([r["mean_utility"] for r in rows])),
        "mean_relays": float(np.mean([r["mean_relays"] for r in rows])),
        "mean_role_switches_per_step": float(
            np.mean([r["mean_role_switches_per_step"] for r in rows])
        ),
        "mean_battery_drain_per_step": float(
            np.mean([r["mean_battery_drain_per_step"] for r in rows])
        ),
        "mean_spatial_closeness_penalty": float(
            np.mean([r["mean_spatial_closeness_penalty"] for r in rows])
        ),
        "mean_pairwise_active_distance_m": float(
            np.mean([r["mean_pairwise_active_distance_m"] for r in rows])
        ),
        "mean_close_pair_fraction": float(np.mean([r["mean_close_pair_fraction"] for r in rows])),
        "rows": rows,
    }


def shaped_training_reward(
    utility: float,
    battery_before,
    battery_after,
    role_switches: int,
    num_uavs: int,
    max_step_drain: float,
    energy_lambda: float = 0.0,
    role_lambda: float = 0.0,
):
    """Training-only shaping. The environment reward remains unmodified utility.

    Positive role bonus is intentionally naive: this ablation tests whether
    *incentivizing switching itself* helps or instead creates role oscillations.
    Neither autonomous Rejoin nor Return counts as a voluntary role switch.
    """
    if num_uavs <= 0 or max_step_drain <= 0:
        raise ValueError("Invalid energy normalization")
    if not (0 <= role_switches <= num_uavs):
        raise ValueError("Invalid role_switches count")
    before = np.asarray(battery_before, dtype=np.float64)
    after = np.asarray(battery_after, dtype=np.float64)
    if before.shape != (num_uavs,) or after.shape != (num_uavs,):
        raise ValueError("Expected one battery value per UAV")
    # Charging increases battery; never subtract that from consumption.
    energy = float(np.maximum(0.0, before - after).sum()) / (num_uavs * max_step_drain)
    switching = float(role_switches) / num_uavs
    return (
        float(utility - energy_lambda * energy + role_lambda * switching),
        energy,
        switching,
    )


def write_json(path: Path, payload: dict):
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(
        description=(
            "No-curriculum Spatial-QMIX baseline: train from random initialization "
            "on the complete Battery-ON + mobility task for all 90k transitions."
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
    ap.add_argument(
        "--energy-lambda",
        type=float,
        default=0.0,
        help="Nonnegative penalty for normalized battery drain (A1/A3)",
    )
    ap.add_argument(
        "--role-lambda",
        type=float,
        default=0.0,
        help="Nonnegative bonus for fraction of voluntary role switches (A2/A3/A4)",
    )
    ap.add_argument(
        "--spatial-lambda",
        type=float,
        default=0.0,
        help="Nonnegative penalty for ACTIVE UAV closeness below min-separation-m (A4)",
    )
    ap.add_argument(
        "--min-separation-m",
        type=float,
        default=100.0,
        help="Distance threshold for spatial closeness penalty; default 100 m = 2 grid cells",
    )
    ap.add_argument("--train-battery-low", type=float, default=24.0)
    ap.add_argument("--train-battery-high", type=float, default=36.0)
    ap.add_argument(
        "--outdir",
        default="results/thesis_curriculum_ablation/direct_full_task_randomized",
    )
    args = ap.parse_args()

    if args.block_transitions <= 0:
        raise ValueError("block-transitions must be positive")
    if args.eval_every <= 0:
        raise ValueError("eval-every must be positive")
    if args.train_battery_low > args.train_battery_high:
        raise ValueError("train-battery-low must be <= train-battery-high")
    if not (
        0 <= args.energy_lambda <= 1
        and 0 <= args.role_lambda <= 1
        and 0 <= args.spatial_lambda <= 1
    ):
        raise ValueError("Reward coefficients must lie in [0,1]")
    if args.min_separation_m <= 0:
        raise ValueError("min-separation-m must be positive")
    baseline_path = "results/thesis_curriculum_ablation/direct_full_task_randomized"
    if (
        args.energy_lambda or args.role_lambda or args.spatial_lambda
    ) and args.outdir == baseline_path:

        def tag(value):
            return format(value, ".5g").replace(".", "p")

        args.outdir = (
            f"results/reward_ablation/energy_{tag(args.energy_lambda)}_role_{tag(args.role_lambda)}"
            f"_spatial_{tag(args.spatial_lambda)}_dmin_{tag(args.min_separation_m)}m"
        )

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
    initial_eval = evaluate(learner, VALIDATION_SEEDS, args.horizon, args.min_separation_m)
    history.append(
        {
            "global_steps": 0,
            "block": 0,
            "block_name": "Random initialization",
            "block_steps": 0,
            "epsilon": None,
            "validation_mean_utility": initial_eval["mean_utility"],
            "validation_std_utility": initial_eval["std_utility"],
            "validation_mean_relays": initial_eval["mean_relays"],
            "validation_mean_role_switches_per_step": initial_eval["mean_role_switches_per_step"],
            "validation_mean_battery_drain_per_step": initial_eval["mean_battery_drain_per_step"],
            "validation_mean_spatial_closeness_penalty": initial_eval[
                "mean_spatial_closeness_penalty"
            ],
            "validation_mean_pairwise_active_distance_m": initial_eval[
                "mean_pairwise_active_distance_m"
            ],
            "validation_mean_close_pair_fraction": initial_eval["mean_close_pair_fraction"],
            "train_td_loss_mean": None,
            "train_mean_sampled_residual_battery": None,
            "runtime_s": float(time.time() - t0),
        }
    )
    print("INITIAL", json.dumps(history[-1]), flush=True)

    run_config = {
        "method_label": "Spatial-QMIX",
        "ablation_role": "no-curriculum direct full-task training",
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
        "reward_changed": bool(args.energy_lambda or args.role_lambda or args.spatial_lambda),
        "reward_formula": "utility - energy_lambda * normalized_energy + role_lambda * normalized_role_switches - spatial_lambda * normalized_closeness",
        "energy_lambda": float(args.energy_lambda),
        "role_lambda": float(args.role_lambda),
        "spatial_lambda": float(args.spatial_lambda),
        "min_separation_m": float(args.min_separation_m),
        "energy_normalizer": "num_uavs * (hover_drain + move_extra_drain + access_radio_drain)",
        "role_normalizer": "num_uavs; only voluntary ACTIVE role changes, as env.info['role_switches_this_slot']",
        "spatial_penalty": "mean over ACTIVE UAV pairs of max(0, (min_separation_m - pair_distance_m) / min_separation_m)",
        "spatial_scope": "ACTIVE UAV pairs only; Returning/Charging UAVs excluded",
        "validation_uses_original_utility": True,
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
        "checkpoint_selection": ("best validation utility in the final 30k block only"),
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
        interval_spatial_penalties = []

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
            sampled_battery = float(rng.uniform(args.train_battery_low, args.train_battery_high))
            env.cfg.residual_initial_battery = sampled_battery
            out = env.reset(seed=scenario_seed)
            interval_batteries.append(sampled_battery)

            while block_steps < args.block_transitions:
                frac = min(
                    block_steps / max(1, int(spec["eps_decay"])),
                    1.0,
                )
                eps = float(spec["eps_start"] + frac * (spec["eps_end"] - spec["eps_start"]))

                actions = learner.act(
                    out["obs"],
                    out["action_mask"],
                    eps,
                    rng,
                )
                battery_before = np.asarray(env.battery, dtype=np.float64).copy()
                active_before = np.asarray(env.status == ACTIVE, dtype=bool).copy()
                nxt = env.step(actions)
                max_step_drain = (
                    env.cfg.hover_drain
                    + env.cfg.move_extra_drain
                    + max(env.cfg.access_radio_drain, env.cfg.relay_radio_drain)
                )
                train_reward, energy_norm, switch_norm = shaped_training_reward(
                    float(nxt["reward"]),
                    battery_before,
                    env.battery,
                    int(nxt["info"]["role_switches_this_slot"]),
                    env.cfg.num_uavs,
                    max_step_drain,
                    args.energy_lambda,
                    args.role_lambda,
                )
                spatial_norm, _, _ = active_pairwise_spatial_metrics(
                    env.uav_positions,
                    active_before,
                    env.cfg.cell_size_m,
                    args.min_separation_m,
                )
                train_reward -= float(args.spatial_lambda) * spatial_norm
                interval_spatial_penalties.append(spatial_norm)

                replay.add(
                    Transition(
                        out["obs"],
                        out["state"],
                        np.asarray(actions, dtype=np.int64),
                        float(train_reward),
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
                    interval_losses.append(float(learner.train_batch(batch)))

                if block_steps % args.target_every == 0:
                    learner.update_targets()

                out = nxt

                if block_steps % args.eval_every == 0 or block_steps == args.block_transitions:
                    ev = evaluate(
                        learner,
                        VALIDATION_SEEDS,
                        args.horizon,
                        args.min_separation_m,
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
                        "validation_mean_role_switches_per_step": ev["mean_role_switches_per_step"],
                        "validation_mean_battery_drain_per_step": ev["mean_battery_drain_per_step"],
                        "validation_mean_spatial_closeness_penalty": ev[
                            "mean_spatial_closeness_penalty"
                        ],
                        "validation_mean_pairwise_active_distance_m": ev[
                            "mean_pairwise_active_distance_m"
                        ],
                        "validation_mean_close_pair_fraction": ev["mean_close_pair_fraction"],
                        "train_mean_spatial_closeness_penalty": (
                            float(np.mean(interval_spatial_penalties))
                            if interval_spatial_penalties
                            else None
                        ),
                        "train_td_loss_mean": (
                            float(np.mean(interval_losses)) if interval_losses else None
                        ),
                        "train_mean_sampled_residual_battery": (
                            float(np.mean(interval_batteries)) if interval_batteries else None
                        ),
                        "runtime_s": float(time.time() - t0),
                    }
                    history.append(row)
                    print("VAL", json.dumps(row), flush=True)

                    write_json(
                        outdir / f"seed{args.seed}_history.json",
                        {
                            "method": "Spatial-QMIX",
                            "training": "direct full task, no curriculum",
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
                    interval_spatial_penalties.clear()

                    # Same model-selection window as curriculum.
                    if block == 3 and ev["mean_utility"] > best_final_block:
                        best_final_block = float(ev["mean_utility"])
                        best_final_block_global_step = int(global_steps)
                        learner.save(str(outdir / f"seed{args.seed}_best_final_block.pt"))

                if nxt["truncated"]:
                    break

        learner.save(str(outdir / f"seed{args.seed}_block{block}_end.pt"))

    best_path = outdir / f"seed{args.seed}_best_final_block.pt"
    if not best_path.exists():
        learner.save(str(best_path))
        best_final_block_global_step = int(global_steps)
        best_final_block = evaluate(
            learner,
            VALIDATION_SEEDS,
            args.horizon,
            args.min_separation_m,
        )["mean_utility"]

    learner.load(str(best_path))
    final_val = evaluate(
        learner,
        VALIDATION_SEEDS,
        args.horizon,
        args.min_separation_m,
    )
    final_test = evaluate(
        learner,
        TEST_SEEDS,
        args.horizon,
        args.min_separation_m,
    )

    summary = {
        "method": "Spatial-QMIX",
        "training": (
            "no curriculum; full Battery-ON + mobility task from scratch; "
            "training residual battery Uniform[24,36]"
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
        json.dumps(
            {
                "method": "Spatial-QMIX",
                "seed": int(args.seed),
                "heldout_test_mean": final_test["mean_utility"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
