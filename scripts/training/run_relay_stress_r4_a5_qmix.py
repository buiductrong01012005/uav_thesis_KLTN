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

EPS = 1e-9


def direct_access_cells(env) -> np.ndarray:
    """Grid cells from which an Access UAV can reach a surviving BS directly."""
    cells = []
    threshold_m = float(env.cfg.backhaul_radius_cells * env.cfg.cell_size_m)
    for r in range(env.cfg.grid_size):
        for c in range(env.cfg.grid_size):
            xy = env._cell_to_xy(np.asarray([r, c], dtype=np.float64))
            d = float(np.min(np.linalg.norm(env.surviving_bs_xy - xy, axis=1)))
            if d <= threshold_m + EPS:
                cells.append((r, c))
    return np.asarray(cells, dtype=np.float64)


def relay_required_ue_mask(env, ue_positions: np.ndarray, direct_cells: np.ndarray) -> np.ndarray:
    """UEs that cannot be covered by any Access position with direct BS backhaul."""
    ue = np.asarray(ue_positions, dtype=np.float64)
    direct = np.asarray(direct_cells, dtype=np.float64)
    if direct.ndim != 2 or direct.shape[1] != 2:
        raise ValueError("direct_cells must be Kx2")
    d2 = ((ue[:, None, :] - direct[None, :, :]) ** 2).sum(axis=2)
    direct_possible = np.min(d2, axis=1) <= float(env.cfg.access_radius_cells) ** 2 + EPS
    return ~direct_possible


def normalized_required_service(
    rates_mbps: np.ndarray, required_mask: np.ndarray, demand_mbps: float
) -> float:
    """Mean demand satisfaction of relay-required UEs, in [0,1]."""
    rates = np.asarray(rates_mbps, dtype=np.float64)
    required = np.asarray(required_mask, dtype=bool)
    if rates.shape != required.shape:
        raise ValueError("rates and required_mask must have the same shape")
    if demand_mbps <= 0:
        raise ValueError("demand_mbps must be positive")
    if not np.any(required):
        return 0.0
    sat = np.clip(rates[required] / float(demand_mbps), 0.0, 1.0)
    return float(np.mean(sat))


def shaped_training_reward(
    utility: float, required_service_score: float, relay_required_lambda: float
) -> float:
    if relay_required_lambda < 0:
        raise ValueError("relay_required_lambda must be nonnegative")
    if not 0.0 <= required_service_score <= 1.0 + 1e-9:
        raise ValueError("required_service_score must lie in [0,1]")
    return float(utility + relay_required_lambda * required_service_score)


class A5RelayRequiredMetricEnv(CartesianR4RelayStressBatteryEnv):
    """R4 env that captures A5 metric without changing the environment reward."""

    def __init__(self, cfg):
        self.last_required_service_score = 0.0
        self.last_relay_required_count = 0
        self.last_relay_required_served_count = 0
        super().__init__(cfg)
        self._a5_direct_cells = direct_access_cells(self)

    def compute_network(self):
        net = super().compute_network()
        rates = np.asarray(net["ue_rate_mbps"], dtype=np.float64)
        required = relay_required_ue_mask(self, self.ue_positions, self._a5_direct_cells)
        self.last_required_service_score = normalized_required_service(
            rates, required, self.cfg.ue_demand_mbps
        )
        self.last_relay_required_count = int(required.sum())
        self.last_relay_required_served_count = int(np.sum(required & (rates > EPS)))
        return net


def make_env(seed: int, horizon: int, residual_battery: float = 30.0):
    return A5RelayRequiredMetricEnv(
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
        required_service_scores = []
        required_counts = []
        required_served_counts = []

        while True:
            actions = learner.act(out["obs"], out["action_mask"], 0.0, rng)
            out = env.step(actions)
            rewards.append(float(out["reward"]))
            info = out.get("info", {})
            active_counts.append(int(info.get("active_count", env.cfg.num_uavs)))
            relay_counts.append(int(info.get("num_relay", 0)))
            required_service_scores.append(float(env.last_required_service_score))
            required_counts.append(int(env.last_relay_required_count))
            required_served_counts.append(int(env.last_relay_required_served_count))
            if out["truncated"]:
                break

        rows.append(
            {
                "seed": int(seed),
                "mean_utility": float(np.mean(rewards)),
                "final20": float(np.mean(rewards[-20:])),
                "mean_active_uavs": float(np.mean(active_counts)),
                "mean_relays": float(np.mean(relay_counts)),
                "mean_relay_required_service_score": float(np.mean(required_service_scores)),
                "mean_relay_required_ues": float(np.mean(required_counts)),
                "mean_relay_required_served_ues": float(np.mean(required_served_counts)),
            }
        )

    return {
        "mean_utility": float(np.mean([r["mean_utility"] for r in rows])),
        "std_utility": float(np.std([r["mean_utility"] for r in rows])),
        "mean_relays": float(np.mean([r["mean_relays"] for r in rows])),
        "mean_relay_required_service_score": float(
            np.mean([r["mean_relay_required_service_score"] for r in rows])
        ),
        "mean_relay_required_ues": float(np.mean([r["mean_relay_required_ues"] for r in rows])),
        "mean_relay_required_served_ues": float(
            np.mean([r["mean_relay_required_served_ues"] for r in rows])
        ),
        "rows": rows,
    }


def write_json(path: Path, payload: dict):
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(
        description=(
            "A5 Relay-required-service R4 Spatial-QMIX: train from random initialization on "
            "the 200 m backhaul task with U + lambda_G * G_req training reward."
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
        "--relay-required-lambda",
        type=float,
        default=0.10,
        help="A5 lambda_G multiplying normalized service of relay-required UEs",
    )
    ap.add_argument(
        "--outdir",
        default="results/relay_stress_r4_qmix/a5_relay_required_lam0p10",
    )
    args = ap.parse_args()

    if args.block_transitions <= 0:
        raise ValueError("block-transitions must be positive")
    if args.eval_every <= 0:
        raise ValueError("eval-every must be positive")
    if args.train_battery_low > args.train_battery_high:
        raise ValueError("train-battery-low must be <= train-battery-high")
    if not (0.0 <= args.relay_required_lambda <= 1.0):
        raise ValueError("relay-required-lambda must lie in [0,1]")

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
            "validation_mean_relay_required_service_score": initial_eval[
                "mean_relay_required_service_score"
            ],
            "validation_a5_objective": float(
                initial_eval["mean_utility"]
                + args.relay_required_lambda * initial_eval["mean_relay_required_service_score"]
            ),
            "train_td_loss_mean": None,
            "train_mean_relay_required_service_score": None,
            "train_mean_shaped_reward": None,
            "train_mean_sampled_residual_battery": None,
            "runtime_s": float(time.time() - t0),
        }
    )
    print("INITIAL", json.dumps(history[-1]), flush=True)

    run_config = {
        "method_label": "Spatial-QMIX",
        "ablation_role": "A5 relay-required-service reward on relay-stress R4 topology",
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
        "reward_changed": True,
        "reward_formula": "U_t + relay_required_lambda * G_req_t",
        "relay_required_lambda": float(args.relay_required_lambda),
        "G_req_definition": "mean min(R_u/d_u,1) over UEs that cannot be covered by any Access position having direct surviving-BS backhaul; 0 if none",
        "evaluation_uses_original_utility": True,
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
            "best ORIGINAL validation utility in the final 30k block only (same rule as R4 original-reward pilot)"
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
        interval_required_scores = []
        interval_shaped_rewards = []

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
                nxt = env.step(actions)
                required_score = float(env.last_required_service_score)
                train_reward = shaped_training_reward(
                    float(nxt["reward"]), required_score, args.relay_required_lambda
                )
                interval_required_scores.append(required_score)
                interval_shaped_rewards.append(train_reward)

                replay.add(
                    Transition(
                        out["obs"],
                        out["state"],
                        np.asarray(actions, dtype=np.int64),
                        train_reward,
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
                        "validation_mean_relay_required_service_score": ev[
                            "mean_relay_required_service_score"
                        ],
                        "validation_a5_objective": float(
                            ev["mean_utility"]
                            + args.relay_required_lambda * ev["mean_relay_required_service_score"]
                        ),
                        "train_mean_relay_required_service_score": (
                            float(np.mean(interval_required_scores))
                            if interval_required_scores
                            else None
                        ),
                        "train_mean_shaped_reward": (
                            float(np.mean(interval_shaped_rewards))
                            if interval_shaped_rewards
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
                            "training": f"relay-stress R4; A5 U + {args.relay_required_lambda:g}*G_req training reward; no curriculum",
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
                    interval_required_scores.clear()
                    interval_shaped_rewards.clear()

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
            "relay-stress R4 (200 m backhaul); no curriculum; A5 relay-required-service reward "
            f"U + {args.relay_required_lambda:g}*G_req; full Battery-ON + mobility task from scratch; "
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
                "method": "Spatial-QMIX R4 A5 relay-required-service",
                "seed": int(args.seed),
                "backhaul_radius_m": 200.0,
                "heldout_test_mean": final_test["mean_utility"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
