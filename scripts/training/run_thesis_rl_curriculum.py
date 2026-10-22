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

from src.envs.v4_fixed_mobile_battery_env import ACTIVE, CHARGING, RETURNING
from src.envs.thesis_cartesian_r6_curriculum_env import (
    CartesianR6Config,
    CartesianR6Env,
    CartesianR6BatteryConfig,
    CartesianR6BatteryEnv,
    CartesianR6BatteryFocusConfig,
    CartesianR6BatteryFocusEnv,
)
from src.envs.v7_r6_macro_relay_stable_env import (
    V7R6MacroRelayStableConfig,
    V7R6MacroRelayStableEnv,
)
from src.envs.v8_r6_macro_relay_battery_env import (
    V8R6MacroRelayBatteryConfig,
    V8R6MacroRelayBatteryEnv,
)
from src.envs.v9_r6_macro_relay_battery_focus_env import (
    V9R6MacroRelayBatteryFocusConfig,
    V9R6MacroRelayBatteryFocusEnv,
)
from src.rl.qmix import ReplayBuffer, Transition
from src.rl.v5_relay_request_spatial_qmix import V5RelayRequestSpatialQMIXLearner

TRAIN_SEEDS = tuple(range(1101, 1121))
VALIDATION_SEEDS = (2101, 2102, 2103, 2104, 2105)
TEST_SEEDS = tuple(range(3101, 3121))

STAGE_SPECS = (
    {
        "id": 1,
        "name": "Battery OFF",
        "lr": 3e-4,
        "eps_start": 1.0,
        "eps_end": 0.05,
        "eps_decay": 25000,
        "focused_replay": False,
    },
    {
        "id": 2,
        "name": "Battery ON",
        "lr": 1e-4,
        "eps_start": 0.35,
        "eps_end": 0.05,
        "eps_decay": 20000,
        "focused_replay": False,
    },
    {
        "id": 3,
        "name": "Battery ON + transition-focused curriculum",
        "lr": 5e-5,
        "eps_start": 0.20,
        "eps_end": 0.05,
        "eps_decay": 15000,
        "focused_replay": True,
    },
)


def make_train_env(
    method: str,
    stage: int,
    seed: int,
    horizon: int,
    residual_min: float,
    residual_max: float,
):
    if method == "cartesian":
        if stage == 1:
            return CartesianR6Env(CartesianR6Config(seed=seed, horizon=horizon))
        if stage == 2:
            return CartesianR6BatteryEnv(CartesianR6BatteryConfig(seed=seed, horizon=horizon))
        return CartesianR6BatteryFocusEnv(
            CartesianR6BatteryFocusConfig(
                seed=seed,
                horizon=horizon,
                randomize_residual_battery=True,
                residual_battery_min=residual_min,
                residual_battery_max=residual_max,
            )
        )

    if stage == 1:
        return V7R6MacroRelayStableEnv(V7R6MacroRelayStableConfig(seed=seed, horizon=horizon))
    if stage == 2:
        return V8R6MacroRelayBatteryEnv(V8R6MacroRelayBatteryConfig(seed=seed, horizon=horizon))
    return V9R6MacroRelayBatteryFocusEnv(
        V9R6MacroRelayBatteryFocusConfig(
            seed=seed,
            horizon=horizon,
            randomize_residual_battery=True,
            residual_battery_min=residual_min,
            residual_battery_max=residual_max,
        )
    )


def make_common_eval_env(method: str, seed: int, horizon: int):
    """One unchanged battery-ON validation profile used from global step 0 to 90k."""
    if method == "cartesian":
        return CartesianR6BatteryEnv(
            CartesianR6BatteryConfig(
                seed=seed,
                horizon=horizon,
                residual_initial_battery=30.0,
                randomize_residual_uav=True,
            )
        )
    return V8R6MacroRelayBatteryEnv(
        V8R6MacroRelayBatteryConfig(
            seed=seed,
            horizon=horizon,
            residual_initial_battery=30.0,
            randomize_residual_uav=True,
        )
    )


def make_stage_native_eval_env(method: str, stage: int, seed: int, horizon: int):
    """Validation profile matching the current curriculum stage.

    Stage 1 is evaluated with battery OFF so the thesis can show clean
    from-scratch learnability of the action formulation.  Stages 2 and 3 use
    the same fixed battery-ON profile (residual battery 30); Stage-3 training
    randomization is not applied during validation.
    """
    if stage == 1:
        if method == "cartesian":
            return CartesianR6Env(CartesianR6Config(seed=seed, horizon=horizon))
        return V7R6MacroRelayStableEnv(V7R6MacroRelayStableConfig(seed=seed, horizon=horizon))
    return make_common_eval_env(method, seed, horizon)


def make_learner(method: str, env, device: str, lr: float):
    # Both formulations use the same rich 16-feature-per-UAV spatial encoder.
    # This keeps network/observation capacity controlled in the action-design comparison.
    return V5RelayRequestSpatialQMIXLearner(
        env.cfg.num_uavs,
        env.obs_dim,
        env.state_dim,
        env.n_actions,
        grid_size=env.cfg.grid_size,
        coarse_size=env.cfg.coarse_obs_size,
        lr=lr,
        gamma=0.99,
        hidden_dim=64,
        mixing_embed_dim=32,
        device=device,
    )


def reset_optimizer(learner, lr: float):
    # Curriculum changes the data distribution at stage boundaries.  Keep the
    # learned network weights, reset replay and optimizer state, and use the
    # documented stage-specific learning rate.
    params = list(learner.agent.parameters()) + list(learner.mixer.parameters())
    learner.optimizer = torch.optim.Adam(params, lr=float(lr))
    learner.update_targets()


def evaluate(learner, method: str, seeds, horizon: int, stage: int | None = None):
    rows = []
    for seed in seeds:
        env = (
            make_common_eval_env(method, int(seed), horizon)
            if stage is None
            else make_stage_native_eval_env(method, int(stage), int(seed), horizon)
        )
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
        rows.append(
            {
                "seed": int(seed),
                "mean_utility": float(np.mean(rewards)),
                "final20": float(np.mean(rewards[-20:])),
                "mean_active_uavs": float(np.mean(active_counts)),
                "mean_relays": float(np.mean(relay_counts)),
            }
        )
    return {
        "mean_utility": float(np.mean([r["mean_utility"] for r in rows])),
        "std_utility": float(np.std([r["mean_utility"] for r in rows])),
        "mean_relays": float(np.mean([r["mean_relays"] for r in rows])),
        "rows": rows,
    }


def _focus_state(env, margin: float) -> bool:
    if np.any(env.status != ACTIVE):
        return True
    if hasattr(env, "min_active_return_margin"):
        value = env.min_active_return_margin()
        return bool(np.isfinite(value) and value <= float(margin))
    vals = []
    for i in range(env.cfg.num_uavs):
        if env.status[i] != ACTIVE:
            continue
        _, dist_steps = env._nearest_station(env.uav_positions[i])
        threshold = dist_steps * env.cfg.returning_step_drain + env.cfg.return_safety_margin
        vals.append(float(env.battery[i] - threshold))
    return bool(vals and min(vals) <= float(margin))


def _concat_batches(*batches):
    batches = [b for b in batches if b is not None]
    return {k: np.concatenate([b[k] for b in batches], axis=0) for k in batches[0]}


def sample_focused(main_replay, focus_replay, batch_size: int, focus_fraction: float):
    bs = int(batch_size)
    want_focus = int(round(bs * focus_fraction))
    n_focus = min(want_focus, len(focus_replay))
    n_main = bs - n_focus
    if n_main > len(main_replay):
        n_main = len(main_replay)
        n_focus = min(bs - n_main, len(focus_replay))
    if n_main + n_focus < bs:
        n_main = min(bs, len(main_replay))
        n_focus = 0
    main_batch = main_replay.sample(n_main)
    if n_focus <= 0:
        return main_batch, 0.0
    focus_batch = focus_replay.sample(n_focus)
    batch = _concat_batches(main_batch, focus_batch)
    order = np.random.permutation(batch["reward"].shape[0])
    return {k: v[order] for k, v in batch.items()}, float(n_focus / len(order))


def save_history(path: Path, payload: dict):
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=("cartesian", "target-aware"), required=True)
    ap.add_argument("--seed", type=int, default=91)
    ap.add_argument(
        "--stage-transitions",
        type=int,
        default=30000,
        help="Transitions per stage. Thesis protocol uses 30000 for all three stages.",
    )
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--horizon", type=int, default=150)
    ap.add_argument("--buffer-capacity", type=int, default=30000)
    ap.add_argument("--focus-buffer-capacity", type=int, default=15000)
    ap.add_argument("--focus-fraction", type=float, default=0.50)
    ap.add_argument("--focus-margin", type=float, default=10.0)
    ap.add_argument("--post-rejoin-focus-slots", type=int, default=12)
    ap.add_argument("--residual-battery-min", type=float, default=24.0)
    ap.add_argument("--residual-battery-max", type=float, default=36.0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=3000)
    ap.add_argument("--update-every", type=int, default=4)
    ap.add_argument("--target-every", type=int, default=1000)
    ap.add_argument("--torch-threads", type=int, default=1)
    ap.add_argument("--validation-limit", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--outdir", default="results/thesis_full_curriculum")
    args = ap.parse_args()

    if args.stage_transitions <= 0:
        raise ValueError("stage-transitions must be positive")
    if not 0.0 <= args.focus_fraction <= 1.0:
        raise ValueError("focus_fraction must be in [0,1]")

    torch.set_num_threads(max(1, args.torch_threads))
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)

    val_seeds = (
        VALIDATION_SEEDS
        if args.validation_limit is None
        else VALIDATION_SEEDS[: max(1, args.validation_limit)]
    )

    base_out = Path(args.outdir)
    if not base_out.is_absolute():
        base_out = ROOT / base_out
    outdir = base_out / args.method.replace("-", "_")
    outdir.mkdir(parents=True, exist_ok=True)

    # Random initialization: no old checkpoint is loaded here.
    initial_env = make_train_env(
        args.method,
        1,
        TRAIN_SEEDS[0],
        args.horizon,
        args.residual_battery_min,
        args.residual_battery_max,
    )
    learner = make_learner(args.method, initial_env, device, STAGE_SPECS[0]["lr"])

    history = []
    global_steps = 0
    total_episodes = 0
    t0 = time.time()
    best_stage3 = -np.inf
    best_stage3_global_step = None

    # TRUE random-initialized point under the same battery-ON validation profile.
    initial_eval = evaluate(learner, args.method, val_seeds, args.horizon)
    history.append(
        {
            "global_steps": 0,
            "stage": 0,
            "stage_name": "Random initialization",
            "stage_steps": 0,
            "epsilon": None,
            "validation_mean_utility": initial_eval["mean_utility"],
            "validation_std_utility": initial_eval["std_utility"],
            "validation_mean_relays": initial_eval["mean_relays"],
            "stage_validation_mean_utility": None,
            "stage_validation_std_utility": None,
            "stage_validation_mean_relays": None,
            "train_td_loss_mean": None,
            "train_focus_transition_fraction": 0.0,
            "train_batch_focus_fraction": 0.0,
            "runtime_s": float(time.time() - t0),
        }
    )
    print("INITIAL", json.dumps(history[-1]), flush=True)

    run_config = {
        "method": args.method,
        "seed": int(args.seed),
        "random_initialization": True,
        "stage_transitions": int(args.stage_transitions),
        "total_transitions": int(3 * args.stage_transitions),
        "eval_every": int(args.eval_every),
        "common_validation_environment": "R6 + UE mobility ON + battery ON + residual battery 30",
        "common_validation_seeds": list(val_seeds),
        "stage_native_validation": {
            "stage_1": "R6 + UE mobility ON + battery OFF",
            "stage_2": "R6 + UE mobility ON + battery ON + residual battery 30",
            "stage_3": "R6 + UE mobility ON + battery ON + residual battery 30",
        },
        "train_seeds": list(TRAIN_SEEDS),
        "heldout_test_seeds": list(TEST_SEEDS),
        "stage_specs": STAGE_SPECS,
        "warmup_per_stage": int(args.warmup),
        "replay_reset_at_stage_boundary": True,
        "optimizer_reset_at_stage_boundary": True,
        "network_weights_continuous_across_stages": True,
        "reward_changed_across_stages": False,
        "stage_1": "Battery OFF; correlated UE mobility ON",
        "stage_2": "Battery ON; automatic Active->Returning->Charging->Active",
        "stage_3": "Battery ON; residual-battery randomization + transition-focused replay",
        "stage_3_training_residual_battery_range": [
            args.residual_battery_min,
            args.residual_battery_max,
        ],
        "final_checkpoint_selection": "best common battery-ON validation utility during Stage 3 only",
    }
    save_history(outdir / f"seed{args.seed}_run_config.json", run_config)

    for spec in STAGE_SPECS:
        stage = int(spec["id"])
        stage_name = str(spec["name"])
        env = make_train_env(
            args.method,
            stage,
            TRAIN_SEEDS[0],
            args.horizon,
            args.residual_battery_min,
            args.residual_battery_max,
        )
        # Assert architecture compatibility before continuing the same network.
        if (env.obs_dim, env.state_dim, env.n_actions) != (
            initial_env.obs_dim,
            initial_env.state_dim,
            initial_env.n_actions,
        ):
            raise RuntimeError(
                f"Architecture changed across curriculum for {args.method}: "
                f"initial={(initial_env.obs_dim, initial_env.state_dim, initial_env.n_actions)} "
                f"stage{stage}={(env.obs_dim, env.state_dim, env.n_actions)}"
            )
        reset_optimizer(learner, spec["lr"])
        main_replay = ReplayBuffer(args.buffer_capacity, args.seed + stage * 1000)
        focus_replay = ReplayBuffer(args.focus_buffer_capacity, args.seed + stage * 1000 + 101)

        stage_steps = 0
        interval_losses = []
        interval_focus_transitions = 0
        interval_transitions = 0
        interval_batch_focus = []

        print(
            f"STAGE {stage}: {stage_name} | transitions={args.stage_transitions}",
            flush=True,
        )
        while stage_steps < args.stage_transitions:
            total_episodes += 1
            scenario_seed = int(rng.choice(TRAIN_SEEDS))
            out = env.reset(seed=scenario_seed)
            post_rejoin_focus_left = 0

            while stage_steps < args.stage_transitions:
                frac = min(stage_steps / max(1, int(spec["eps_decay"])), 1.0)
                eps = float(spec["eps_start"] + frac * (spec["eps_end"] - spec["eps_start"]))
                status_before = env.status.copy()
                focus_before = (
                    _focus_state(env, args.focus_margin) if spec["focused_replay"] else False
                )

                actions = learner.act(out["obs"], out["action_mask"], eps, rng)
                nxt = env.step(actions)
                status_after = env.status.copy()
                n_return = int(np.sum((status_before == ACTIVE) & (status_after == RETURNING)))
                n_rejoin = int(np.sum((status_before == CHARGING) & (status_after == ACTIVE)))
                if n_rejoin:
                    post_rejoin_focus_left = int(args.post_rejoin_focus_slots)

                tr = Transition(
                    out["obs"],
                    out["state"],
                    np.asarray(actions, dtype=np.int64),
                    float(nxt["reward"]),
                    nxt["obs"],
                    nxt["state"],
                    nxt["action_mask"],
                    float(nxt["truncated"]),
                )
                main_replay.add(tr)

                if spec["focused_replay"]:
                    focus_after = _focus_state(env, args.focus_margin)
                    is_focus = bool(
                        focus_before
                        or focus_after
                        or n_return
                        or n_rejoin
                        or post_rejoin_focus_left > 0
                    )
                    if is_focus:
                        focus_replay.add(tr)
                        interval_focus_transitions += 1
                    if post_rejoin_focus_left > 0:
                        post_rejoin_focus_left -= 1
                interval_transitions += 1

                stage_steps += 1
                global_steps += 1

                if (
                    len(main_replay) >= max(args.warmup, args.batch_size)
                    and stage_steps % args.update_every == 0
                ):
                    if spec["focused_replay"]:
                        batch, actual_focus = sample_focused(
                            main_replay,
                            focus_replay,
                            args.batch_size,
                            args.focus_fraction,
                        )
                        interval_batch_focus.append(actual_focus)
                    else:
                        batch = main_replay.sample(args.batch_size)
                    interval_losses.append(float(learner.train_batch(batch)))

                if stage_steps % args.target_every == 0:
                    learner.update_targets()

                out = nxt

                if stage_steps % args.eval_every == 0 or stage_steps == args.stage_transitions:
                    ev = evaluate(learner, args.method, val_seeds, args.horizon)
                    stage_ev = evaluate(learner, args.method, val_seeds, args.horizon, stage=stage)
                    row = {
                        "global_steps": int(global_steps),
                        "stage": int(stage),
                        "stage_name": stage_name,
                        "stage_steps": int(stage_steps),
                        "epsilon": float(eps),
                        "validation_mean_utility": ev["mean_utility"],
                        "validation_std_utility": ev["std_utility"],
                        "validation_mean_relays": ev["mean_relays"],
                        "stage_validation_mean_utility": stage_ev["mean_utility"],
                        "stage_validation_std_utility": stage_ev["std_utility"],
                        "stage_validation_mean_relays": stage_ev["mean_relays"],
                        "train_td_loss_mean": float(np.mean(interval_losses))
                        if interval_losses
                        else None,
                        "train_focus_transition_fraction": (
                            float(interval_focus_transitions / max(1, interval_transitions))
                            if spec["focused_replay"]
                            else 0.0
                        ),
                        "train_batch_focus_fraction": (
                            float(np.mean(interval_batch_focus)) if interval_batch_focus else 0.0
                        ),
                        "runtime_s": float(time.time() - t0),
                    }
                    history.append(row)
                    print("VAL", json.dumps(row), flush=True)
                    save_history(
                        outdir / f"seed{args.seed}_history.json",
                        {
                            "method": args.method,
                            "seed": args.seed,
                            "stage_boundaries": [
                                args.stage_transitions,
                                2 * args.stage_transitions,
                            ],
                            "history": history,
                        },
                    )
                    interval_losses.clear()
                    interval_focus_transitions = 0
                    interval_transitions = 0
                    interval_batch_focus.clear()

                    if stage == 3 and ev["mean_utility"] > best_stage3:
                        best_stage3 = float(ev["mean_utility"])
                        best_stage3_global_step = int(global_steps)
                        learner.save(str(outdir / f"seed{args.seed}_best_stage3.pt"))

                if nxt["truncated"]:
                    break

        learner.save(str(outdir / f"seed{args.seed}_stage{stage}_end.pt"))

    best_path = outdir / f"seed{args.seed}_best_stage3.pt"
    if not best_path.exists():
        # Smoke configurations can have no eval inside Stage 3 only if misconfigured.
        learner.save(str(best_path))
        best_stage3_global_step = global_steps
        best_stage3 = evaluate(learner, args.method, val_seeds, args.horizon)["mean_utility"]
    learner.load(str(best_path))
    final_val = evaluate(learner, args.method, val_seeds, args.horizon)
    final_test = evaluate(learner, args.method, TEST_SEEDS, args.horizon)

    summary = {
        "method": args.method,
        "seed": int(args.seed),
        "total_transitions": int(global_steps),
        "stage_transitions": int(args.stage_transitions),
        "best_stage3_validation": float(best_stage3),
        "best_stage3_global_step": int(best_stage3_global_step),
        "final_validation": final_val,
        "heldout_test": final_test,
        "stage_boundaries": [
            int(args.stage_transitions),
            int(2 * args.stage_transitions),
        ],
        "stage_labels": [s["name"] for s in STAGE_SPECS],
    }
    save_history(outdir / f"seed{args.seed}_summary.json", summary)
    print(
        "DONE",
        json.dumps(
            {
                "method": args.method,
                "seed": args.seed,
                "best_stage3_validation": best_stage3,
                "heldout_test_mean": final_test["mean_utility"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
