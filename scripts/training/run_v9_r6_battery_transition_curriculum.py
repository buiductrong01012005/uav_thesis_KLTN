from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.envs.v4_fixed_mobile_battery_env import ACTIVE, CHARGING, RETURNING
from src.envs.v9_r6_macro_relay_battery_focus_env import (
    V9R6MacroRelayBatteryFocusConfig,
    V9R6MacroRelayBatteryFocusEnv,
)
from src.rl.qmix import ReplayBuffer, Transition
from src.rl.v5_relay_request_spatial_qmix import V5RelayRequestSpatialQMIXLearner

TRAIN_SEEDS = tuple(range(1101, 1121))
VALIDATION_SEEDS = (2101, 2102, 2103, 2104, 2105)
TEST_SEEDS = tuple(range(3101, 3121))


def make_env(
    seed: int,
    horizon: int = 150,
    *,
    randomized_training_battery: bool = False,
    residual_battery_min: float = 24.0,
    residual_battery_max: float = 36.0,
):
    return V9R6MacroRelayBatteryFocusEnv(
        V9R6MacroRelayBatteryFocusConfig(
            seed=seed,
            horizon=horizon,
            randomize_residual_battery=bool(randomized_training_battery),
            residual_battery_min=float(residual_battery_min),
            residual_battery_max=float(residual_battery_max),
            # Canonical evaluation initial condition remains 30.
            residual_initial_battery=30.0,
        )
    )


def movement_only_mask(env):
    mask = env.get_action_mask().copy()
    mask[:, 5:] = 0
    return mask


def discover_init_checkpoint(seed: int, explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit).expanduser()
        if not p.is_absolute():
            p = ROOT / p
        if not p.exists():
            raise FileNotFoundError(f"Initial checkpoint not found: {p}")
        return p

    candidates = [
        ROOT / "results" / "v8_r6_macro_relay_battery_curriculum" / f"seed{seed}_best.pt",
        ROOT / "results" / "v8_r6_macro_relay_battery_curriculum" / "seed91_best.pt",
        ROOT / "results" / "v8_seed91_best_actual.pt",  # local smoke convenience
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not auto-discover a V8 battery checkpoint. "
        "Pass --init-checkpoint <path-to-V8-seed91_best.pt>."
    )


def _return_focus_state(env, focus_margin: float) -> bool:
    """Training sampler only: identify rare availability-transition states.

    No policy action is masked or forced by this predicate.
    """
    if np.any(env.status != ACTIVE):
        return True
    margin = env.min_active_return_margin()
    return bool(np.isfinite(margin) and margin <= float(focus_margin))


def _concat_batches(*batches: dict) -> dict:
    batches = tuple(b for b in batches if b is not None)
    if not batches:
        raise ValueError("at least one batch is required")
    return {k: np.concatenate([b[k] for b in batches], axis=0) for k in batches[0]}


def sample_transition_focused_batch(
    main_replay: ReplayBuffer,
    focus_replay: ReplayBuffer,
    batch_size: int,
    focus_fraction: float,
):
    """Mix uniform replay with extra transition-window samples.

    Main replay still contains every transition.  Focus replay only changes the
    *sampling distribution* during training; reward and environment dynamics are
    untouched.
    """
    bs = int(batch_size)
    desired_focus = int(round(bs * float(focus_fraction)))
    n_focus = min(desired_focus, len(focus_replay))
    n_main = bs - n_focus
    if n_main > len(main_replay):
        n_main = len(main_replay)
        n_focus = min(bs - n_main, len(focus_replay))
    if n_main + n_focus < bs:
        # During early warmup, fall back to the main buffer if possible.
        n_main = min(bs, len(main_replay))
        n_focus = 0
    if n_main < 1:
        raise ValueError("main replay does not contain enough data")

    main_batch = main_replay.sample(n_main)
    if n_focus <= 0:
        return main_batch, 0.0
    focus_batch = focus_replay.sample(n_focus)
    batch = _concat_batches(main_batch, focus_batch)
    # Shuffle after concatenation so the learner sees no ordering artifact.
    order = np.random.permutation(batch["reward"].shape[0])
    batch = {k: v[order] for k, v in batch.items()}
    return batch, float(n_focus / batch["reward"].shape[0])


def evaluate(learner, seeds, horizon=150, *, movement_only=False, focus_margin=10.0):
    rows = []
    for seed in seeds:
        env = make_env(seed, horizon, randomized_training_battery=False)
        out = env.reset(seed=seed)
        rng = np.random.default_rng(990000 + int(seed))

        rewards = []
        pre_rewards = []
        unavailable_rewards = []
        post_rejoin_rewards = []
        relays = []
        active_counts = []
        macro_count = conflicts = role_changes = 0
        return_events = charge_arrivals = rejoin_events = 0
        target_counts = Counter()
        assisted_move_counts = Counter()
        prev_roles = env.roles.copy()
        seen_unavailable = False
        seen_rejoin_after_unavailable = False
        first_return_slot = first_charge_slot = first_rejoin_slot = None

        phase_relay_actions = Counter()
        phase_active_decisions = Counter()

        while True:
            active_before = int(np.sum(env.status == ACTIVE))
            status_before = env.status.copy()
            if active_before < env.cfg.num_uavs:
                seen_unavailable = True
            elif seen_unavailable:
                seen_rejoin_after_unavailable = True

            if not seen_unavailable:
                margin = env.min_active_return_margin()
                phase = (
                    "pre_return_near"
                    if np.isfinite(margin) and margin <= float(focus_margin)
                    else "pre_return_healthy"
                )
            elif active_before < env.cfg.num_uavs:
                phase = "unavailable"
            else:
                phase = "post_rejoin"

            mask = movement_only_mask(env) if movement_only else out["action_mask"]
            actions = learner.act(out["obs"], mask, 0.0, rng)
            active_mask = status_before == ACTIVE
            phase_active_decisions[phase] += int(np.sum(active_mask))
            phase_relay_actions[phase] += sum(
                int(a) >= 5 for a, active in zip(actions, active_mask) if active
            )

            out = env.step(actions)
            info = out["info"]
            reward = float(out["reward"])
            rewards.append(reward)
            active_counts.append(active_before)
            relays.append(int(info["num_relay"]))

            if not seen_unavailable:
                pre_rewards.append(reward)
            elif active_before < env.cfg.num_uavs:
                unavailable_rewards.append(reward)
            elif seen_rejoin_after_unavailable:
                post_rejoin_rewards.append(reward)

            status_after = env.status.copy()
            n_return = int(np.sum((status_before == ACTIVE) & (status_after == RETURNING)))
            n_charge = int(np.sum((status_before == RETURNING) & (status_after == CHARGING)))
            n_rejoin = int(np.sum((status_before == CHARGING) & (status_after == ACTIVE)))
            return_events += n_return
            charge_arrivals += n_charge
            rejoin_events += n_rejoin
            if n_return and first_return_slot is None:
                first_return_slot = int(env.t)
            if n_charge and first_charge_slot is None:
                first_charge_slot = int(env.t)
            if n_rejoin and first_rejoin_slot is None:
                first_rejoin_slot = int(env.t)

            role_changes += int(np.sum(env.roles != prev_roles))
            prev_roles = env.roles.copy()
            accepted = info.get("accepted_macro_relays_this_slot", [])
            macro_count += len(accepted)
            conflicts += int(info.get("relay_claim_conflicts_this_slot", 0))
            for actor, target, move in accepted:
                target_counts[int(target)] += 1
                assisted_move_counts[int(move)] += 1
            if out["truncated"]:
                break

        relay_fracs = {
            phase: float(phase_relay_actions[phase] / max(1, phase_active_decisions[phase]))
            for phase in (
                "pre_return_healthy",
                "pre_return_near",
                "unavailable",
                "post_rejoin",
            )
        }
        row = {
            "seed": int(seed),
            "residual_uav_idx": int(env.residual_uav_idx),
            "mean_utility": float(np.mean(rewards)),
            "final20": float(np.mean(rewards[-20:])),
            "pre_return_mean_utility": float(np.mean(pre_rewards)) if pre_rewards else None,
            "unavailable_mean_utility": float(np.mean(unavailable_rewards))
            if unavailable_rewards
            else None,
            "post_rejoin_mean_utility": float(np.mean(post_rejoin_rewards))
            if post_rejoin_rewards
            else None,
            "mean_active_uavs": float(np.mean(active_counts)),
            "mean_relays": float(np.mean(relays)),
            "relay_count_switches": int(sum(a != b for a, b in zip(relays, relays[1:]))),
            "role_changes": int(role_changes),
            "macro_relay_count": int(macro_count),
            "claim_conflicts": int(conflicts),
            "claim_accept_rate": float(macro_count / max(1, macro_count + conflicts)),
            "return_events": int(return_events),
            "charge_arrivals": int(charge_arrivals),
            "rejoin_events": int(rejoin_events),
            "first_return_slot": first_return_slot,
            "first_charge_slot": first_charge_slot,
            "first_rejoin_slot": first_rejoin_slot,
            "relay_action_fraction_by_phase": relay_fracs,
            "target_counts": {str(k): int(v) for k, v in sorted(target_counts.items())},
            "assisted_move_counts": {
                str(k): int(v) for k, v in sorted(assisted_move_counts.items())
            },
        }
        rows.append(row)

    def mean_optional(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return float(np.mean(vals)) if vals else None

    def mean_phase_frac(phase):
        return float(np.mean([r["relay_action_fraction_by_phase"][phase] for r in rows]))

    mean_macro = float(np.mean([r["macro_relay_count"] for r in rows]))
    mean_conflicts = float(np.mean([r["claim_conflicts"] for r in rows]))
    return {
        "mean_utility": float(np.mean([r["mean_utility"] for r in rows])),
        "std_utility": float(np.std([r["mean_utility"] for r in rows])),
        "pre_return_mean_utility": mean_optional("pre_return_mean_utility"),
        "unavailable_mean_utility": mean_optional("unavailable_mean_utility"),
        "post_rejoin_mean_utility": mean_optional("post_rejoin_mean_utility"),
        "mean_active_uavs": float(np.mean([r["mean_active_uavs"] for r in rows])),
        "mean_relays": float(np.mean([r["mean_relays"] for r in rows])),
        "mean_role_changes": float(np.mean([r["role_changes"] for r in rows])),
        "mean_macro_relay_count": mean_macro,
        "mean_claim_conflicts": mean_conflicts,
        "mean_claim_accept_rate": float(mean_macro / max(1e-9, mean_macro + mean_conflicts)),
        "mean_return_events": float(np.mean([r["return_events"] for r in rows])),
        "mean_charge_arrivals": float(np.mean([r["charge_arrivals"] for r in rows])),
        "mean_rejoin_events": float(np.mean([r["rejoin_events"] for r in rows])),
        "relay_action_fraction_by_phase": {
            phase: mean_phase_frac(phase)
            for phase in (
                "pre_return_healthy",
                "pre_return_near",
                "unavailable",
                "post_rejoin",
            )
        },
        "rows": rows,
    }


def compare(dynamic, movement_only):
    paired = []
    wins = 0
    for d, m in zip(dynamic["rows"], movement_only["rows"]):
        if d["seed"] != m["seed"]:
            raise ValueError("Evaluation seed order mismatch")
        gain = float(d["mean_utility"] - m["mean_utility"])
        wins += int(gain > 0.0)
        paired.append(
            {
                "seed": int(d["seed"]),
                "macro_dynamic": float(d["mean_utility"]),
                "movement_only_5A": float(m["mean_utility"]),
                "gain_pp": float(100.0 * gain),
            }
        )
    return {
        "dynamic_minus_5A_pp": float(
            100.0 * (dynamic["mean_utility"] - movement_only["mean_utility"])
        ),
        "win_fraction": float(wins / max(1, len(paired))),
        "wins": int(wins),
        "n": int(len(paired)),
        "paired_rows": paired,
    }


def compare_curriculum(finetuned, initial):
    paired = []
    wins = 0
    for f, b in zip(finetuned["rows"], initial["rows"]):
        if f["seed"] != b["seed"]:
            raise ValueError("Evaluation seed order mismatch")
        gain = float(f["mean_utility"] - b["mean_utility"])
        wins += int(gain > 0.0)
        paired.append(
            {
                "seed": int(f["seed"]),
                "finetuned": float(f["mean_utility"]),
                "initial_v8_checkpoint": float(b["mean_utility"]),
                "gain_pp": float(100.0 * gain),
            }
        )

    def optional_gain_pp(key):
        a = finetuned.get(key)
        b = initial.get(key)
        if a is None or b is None:
            return None
        return float(100.0 * (a - b))

    return {
        "finetuned_minus_initial_pp": float(
            100.0 * (finetuned["mean_utility"] - initial["mean_utility"])
        ),
        "unavailable_gain_pp": optional_gain_pp("unavailable_mean_utility"),
        "post_rejoin_gain_pp": optional_gain_pp("post_rejoin_mean_utility"),
        "win_fraction": float(wins / max(1, len(paired))),
        "wins": int(wins),
        "n": int(len(paired)),
        "paired_rows": paired,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total-transitions", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=91)
    ap.add_argument("--horizon", type=int, default=150)
    ap.add_argument("--eval-every", type=int, default=5000)
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
    ap.add_argument("--eps-start", type=float, default=0.20)
    ap.add_argument("--eps-end", type=float, default=0.05)
    ap.add_argument("--eps-decay-steps", type=int, default=15000)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--init-checkpoint", default=None)
    ap.add_argument("--outdir", default="results/v9_r6_battery_transition_curriculum")
    ap.add_argument("--torch-threads", type=int, default=1)
    ap.add_argument("--validation-limit", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--test-limit", type=int, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if not 0.0 <= args.focus_fraction <= 1.0:
        raise ValueError("focus_fraction must be in [0,1]")

    validation_seeds = (
        VALIDATION_SEEDS
        if args.validation_limit is None
        else VALIDATION_SEEDS[: max(1, args.validation_limit)]
    )
    test_seeds = TEST_SEEDS if args.test_limit is None else TEST_SEEDS[: max(1, args.test_limit)]

    torch.set_num_threads(max(1, args.torch_threads))
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

    init_checkpoint = discover_init_checkpoint(args.seed, args.init_checkpoint)
    env = make_env(
        TRAIN_SEEDS[0],
        args.horizon,
        randomized_training_battery=True,
        residual_battery_min=args.residual_battery_min,
        residual_battery_max=args.residual_battery_max,
    )
    learner = V5RelayRequestSpatialQMIXLearner(
        env.cfg.num_uavs,
        env.obs_dim,
        env.state_dim,
        env.n_actions,
        grid_size=env.cfg.grid_size,
        coarse_size=env.cfg.coarse_obs_size,
        lr=args.lr,
        gamma=0.99,
        hidden_dim=64,
        mixing_embed_dim=32,
        device=device,
    )
    learner.load(str(init_checkpoint))

    initial_validation = evaluate(
        learner,
        validation_seeds,
        args.horizon,
        movement_only=False,
        focus_margin=args.focus_margin,
    )
    initial_validation_5a = evaluate(
        learner,
        validation_seeds,
        args.horizon,
        movement_only=True,
        focus_margin=args.focus_margin,
    )

    run_config = {
        "total_transitions": int(args.total_transitions),
        "seed": int(args.seed),
        "horizon": int(args.horizon),
        "eval_every": int(args.eval_every),
        "buffer_capacity": int(args.buffer_capacity),
        "focus_buffer_capacity": int(args.focus_buffer_capacity),
        "focus_fraction": float(args.focus_fraction),
        "focus_margin": float(args.focus_margin),
        "post_rejoin_focus_slots": int(args.post_rejoin_focus_slots),
        "training_residual_battery_range": [
            float(args.residual_battery_min),
            float(args.residual_battery_max),
        ],
        "evaluation_residual_battery": 30.0,
        "batch_size": int(args.batch_size),
        "warmup": int(args.warmup),
        "update_every": int(args.update_every),
        "target_every": int(args.target_every),
        "epsilon_start": float(args.eps_start),
        "epsilon_end": float(args.eps_end),
        "eps_decay_steps": int(args.eps_decay_steps),
        "lr": float(args.lr),
        "train_seeds": list(TRAIN_SEEDS),
        "validation_seeds": list(validation_seeds),
        "test_seeds": list(test_seeds),
        "curriculum_from_v8_battery": True,
        "init_checkpoint": str(init_checkpoint),
        "checkpoint_selection": "highest standard-battery dynamic mean utility on validation seeds only",
        "test_policy": "held-out test seeds evaluated only after checkpoint selection",
        "reward_changed": False,
        "action_changed": False,
        "action_gating_rule_added": False,
        "training_only_changes": [
            "randomized residual initial battery",
            "transition-focused replay oversampling around Return/Charging/Rejoin",
        ],
    }
    (outdir / f"seed{args.seed}_run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    main_replay = ReplayBuffer(args.buffer_capacity, args.seed)
    focus_replay = ReplayBuffer(args.focus_buffer_capacity, args.seed + 101)
    steps = episodes = 0
    best = -1.0
    best_step = None
    history = []
    t0 = time.time()

    interval_losses = []
    interval_batch_focus_fractions = []
    interval_focus_transitions = 0
    interval_total_transitions = 0
    interval_residual_batteries = []
    interval_relay_actions = 0
    interval_agent_actions = 0
    interval_return_events = 0
    interval_rejoin_events = 0

    print(
        "SCENARIO=V9 V8-policy semantics + battery transition curriculum (no reward/action rule change)",
        flush=True,
    )
    print(f"INIT_CHECKPOINT={init_checkpoint}", flush=True)
    print(f"DEVICE={device}", flush=True)
    print(
        "INITIAL_VAL",
        json.dumps(
            {
                "dynamic_mean_utility": initial_validation["mean_utility"],
                "movement_only_5A_mean_utility": initial_validation_5a["mean_utility"],
                "unavailable_mean_utility": initial_validation["unavailable_mean_utility"],
                "relay_fraction_by_phase": initial_validation["relay_action_fraction_by_phase"],
            }
        ),
        flush=True,
    )

    while steps < args.total_transitions:
        episodes += 1
        scenario_seed = int(rng.choice(TRAIN_SEEDS))
        out = env.reset(seed=scenario_seed)
        interval_residual_batteries.append(float(env.sampled_residual_initial_battery))
        post_rejoin_focus_left = 0

        while steps < args.total_transitions:
            frac = min(steps / max(1, args.eps_decay_steps), 1.0)
            eps = args.eps_start + frac * (args.eps_end - args.eps_start)
            status_before = env.status.copy()
            focus_before = _return_focus_state(env, args.focus_margin)

            actions = learner.act(out["obs"], out["action_mask"], eps, rng)
            active_mask = status_before == ACTIVE
            interval_agent_actions += int(np.sum(active_mask))
            interval_relay_actions += sum(
                int(a) >= 5 for a, active in zip(actions, active_mask) if active
            )

            nxt = env.step(actions)
            status_after = env.status.copy()
            n_return = int(np.sum((status_before == ACTIVE) & (status_after == RETURNING)))
            n_rejoin = int(np.sum((status_before == CHARGING) & (status_after == ACTIVE)))
            interval_return_events += n_return
            interval_rejoin_events += n_rejoin
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

            focus_after = _return_focus_state(env, args.focus_margin)
            focus_transition = bool(
                focus_before or focus_after or n_return or n_rejoin or post_rejoin_focus_left > 0
            )
            if focus_transition:
                focus_replay.add(tr)
                interval_focus_transitions += 1
            interval_total_transitions += 1
            if post_rejoin_focus_left > 0:
                post_rejoin_focus_left -= 1

            steps += 1
            if (
                len(main_replay) >= max(args.warmup, args.batch_size)
                and steps % args.update_every == 0
            ):
                batch, actual_focus_fraction = sample_transition_focused_batch(
                    main_replay, focus_replay, args.batch_size, args.focus_fraction
                )
                interval_batch_focus_fractions.append(actual_focus_fraction)
                interval_losses.append(float(learner.train_batch(batch)))
            if steps % args.target_every == 0:
                learner.update_targets()
            out = nxt

            if steps % args.eval_every == 0:
                dyn = evaluate(
                    learner,
                    validation_seeds,
                    args.horizon,
                    movement_only=False,
                    focus_margin=args.focus_margin,
                )
                no_relay = evaluate(
                    learner,
                    validation_seeds,
                    args.horizon,
                    movement_only=True,
                    focus_margin=args.focus_margin,
                )
                cmp = compare(dyn, no_relay)
                init_cmp = compare_curriculum(dyn, initial_validation)
                row = {
                    "steps": int(steps),
                    "episodes": int(episodes),
                    "epsilon": float(eps),
                    "validation_dynamic_mean_utility": dyn["mean_utility"],
                    "validation_5A_mean_utility": no_relay["mean_utility"],
                    "validation_dynamic_minus_5A_pp": cmp["dynamic_minus_5A_pp"],
                    "validation_win_fraction": cmp["win_fraction"],
                    "validation_gain_vs_initial_v8_pp": init_cmp["finetuned_minus_initial_pp"],
                    "validation_unavailable_gain_vs_initial_v8_pp": init_cmp["unavailable_gain_pp"],
                    "validation_pre_return_utility": dyn["pre_return_mean_utility"],
                    "validation_unavailable_utility": dyn["unavailable_mean_utility"],
                    "validation_post_rejoin_utility": dyn["post_rejoin_mean_utility"],
                    "validation_relay_action_fraction_by_phase": dyn[
                        "relay_action_fraction_by_phase"
                    ],
                    "dynamic_mean_relays": dyn["mean_relays"],
                    "dynamic_mean_claim_accept_rate": dyn["mean_claim_accept_rate"],
                    "train_td_loss_mean": float(np.mean(interval_losses))
                    if interval_losses
                    else None,
                    "train_relay_action_fraction_active": float(
                        interval_relay_actions / max(1, interval_agent_actions)
                    ),
                    "train_focus_transition_fraction": float(
                        interval_focus_transitions / max(1, interval_total_transitions)
                    ),
                    "train_batch_focus_fraction": float(np.mean(interval_batch_focus_fractions))
                    if interval_batch_focus_fractions
                    else 0.0,
                    "train_mean_sampled_residual_battery": float(
                        np.mean(interval_residual_batteries)
                    )
                    if interval_residual_batteries
                    else None,
                    "train_return_events": int(interval_return_events),
                    "train_rejoin_events": int(interval_rejoin_events),
                    "runtime_s": float(time.time() - t0),
                }
                history.append(row)
                print("VAL", json.dumps(row), flush=True)
                if dyn["mean_utility"] > best:
                    best = dyn["mean_utility"]
                    best_step = int(steps)
                    learner.save(str(outdir / f"seed{args.seed}_best.pt"))
                (outdir / f"seed{args.seed}_history.json").write_text(
                    json.dumps(
                        {"best": best, "best_step": best_step, "history": history},
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                interval_losses.clear()
                interval_batch_focus_fractions.clear()
                interval_focus_transitions = 0
                interval_total_transitions = 0
                interval_residual_batteries.clear()
                interval_relay_actions = 0
                interval_agent_actions = 0
                interval_return_events = 0
                interval_rejoin_events = 0
            if nxt["truncated"]:
                break

    best_path = outdir / f"seed{args.seed}_best.pt"
    if not best_path.exists():
        raise RuntimeError("No best checkpoint was saved")
    learner.load(str(best_path))

    val_dyn = evaluate(
        learner,
        validation_seeds,
        args.horizon,
        movement_only=False,
        focus_margin=args.focus_margin,
    )
    val_5a = evaluate(
        learner,
        validation_seeds,
        args.horizon,
        movement_only=True,
        focus_margin=args.focus_margin,
    )
    val_cmp = compare(val_dyn, val_5a)
    val_init_cmp = compare_curriculum(val_dyn, initial_validation)

    # Held-out test is touched only after checkpoint selection.
    test_dyn = evaluate(
        learner,
        test_seeds,
        args.horizon,
        movement_only=False,
        focus_margin=args.focus_margin,
    )
    test_5a = evaluate(
        learner,
        test_seeds,
        args.horizon,
        movement_only=True,
        focus_margin=args.focus_margin,
    )
    test_cmp = compare(test_dyn, test_5a)

    initial_learner = V5RelayRequestSpatialQMIXLearner(
        env.cfg.num_uavs,
        env.obs_dim,
        env.state_dim,
        env.n_actions,
        grid_size=env.cfg.grid_size,
        coarse_size=env.cfg.coarse_obs_size,
        lr=args.lr,
        gamma=0.99,
        hidden_dim=64,
        mixing_embed_dim=32,
        device=device,
    )
    initial_learner.load(str(init_checkpoint))
    initial_test = evaluate(
        initial_learner,
        test_seeds,
        args.horizon,
        movement_only=False,
        focus_margin=args.focus_margin,
    )
    test_init_cmp = compare_curriculum(test_dyn, initial_test)

    relay_phase = test_dyn["relay_action_fraction_by_phase"]
    gate = {
        "battery_lifecycle_exercised": bool(
            test_dyn["mean_return_events"] >= 1.0
            and test_dyn["mean_charge_arrivals"] >= 1.0
            and test_dyn["mean_rejoin_events"] >= 1.0
        ),
        "positive_test_gain_vs_v8": bool(test_init_cmp["finetuned_minus_initial_pp"] > 0.0),
        "positive_unavailable_gain_vs_v8": bool(test_init_cmp["unavailable_gain_pp"] > 0.0),
        "positive_test_gain_vs_5A_ablation": bool(test_cmp["dynamic_minus_5A_pp"] > 0.0),
        "majority_test_seeds_beat_5A_ablation": bool(test_cmp["win_fraction"] > 0.5),
        "near_return_relay_lower_than_healthy": bool(
            relay_phase["pre_return_near"] < relay_phase["pre_return_healthy"]
        ),
    }
    gate["pass_anticipatory_curriculum_gate"] = bool(
        gate["battery_lifecycle_exercised"]
        and gate["positive_test_gain_vs_v8"]
        and gate["positive_unavailable_gain_vs_v8"]
    )

    summary = {
        "seed": int(args.seed),
        "steps": int(steps),
        "episodes": int(episodes),
        "init_checkpoint": str(init_checkpoint),
        "best_validation": float(best),
        "best_step": best_step,
        "initial_validation_v8": {
            "dynamic": initial_validation,
            "movement_only_5A": initial_validation_5a,
        },
        "validation": {
            "dynamic": val_dyn,
            "movement_only_5A": val_5a,
            "comparison": val_cmp,
            "curriculum_vs_initial_v8": val_init_cmp,
        },
        "heldout_test": {
            "dynamic": test_dyn,
            "movement_only_5A": test_5a,
            "comparison": test_cmp,
            "initial_v8_checkpoint_dynamic": initial_test,
            "curriculum_vs_initial_v8": test_init_cmp,
        },
        "anticipatory_gate": gate,
        "scenario": {
            "map": "canonical V4 topology",
            "backhaul_radius_cells": 6.0,
            "ue_mobility": "canonical stochastic correlated-group mobility",
            "battery": True,
            "evaluation_residual_battery": 30.0,
            "training_residual_battery_range": [
                float(args.residual_battery_min),
                float(args.residual_battery_max),
            ],
            "battery_lifecycle": "automatic Active->Returning->Charging->Active",
            "proactive_return_action": False,
            "action_design": "unchanged V8/V7 10-action macro RelayTo",
            "reward": "mean user satisfaction only; unchanged",
            "policy_action_gating": "none",
            "training_curriculum": "residual-battery domain randomization + transition-focused replay only",
            "focus_margin_for_replay_sampling": float(args.focus_margin),
            "focus_fraction": float(args.focus_fraction),
            "train_seeds": list(TRAIN_SEEDS),
            "validation_seeds": list(validation_seeds),
            "heldout_test_seeds": list(test_seeds),
        },
    }
    (outdir / f"seed{args.seed}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(
        "DONE",
        json.dumps(
            {
                "best_validation": best,
                "best_step": best_step,
                "test_gain_vs_v8_pp": test_init_cmp["finetuned_minus_initial_pp"],
                "test_unavailable_gain_vs_v8_pp": test_init_cmp["unavailable_gain_pp"],
                "test_gain_vs_5A_pp": test_cmp["dynamic_minus_5A_pp"],
                "test_win_fraction_vs_5A": test_cmp["win_fraction"],
                "test_relay_action_fraction_by_phase": relay_phase,
                "pass_anticipatory_curriculum_gate": gate["pass_anticipatory_curriculum_gate"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
