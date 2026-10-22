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
from src.envs.v8_r6_macro_relay_battery_env import (
    V8R6MacroRelayBatteryConfig,
    V8R6MacroRelayBatteryEnv,
)
from src.rl.qmix import ReplayBuffer, Transition
from src.rl.v5_relay_request_spatial_qmix import V5RelayRequestSpatialQMIXLearner

TRAIN_SEEDS = tuple(range(1101, 1121))
VALIDATION_SEEDS = (2101, 2102, 2103, 2104, 2105)
TEST_SEEDS = tuple(range(3101, 3121))


def make_env(seed: int, horizon: int = 150):
    return V8R6MacroRelayBatteryEnv(V8R6MacroRelayBatteryConfig(seed=seed, horizon=horizon))


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
        ROOT / "results" / "v7_r6_macro_relay_stability_pilot" / f"seed{seed}_best.pt",
        ROOT / "results" / "v7_r6_macro_relay_stability_pilot" / "seed91_best.pt",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not auto-discover a V7 no-battery checkpoint. "
        "Pass --init-checkpoint <path-to-seed91_best.pt>."
    )


def evaluate(learner, seeds, horizon=150, *, movement_only=False):
    rows = []
    for seed in seeds:
        env = make_env(seed, horizon)
        out = env.reset(seed=seed)
        rng = np.random.default_rng(980000 + int(seed))

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

        while True:
            active_before = int(np.sum(env.status == ACTIVE))
            status_before = env.status.copy()
            if active_before < env.cfg.num_uavs:
                seen_unavailable = True
            elif seen_unavailable:
                seen_rejoin_after_unavailable = True

            mask = movement_only_mask(env) if movement_only else out["action_mask"]
            actions = learner.act(out["obs"], mask, 0.0, rng)
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
            "target_counts": {str(k): int(v) for k, v in sorted(target_counts.items())},
            "assisted_move_counts": {
                str(k): int(v) for k, v in sorted(assisted_move_counts.items())
            },
        }
        rows.append(row)

    def mean_optional(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return float(np.mean(vals)) if vals else None

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
    delta = float(dynamic["mean_utility"] - movement_only["mean_utility"])
    return {
        "dynamic_minus_5A_pp": float(100.0 * delta),
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
                "initial_v7_checkpoint": float(b["mean_utility"]),
                "gain_pp": float(100.0 * gain),
            }
        )
    return {
        "finetuned_minus_initial_pp": float(
            100.0 * (finetuned["mean_utility"] - initial["mean_utility"])
        ),
        "win_fraction": float(wins / max(1, len(paired))),
        "wins": int(wins),
        "n": int(len(paired)),
        "paired_rows": paired,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total-transitions", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=91)
    ap.add_argument("--horizon", type=int, default=150)
    ap.add_argument("--eval-every", type=int, default=5000)
    ap.add_argument("--buffer-capacity", type=int, default=30000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=3000)
    ap.add_argument("--update-every", type=int, default=4)
    ap.add_argument("--target-every", type=int, default=1000)
    ap.add_argument("--eps-start", type=float, default=0.35)
    ap.add_argument("--eps-end", type=float, default=0.05)
    ap.add_argument("--eps-decay-steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--init-checkpoint", default=None)
    ap.add_argument("--outdir", default="results/v8_r6_macro_relay_battery_curriculum")
    ap.add_argument("--torch-threads", type=int, default=1)
    ap.add_argument("--validation-limit", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--test-limit", type=int, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

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
    env = make_env(TRAIN_SEEDS[0], args.horizon)
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

    initial_validation = evaluate(learner, validation_seeds, args.horizon, movement_only=False)
    initial_validation_5a = evaluate(learner, validation_seeds, args.horizon, movement_only=True)

    run_config = {
        "total_transitions": int(args.total_transitions),
        "seed": int(args.seed),
        "horizon": int(args.horizon),
        "eval_every": int(args.eval_every),
        "buffer_capacity": int(args.buffer_capacity),
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
        "curriculum_from_no_battery": True,
        "init_checkpoint": str(init_checkpoint),
        "checkpoint_selection": "highest battery-aware macro-dynamic mean utility on validation seeds only",
        "test_policy": "held-out test seeds evaluated only after checkpoint selection",
        "battery": "environment-controlled Active->Returning->Charging->Active",
        "reward_changed": False,
        "action_changed": False,
        "action_design": "0-4 Access movement; 5+j RelayTo(UAV j), self masked; assisted one-step movement",
        "routing": "target-exclusive designated Relay; geometry-ranked conflict resolver",
    }
    (outdir / f"seed{args.seed}_run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    replay = ReplayBuffer(args.buffer_capacity, args.seed)
    steps = episodes = 0
    best = -1.0
    best_step = None
    history = []
    t0 = time.time()
    interval_losses = []
    interval_relay_actions = 0
    interval_agent_actions = 0
    interval_accepted_claims = 0
    interval_rejected_claims = 0
    interval_return_events = 0
    interval_rejoin_events = 0

    print(
        "SCENARIO=V8 R6 + natural UE mobility + stable macro RelayTo + automatic battery lifecycle",
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
                "post_rejoin_mean_utility": initial_validation["post_rejoin_mean_utility"],
            }
        ),
        flush=True,
    )

    while steps < args.total_transitions:
        episodes += 1
        scenario_seed = int(rng.choice(TRAIN_SEEDS))
        out = env.reset(seed=scenario_seed)
        while steps < args.total_transitions:
            frac = min(steps / max(1, args.eps_decay_steps), 1.0)
            eps = args.eps_start + frac * (args.eps_end - args.eps_start)
            status_before = env.status.copy()
            actions = learner.act(out["obs"], out["action_mask"], eps, rng)
            active_mask = status_before == ACTIVE
            interval_agent_actions += int(np.sum(active_mask))
            interval_relay_actions += sum(
                int(a) >= 5 for a, active in zip(actions, active_mask) if active
            )
            nxt = env.step(actions)
            status_after = env.status.copy()
            interval_return_events += int(
                np.sum((status_before == ACTIVE) & (status_after == RETURNING))
            )
            interval_rejoin_events += int(
                np.sum((status_before == CHARGING) & (status_after == ACTIVE))
            )
            interval_accepted_claims += len(nxt["info"].get("accepted_macro_relays_this_slot", []))
            interval_rejected_claims += int(nxt["info"].get("relay_claim_conflicts_this_slot", 0))
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
                interval_losses.append(float(learner.train_batch(replay.sample(args.batch_size))))
            if steps % args.target_every == 0:
                learner.update_targets()
            out = nxt

            if steps % args.eval_every == 0:
                dyn = evaluate(learner, validation_seeds, args.horizon, movement_only=False)
                no_relay = evaluate(learner, validation_seeds, args.horizon, movement_only=True)
                cmp = compare(dyn, no_relay)
                row = {
                    "steps": int(steps),
                    "episodes": int(episodes),
                    "epsilon": float(eps),
                    "validation_dynamic_mean_utility": dyn["mean_utility"],
                    "validation_5A_mean_utility": no_relay["mean_utility"],
                    "validation_dynamic_minus_5A_pp": cmp["dynamic_minus_5A_pp"],
                    "validation_win_fraction": cmp["win_fraction"],
                    "validation_pre_return_utility": dyn["pre_return_mean_utility"],
                    "validation_unavailable_utility": dyn["unavailable_mean_utility"],
                    "validation_post_rejoin_utility": dyn["post_rejoin_mean_utility"],
                    "validation_mean_active_uavs": dyn["mean_active_uavs"],
                    "dynamic_mean_relays": dyn["mean_relays"],
                    "dynamic_mean_macro_relay_count": dyn["mean_macro_relay_count"],
                    "dynamic_mean_claim_accept_rate": dyn["mean_claim_accept_rate"],
                    "dynamic_mean_return_events": dyn["mean_return_events"],
                    "dynamic_mean_rejoin_events": dyn["mean_rejoin_events"],
                    "train_td_loss_mean": float(np.mean(interval_losses))
                    if interval_losses
                    else None,
                    "train_relay_action_fraction_active": float(
                        interval_relay_actions / max(1, interval_agent_actions)
                    ),
                    "train_claim_accept_rate": float(
                        interval_accepted_claims
                        / max(1, interval_accepted_claims + interval_rejected_claims)
                    ),
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
                interval_relay_actions = 0
                interval_agent_actions = 0
                interval_accepted_claims = 0
                interval_rejected_claims = 0
                interval_return_events = 0
                interval_rejoin_events = 0
            if done:
                break

    best_path = outdir / f"seed{args.seed}_best.pt"
    if not best_path.exists():
        raise RuntimeError("No best checkpoint was saved")
    learner.load(str(best_path))

    val_dyn = evaluate(learner, validation_seeds, args.horizon, movement_only=False)
    val_5a = evaluate(learner, validation_seeds, args.horizon, movement_only=True)
    val_cmp = compare(val_dyn, val_5a)
    val_curriculum_cmp = compare_curriculum(val_dyn, initial_validation)

    # Held-out test is touched only now, after best-checkpoint selection.
    test_dyn = evaluate(learner, test_seeds, args.horizon, movement_only=False)
    test_5a = evaluate(learner, test_seeds, args.horizon, movement_only=True)
    test_cmp = compare(test_dyn, test_5a)

    # Initial no-battery checkpoint is evaluated on the same held-out battery
    # episodes only after selection, to quantify the value of battery fine-tuning.
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
    initial_test = evaluate(initial_learner, test_seeds, args.horizon, movement_only=False)
    test_curriculum_cmp = compare_curriculum(test_dyn, initial_test)

    lifecycle_exercised = bool(
        test_dyn["mean_return_events"] >= 1.0
        and test_dyn["mean_charge_arrivals"] >= 1.0
        and test_dyn["mean_rejoin_events"] >= 1.0
    )
    gate = {
        "battery_lifecycle_exercised": lifecycle_exercised,
        "positive_test_gain_vs_5A_ablation": bool(test_cmp["dynamic_minus_5A_pp"] > 0.0),
        "majority_test_seeds_beat_5A_ablation": bool(test_cmp["win_fraction"] > 0.5),
        "positive_test_gain_vs_initial_v7_checkpoint": bool(
            test_curriculum_cmp["finetuned_minus_initial_pp"] > 0.0
        ),
        "post_rejoin_observed": bool(test_dyn["post_rejoin_mean_utility"] is not None),
    }
    gate["pass_battery_extension_gate"] = bool(
        gate["battery_lifecycle_exercised"]
        and gate["post_rejoin_observed"]
        and gate["positive_test_gain_vs_initial_v7_checkpoint"]
    )

    summary = {
        "seed": int(args.seed),
        "steps": int(steps),
        "episodes": int(episodes),
        "init_checkpoint": str(init_checkpoint),
        "best_validation": float(best),
        "best_step": best_step,
        "initial_validation": {
            "dynamic": initial_validation,
            "movement_only_5A": initial_validation_5a,
        },
        "validation": {
            "dynamic": val_dyn,
            "movement_only_5A": val_5a,
            "comparison": val_cmp,
            "curriculum_vs_initial": val_curriculum_cmp,
        },
        "heldout_test": {
            "dynamic": test_dyn,
            "movement_only_5A": test_5a,
            "comparison": test_cmp,
            "initial_v7_checkpoint_dynamic": initial_test,
            "curriculum_vs_initial": test_curriculum_cmp,
        },
        "battery_gate": gate,
        "scenario": {
            "map": "canonical V4 topology",
            "backhaul_radius_cells": 6.0,
            "ue_mobility": "canonical stochastic correlated-group mobility",
            "battery": True,
            "battery_lifecycle": "automatic Active->Returning->Charging->Active",
            "proactive_return_action": False,
            "action_design": "0-4 Access movement; 5+j RelayTo(UAV j), self masked; assisted one-step movement",
            "reward": "mean user satisfaction only; unchanged from V7",
            "routing": "target-exclusive designated Relay; geometry-ranked conflict resolver",
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
                "validation_gain_vs_initial_pp": val_curriculum_cmp["finetuned_minus_initial_pp"],
                "test_gain_vs_5A_pp": test_cmp["dynamic_minus_5A_pp"],
                "test_gain_vs_initial_pp": test_curriculum_cmp["finetuned_minus_initial_pp"],
                "test_unavailable_utility": test_dyn["unavailable_mean_utility"],
                "test_post_rejoin_utility": test_dyn["post_rejoin_mean_utility"],
                "pass_battery_extension_gate": gate["pass_battery_extension_gate"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
