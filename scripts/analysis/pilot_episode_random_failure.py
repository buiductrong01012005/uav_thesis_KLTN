"""Short no-curriculum fixed-BS5/random-BS pipeline pilot, not final research evaluation."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.envs.episode_random_failure_env import (
    EpisodeFailureCartesianConfig,
    EpisodeFailureCartesianEnv,
)
from src.rl.episode_failure_spatial_qmix import EpisodeFailureSpatialQMIX
from src.rl.qmix import ReplayBuffer, Transition
from scripts.analysis.audit_episode_random_failure import plain, write_json


def make_env(mode):
    return EpisodeFailureCartesianEnv(
        EpisodeFailureCartesianConfig(
            failed_bs_idx=5,
            failure_mode=mode,
            horizon=150,
            record_service_snapshot=True,
        )
    )


def evaluate(learner, mode, output, step, trace):
    # Diagnostic validation only: one held-out scenario seed per BS. Not a
    # statistically adequate final test set. Never used to retune this pilot.
    env, rng, rows = make_env(mode), np.random.default_rng(991), []
    for bs in range(6):
        out = env.reset(seed=9401, options={"failed_bs_idx": bs})
        rewards = []
        for t in range(env.cfg.horizon):
            actions = learner.act(out["obs"], out["action_mask"], 0.0, rng)
            out = env.step(actions)
            trace.write(
                json.dumps(
                    dict(
                        phase="validation",
                        train_step=step,
                        scenario_seed=9401,
                        bs=bs,
                        t=t,
                        actions=actions,
                        reward=out["reward"],
                        service=env.last_service_snapshot,
                    ),
                    default=plain,
                    separators=(",", ":"),
                )
                + "\n"
            )
            rewards.append(out["reward"])
        rows.append(dict(bs=bs, utility=float(np.mean(rewards)), slots=len(rewards)))
    payload = dict(
        train_step=step, rows=rows, macro_mean_utility=float(np.mean([x["utility"] for x in rows]))
    )
    write_json(output / f"validation_{step:06d}.json", payload)
    print(json.dumps(dict(event="validation", mode=mode, **payload)), flush=True)
    return payload


def run_one(mode, seed, output, steps=6000):
    outdir = Path(output) / f"{mode}_seed{seed}"
    outdir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    random.seed(seed)
    rng = np.random.default_rng(seed)
    env = make_env(mode)
    learner = EpisodeFailureSpatialQMIX(env, lr=3e-4)
    replay = ReplayBuffer(12000, seed)
    protocol = dict(
        schema_version=1,
        mode=mode,
        seed=seed,
        steps=steps,
        config=asdict(env.cfg),
        train_scenario_seed_start=9200,
        validation_seeds=[9401],
        epsilon="1.0 -> 0.1 linear over pilot steps",
        warmup=300,
        update_every=4,
        target_every=500,
        batch_size=64,
        checkpoint_selection="none; use last, fixed in advance",
        scope="no curriculum, full dynamics; pipeline feasibility only",
        trace_coverage="all validation slots; first train episode encountered for each BS",
    )
    write_json(outdir / "protocol.json", protocol)
    episode, step, seen, train_bs = 0, 0, set(), Counter()
    evaluations, losses = [], []
    started = time.monotonic()
    with (
        (outdir / "episodes.jsonl").open("w") as ep_log,
        (outdir / "updates.jsonl").open("w") as update_log,
        gzip.open(outdir / "service_traces.jsonl.gz", "wt") as trace,
    ):
        evaluations.append(evaluate(learner, mode, outdir, 0, trace))
        while step < steps:
            scenario_seed = 9200 + episode
            out = env.reset(seed=scenario_seed)
            bs = env.cfg.failed_bs_idx
            record_episode = bs not in seen
            seen.add(bs)
            values, actions_count = [], Counter()
            for t in range(env.cfg.horizon):
                epsilon = max(0.1, 1.0 - 0.9 * step / max(steps, 1))
                actions = learner.act(out["obs"], out["action_mask"], epsilon, rng)
                nxt = env.step(actions)
                done = float(nxt["terminated"] or nxt["truncated"])
                replay.add(
                    Transition(
                        out["obs"],
                        out["state"],
                        np.array(actions),
                        nxt["reward"],
                        nxt["obs"],
                        nxt["state"],
                        nxt["action_mask"],
                        done,
                    )
                )
                step += 1
                train_bs[bs] += 1
                values.append(nxt["reward"])
                actions_count.update(actions)
                if record_episode:
                    trace.write(
                        json.dumps(
                            dict(
                                phase="train_sample",
                                train_step=step,
                                episode=episode,
                                scenario_seed=scenario_seed,
                                bs=bs,
                                t=t,
                                actions=actions,
                                epsilon=epsilon,
                                reward=nxt["reward"],
                                service=env.last_service_snapshot,
                            ),
                            default=plain,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                if step >= 300 and step % 4 == 0:
                    loss = learner.train_batch(replay.sample(64))
                    assert np.isfinite(loss)
                    params = list(learner.agent.parameters()) + list(learner.mixer.parameters())
                    assert all(torch.isfinite(p).all() for p in params)
                    grad = float(
                        torch.sqrt(sum(p.grad.square().sum() for p in params if p.grad is not None))
                    )
                    update_log.write(
                        json.dumps(
                            dict(
                                step=step,
                                loss=loss,
                                grad_norm_after_clip=grad,
                                buffer_size=len(replay),
                            )
                        )
                        + "\n"
                    )
                    losses.append(loss)
                if step % 500 == 0:
                    learner.update_targets()
                out = nxt
                if done or step >= steps:
                    break
            ep_log.write(
                json.dumps(
                    dict(
                        episode=episode,
                        end_step=step,
                        seed=scenario_seed,
                        bs=bs,
                        slots=len(values),
                        mean_utility=float(np.mean(values)),
                        truncated=bool(done),
                        actions=dict(actions_count),
                    )
                )
                + "\n"
            )
            ep_log.flush()
            update_log.flush()
            episode += 1
            if step == steps // 2 or step == steps:
                evaluations.append(evaluate(learner, mode, outdir, step, trace))
        checkpoint = outdir / "last.pt"
        learner.save(str(checkpoint))
        # Independent model instance, exact Q values after reload; no new
        # evaluation is needed to produce the analysis tables.
        fresh = EpisodeFailureSpatialQMIX(env)
        fresh.load(str(checkpoint))
        obs = torch.as_tensor(out["obs"])
        with torch.no_grad():
            torch.testing.assert_close(learner.agent(obs), fresh.agent(obs), rtol=0, atol=0)
    summary = dict(
        mode=mode,
        seed=seed,
        steps=step,
        episodes=episode,
        update_count=len(losses),
        finite_loss=bool(np.isfinite(losses).all()),
        last_loss=losses[-1],
        train_bs_slots=dict(train_bs),
        evaluations=evaluations,
        checkpoint_roundtrip_exact=True,
        elapsed_seconds=time.monotonic() - started,
        final_model_selection="last; no best-of-test selection",
        claim="pipeline verified only, insufficient budget/seeds/scenarios for learned relay advantage",
    )
    write_json(outdir / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--seeds", type=int, nargs="+", default=[251, 252])
    p.add_argument("--modes", nargs="+", choices=["fixed", "random"], default=["fixed", "random"])
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    write_json(
        a.output / "protocol.json",
        dict(
            steps_per_run=a.steps,
            seeds=a.seeds,
            modes=a.modes,
            total_steps=a.steps * len(a.seeds) * len(a.modes),
        ),
    )
    rows = []
    for mode in a.modes:
        for seed in a.seeds:
            rows.append(run_one(mode, seed, a.output, a.steps))
    write_json(a.output / "summary.json", rows)
