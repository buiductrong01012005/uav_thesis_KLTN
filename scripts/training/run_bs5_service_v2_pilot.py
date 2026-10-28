"""Fresh 24k Spatial-QMIX curriculum pilot; utility control, C-R/C-N, v2 only.

Checkpoint/resume at episode boundaries, including replay/optimizer/RNG.
Use --diagnostic for a separate 96-step smoke protocol; never call it 24k evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import sys
import time
import zipfile

for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(key, "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import scipy
import torch
from filelock import FileLock

from src.envs.bs5_curriculum_env import condition_for, task_signature
from src.envs.bs5_service_v2 import BS5ServiceV2Env, ServiceV2Config, FULL_CONDITION, VERSION
from src.rl.bs5_service_v2_qmix import ServiceV2QMIX
from src.rl.qmix import ReplayBuffer, Transition

STOP = False


def stop_handler(*_):
    global STOP
    STOP = True
    print(
        "Stop requested: finishing this episode before saving a consistent checkpoint.", flush=True
    )


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def source_lock():
    paths = sorted((ROOT / "src").rglob("*.py")) + [Path(__file__)]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    replace_file(temp, path)


def atomic_torch(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temp)
    replace_file(temp, path)


def load_protocol(args):
    p = json.loads((ROOT / "configs/bs5_service_v2_pilot.json").read_text())
    if args.diagnostic:
        p.update(
            total_steps=96,
            stage_ends=[12, 24, 48, 96],
            early_horizon=6,
            full_horizon=24,
            evaluation_steps=[0, 48, 96],
            validation_seeds=[640001],
            train_seed_base=630001,
            batch_size=8,
            replay_capacity=128,
            warmup=8,
            target_every=24,
            epsilon_decay_steps=72,
        )
    p["service"]["reward_mode"] = args.reward
    p["service"]["backhaul_total_mhz"] = args.backhaul
    p["service"]["report_delay_slots"] = args.delay
    p.update(
        arm=args.arm,
        diagnostic=args.diagnostic,
        device=args.device,
        environment_version=VERSION,
        source_sha256=source_lock(),
        torch_version=torch.__version__,
        runtime=dict(
            python=sys.version.split()[0], numpy=np.__version__, scipy=scipy.__version__, threads=1
        ),
    )
    return p


def replace_file(temp, destination):
    """Retry Windows sharing locks while preserving atomic file replacement."""
    for attempt in range(20):
        try:
            temp.replace(destination)
            return
        except PermissionError as error:
            if os.name != "nt" or error.winerror not in {5, 32, 33} or attempt == 19:
                raise
            time.sleep(min(0.05 * 2**attempt, 0.5))


def export_tables(output, saved):
    for name, rows in [("training", saved["training"]), ("validation", saved["validation"])]:
        if not rows:
            continue
        fields = list(dict.fromkeys(k for row in rows for k in row))
        temp = output / (name + ".csv.tmp")
        with temp.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        replace_file(temp, output / (name + ".csv"))
    review = output / "v2_pilot_review.zip"
    with zipfile.ZipFile(review.with_suffix(".tmp"), "w", zipfile.ZIP_DEFLATED) as archive:
        for name in [
            "protocol.json",
            "status.json",
            "training.csv",
            "validation.csv",
            "best.pt",
            "session_events.jsonl",
            "learning_curve.png",
            "learning_curve.pdf",
        ]:
            path = output / name
            if path.exists():
                archive.write(path, name)
    replace_file(review.with_suffix(".tmp"), review)


def plot_curves(output, saved):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    training = saved["training"]
    validation = saved["validation"]
    for axis, field, label in zip(
        axes, ["utility", "qos_fraction"], ["Utility", "Fraction of UEs at 4 Mbps"]
    ):
        if training:
            axis.plot(
                [r["step"] for r in training],
                [r[field] for r in training],
                alpha=0.5,
                label="Training episode mean",
            )
        steps = sorted({r["step"] for r in validation})
        if steps:
            axis.plot(
                steps,
                [np.mean([r[field] for r in validation if r["step"] == s]) for s in steps],
                marker="o",
                label="Fixed validation mean",
            )
        axis.set(xlabel="Training steps", ylabel=label, ylim=(0, 1))
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
    fig.suptitle(
        f"{saved['protocol']['arm']} / v2 / {'96-step diagnostic' if saved['protocol']['diagnostic'] else '24k pilot'}"
    )
    for extension in ["png", "pdf"]:
        fig.savefig(output / f"learning_curve.{extension}", dpi=160)
    plt.close(fig)


def model_payload(learner):
    return dict(
        agent=learner.agent.state_dict(),
        mixer=learner.mixer.state_dict(),
        target_agent=learner.target_agent.state_dict(),
        target_mixer=learner.target_mixer.state_dict(),
        optimizer=learner.optimizer.state_dict(),
    )


def save_boundary(output, learner, replay, rng, saved):
    saved.update(
        model=model_payload(learner),
        replay=replay,
        rng=dict(
            python=random.getstate(),
            numpy=np.random.get_state(),
            torch=torch.get_rng_state(),
            cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            actions=rng.bit_generator.state,
        ),
    )
    atomic_torch(output / "resume.pt", saved)
    atomic_json(
        output / "status.json",
        dict(
            status=saved["status"],
            committed_training_steps=saved["step"],
            training_updates=saved["updates"],
            completed_episodes=saved["episode"],
            best_validation_utility=saved["best"],
            best_step=saved["best_step"],
            diagnostic=saved["protocol"]["diagnostic"],
            environment_version=VERSION,
        ),
    )
    export_tables(output, saved)


def validate(learner, p, step):
    result = []
    for seed in p["validation_seeds"]:
        env = BS5ServiceV2Env(
            FULL_CONDITION,
            service_config=ServiceV2Config(**p["service"]),
            allow_relay=p["arm"] == "C-R",
            horizon=p["full_horizon"],
        )
        current = env.reset(seed=seed, assignment_seed=seed)
        rng = np.random.default_rng(seed)
        totals = np.zeros(4)
        for _ in range(p["full_horizon"]):
            actions = learner.act(current["obs"], current["action_mask"], 0, rng)
            current = env.step(actions)
            info = current["info"]
            totals += [
                info["utility"],
                info["qos_fraction"],
                info["ue_outage_fraction"],
                info["sum_served_mbps"],
            ]
        values = totals / p["full_horizon"]
        result.append(
            dict(
                step=step,
                seed=seed,
                utility=values[0],
                qos_fraction=values[1],
                ue_outage_fraction=values[2],
                throughput_mbps=values[3],
                slots=p["full_horizon"],
            )
        )
    return result


def _run_unlocked(args):
    global STOP
    STOP = False
    output = args.output.resolve()
    if output.exists() and not args.resume:
        raise ValueError("Output exists: use --resume with the same protocol, or a new --output")
    p = load_protocol(args)
    if p["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; select --device cpu")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    random.seed(p["training_seed"])
    np.random.seed(p["training_seed"])
    torch.manual_seed(p["training_seed"])
    learner = ServiceV2QMIX(lr=p["lr"], gamma=p["gamma"], device=p["device"])
    replay = ReplayBuffer(p["replay_capacity"], seed=p["training_seed"])
    rng = np.random.default_rng(p["training_seed"])
    if args.resume:
        saved = torch.load(output / "resume.pt", map_location=p["device"], weights_only=False)
        if saved["protocol"] != p:
            raise RuntimeError("Protocol/source/runtime changed; resume is refused")
        for name in ["agent", "mixer", "target_agent", "target_mixer"]:
            getattr(learner, name).load_state_dict(saved["model"][name])
        learner.optimizer.load_state_dict(saved["model"]["optimizer"])
        replay = saved["replay"]
        random.setstate(saved["rng"]["python"])
        np.random.set_state(saved["rng"]["numpy"])
        torch.set_rng_state(saved["rng"]["torch"].cpu())
        if saved["rng"]["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(saved["rng"]["cuda"])
        rng.bit_generator.state = saved["rng"]["actions"]
        print(f"Resume from committed episode boundary: {saved['step']} steps", flush=True)
    else:
        output.mkdir(parents=True)
        atomic_json(output / "protocol.json", p)
        saved = dict(
            protocol=p,
            step=0,
            episode=0,
            updates=0,
            best=None,
            best_step=None,
            validated_steps=[],
            training=[],
            validation=[],
            task_signature=None,
            status="prepared",
        )
        save_boundary(output, learner, replay, rng, saved)
    started = time.perf_counter()
    session_start_episode = saved["episode"]
    cpu = time.process_time()
    with (output / "session_events.jsonl").open("a") as f:
        f.write(
            json.dumps(
                dict(
                    event="start", committed_step=saved["step"], resume=args.resume, utc=time.time()
                )
            )
            + "\n"
        )
    while True:
        step = saved["step"]
        if step in p["evaluation_steps"] and step not in saved["validated_steps"]:
            rows = validate(learner, p, step)
            score = float(np.mean([x["utility"] for x in rows]))
            saved["validation"].extend(rows)
            saved["validated_steps"].append(step)
            if saved["best"] is None or score > saved["best"]:
                saved["best"], saved["best_step"] = score, step
                atomic_torch(
                    output / "best.pt",
                    dict(
                        agent=learner.agent.state_dict(),
                        mixer=learner.mixer.state_dict(),
                        selected_step=step,
                        validation_utility=score,
                        protocol_sha256=digest(p),
                        obs_dim=148,
                        state_dim=66,
                        environment_version=VERSION,
                    ),
                )
            save_boundary(output, learner, replay, rng, saved)
            print(f"validation step={step} utility={score:.6f}", flush=True)
        if step >= p["total_steps"]:
            saved["status"] = "complete"
            break
        if (
            STOP
            or (args.max_seconds > 0 and time.perf_counter() - started >= args.max_seconds)
            or (
                args.stop_after_episodes > 0
                and saved["episode"] - session_start_episode >= args.stop_after_episodes
            )
        ):
            saved["status"] = "paused_at_episode_boundary"
            break
        condition = condition_for(step, "C-R", p)
        signature = [list(x) for x in task_signature(condition)] + [
            ["S4_entry", condition["stage"] == "S4"]
        ]
        if signature != saved["task_signature"]:
            replay = ReplayBuffer(p["replay_capacity"], seed=p["training_seed"] + saved["episode"])
            saved["task_signature"] = signature
        horizon = p["full_horizon"] if condition["stage"] == "S4" else p["early_horizon"]
        if step + horizon > p["total_steps"]:
            raise RuntimeError("Episode would cross training budget")
        episode = saved["episode"]
        scenario = p["train_seed_base"] + episode % 40
        setup_seed = int(
            np.random.SeedSequence([p["training_seed"], episode, 0x5632]).generate_state(1)[0]
        )
        residual = float(np.random.default_rng(setup_seed).uniform(24, 36))
        env = BS5ServiceV2Env(
            condition,
            service_config=ServiceV2Config(**p["service"]),
            allow_relay=p["arm"] == "C-R",
            horizon=horizon,
            residual_battery=residual,
        )
        current = env.reset(seed=scenario, assignment_seed=setup_seed)
        sums = np.zeros(6)
        losses = []
        episode_start = time.perf_counter()
        for local_step in range(horizon):
            collection_step = step + local_step
            epsilon = max(
                p["epsilon_end"],
                1 - (1 - p["epsilon_end"]) * collection_step / p["epsilon_decay_steps"],
            )
            tic = time.perf_counter()
            actions = learner.act(current["obs"], current["action_mask"], epsilon, rng)
            decision_ms = (time.perf_counter() - tic) * 1000
            assert all(current["action_mask"][a, action] for a, action in enumerate(actions))
            nxt = env.step(actions)
            done = float(nxt["terminated"] or (nxt["truncated"] and condition["stage"] != "S4"))
            replay.add(
                Transition(
                    current["obs"],
                    current["state"],
                    np.asarray(actions),
                    float(nxt["reward"]),
                    nxt["obs"],
                    nxt["state"],
                    nxt["action_mask"],
                    done,
                )
            )
            global_step = collection_step + 1
            if (
                global_step >= p["warmup"]
                and len(replay) >= p["batch_size"]
                and global_step % p["update_every"] == 0
            ):
                losses.append(learner.train_batch(replay.sample(p["batch_size"])))
                saved["updates"] += 1
            if global_step % p["target_every"] == 0:
                learner.update_targets()
            info = nxt["info"]
            sums += [
                info["utility"],
                info["qos_fraction"],
                info["ue_outage_fraction"],
                nxt["reward"],
                decision_ms,
                info["radio_runtime_ms"],
            ]
            current = nxt
        means = sums / horizon
        saved["step"] = step + horizon
        saved["episode"] += 1
        saved["training"].append(
            dict(
                step=saved["step"],
                episode=episode,
                stage=condition["substage"],
                slots=horizon,
                utility=means[0],
                qos_fraction=means[1],
                ue_outage_fraction=means[2],
                reward=means[3],
                actor_ms=means[4],
                radio_ms=means[5],
                mean_loss=None if not losses else float(np.mean(losses)),
                epsilon=epsilon,
                updates=saved["updates"],
                scenario_seed=scenario,
                wall_seconds=time.perf_counter() - episode_start,
            )
        )
        saved["status"] = "running"
        save_boundary(output, learner, replay, rng, saved)
        print(
            f"{p['arm']} step={saved['step']}/{p['total_steps']} {condition['substage']} U={means[0]:.4f}",
            flush=True,
        )
    save_boundary(output, learner, replay, rng, saved)
    with (output / "session_events.jsonl").open("a") as f:
        f.write(
            json.dumps(
                dict(
                    event="finish",
                    committed_step=saved["step"],
                    status=saved["status"],
                    wall_seconds=time.perf_counter() - started,
                    cpu_seconds=time.process_time() - cpu,
                )
            )
            + "\n"
        )
    plot_curves(output, saved)
    export_tables(output, saved)
    return saved


def run(args):
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output) + ".lock", timeout=0):
        return _run_unlocked(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["C-R", "C-N"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--reward", choices=["utility", "qos_outage"], default="utility")
    parser.add_argument("--backhaul", type=float, choices=[40, 100], default=100)
    parser.add_argument("--delay", type=int, choices=[0, 1, 3], default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=0)
    parser.add_argument(
        "--stop-after-episodes",
        type=int,
        default=0,
        help="Explicit episode-boundary stop, e.g. for resume verification",
    )
    args = parser.parse_args()
    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)
    run(args)


if __name__ == "__main__":
    main()
