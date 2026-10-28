"""Fresh bounded TCF Spatial-QMIX; full episode-boundary resume, CPU by default."""

from __future__ import annotations
import argparse
from copy import deepcopy
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import sys
import time

for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(name, "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import scipy
import torch
from filelock import FileLock
from src.envs.bs5_curriculum_env import condition_for, task_signature
from src.envs.bs5_service_v2 import FULL_CONDITION, ServiceV2Config
from src.envs.bs5_tcf_env import BS5TCFEnv, TCFWeights, VERSION, jain, rate_metrics
from src.rl.bs5_service_v2_qmix import ServiceV2QMIX
from src.rl.qmix import ReplayBuffer, Transition
from scripts.training.run_bs5_service_v2_pilot import (
    atomic_json,
    atomic_torch,
    digest,
    model_payload,
    replace_file,
)

CONFIG = ROOT / "configs/bs5_tcf120k.json"
STOP = False
PRIMARY_METRICS = [
    "throughput_mbps",
    "throughput_norm",
    "service_coverage_fraction",
    "service_coverage_ues",
    "positive_fraction",
    "positive_ues",
    "qos_fraction",
    "qos_ues",
    "ue_outage_fraction",
    "jain_raw",
    "jain_service_qualified",
    "tcf_score",
    "reward_throughput",
    "reward_coverage",
    "reward_fairness",
]
SELECTIONS = {
    "tcf_score": "best.pt",
    "throughput_mbps": "best_throughput.pt",
    "service_coverage_fraction": "best_coverage.pt",
    "jain_raw": "best_jain_raw.pt",
    "qos_fraction": "best_qos.pt",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_lock():
    files = list((ROOT / "src").rglob("*.py")) + [
        Path(__file__),
        CONFIG,
        ROOT / "scripts/training/run_bs5_service_v2_pilot.py",
        ROOT / "scripts/evaluation/eval_bs5_tcf_checkpoint.py",
    ]
    return {str(f.relative_to(ROOT)): sha(f) for f in sorted(files)}


def stop_handler(*_):
    global STOP
    STOP = True
    print("Finishing this episode, then committing optimizer/replay/RNG.", flush=True)


def load_protocol(args):
    p = json.loads(CONFIG.read_text())
    presets = p.pop("presets")
    if args.preset not in presets:
        raise ValueError("Unknown preset: " + args.preset)
    weights = presets[args.preset]
    if args.weights is not None:
        weights = dict(zip(["throughput", "coverage", "fairness"], args.weights))
    TCFWeights(**weights)
    p.update(
        arm=args.arm,
        preset=args.preset,
        reward_weights=weights,
        coverage_floor_mbps=args.coverage_floor,
        diagnostic=args.diagnostic,
        device=args.device,
        total_steps=args.total_steps,
        reward_version=VERSION,
        source_sha256=source_lock(),
        torch_version=str(torch.__version__),
        runtime=dict(
            python=sys.version.split()[0], numpy=np.__version__, scipy=scipy.__version__, threads=1
        ),
    )
    p["stage_ends"][-1] = p["total_steps"]
    p["evaluation_steps"] = [0, 12000, 18000] + list(
        range(24000, p["total_steps"] + 1, p["evaluation_every"])
    )
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
    if not args.diagnostic and (
        p["total_steps"] < 24000 or (p["total_steps"] - 12000) % p["full_horizon"]
    ):
        raise ValueError("Budget must be >=24k and contain whole full-task episodes after 12k")
    p["evaluation_steps"] = sorted(set(p["evaluation_steps"] + [p["total_steps"]]))
    rate_metrics([0.0], coverage_floor_mbps=p["coverage_floor_mbps"], weights=TCFWeights(**weights))
    return p


def make_env(p, *, condition=None, horizon=None, residual_battery=30.0):
    return BS5TCFEnv(
        condition or FULL_CONDITION,
        service_config=ServiceV2Config(**p["service"]),
        reward_weights=TCFWeights(**p["reward_weights"]),
        coverage_floor_mbps=p["coverage_floor_mbps"],
        allow_relay=p["arm"] == "C-R",
        horizon=horizon or p["full_horizon"],
        residual_battery=residual_battery,
    )


def observe_step(env, info):
    snap, net = env.last_service_snapshot, env.last_v2_network
    active = np.flatnonzero(snap["status"] == 0)
    access = [a for a in active if snap["roles"][a] == 0]
    relays = [a for a in active if snap["roles"][a] == 1]
    covered = np.zeros(env.cfg.num_ues, dtype=bool)
    for aid in access:
        covered |= (
            np.linalg.norm(snap["ue_positions"] - snap["uav_positions"][aid], axis=1)
            <= env.cfg.access_radius_cells + 1e-9
        )
    paths = env.last_v2_radio_detail["paths"]
    forwarding = {
        r
        for a, r in net["routes"].items()
        if r is not None and net["backhaul_caps_mbps"].get(a, 0) > 1e-9
    }
    assert net["schedule_time_sum"] <= 1 + 1e-6
    assert net["radio_flow_violation_mbps"] <= 1e-6
    return {
        **{k: float(info[k]) for k in PRIMARY_METRICS},
        "utility": float(info["utility"]),
        "covered_ues": float(covered.sum()),
        "associated_ues": float(sum(map(len, net["association"].values()))),
        "reachable_ues": float(
            sum(len(net["association"][a]) for a, path in paths.items() if path)
        ),
        "active_access": float(len(access)),
        "active_relay": float(len(relays)),
        "forwarding_relay": float(len(forwarding)),
        "access_offer_mbps": float(net["access_raw_ue_rates"].sum()),
        "reachable_offer_mbps": float(
            sum(net["access_offer_mbps"][a] for a, path in paths.items() if path)
        ),
        "schedule_airtime": float(net["schedule_time_sum"]),
        "zero_service_slot_fraction": float(info["positive_ues"] == 0),
        "returning_uavs": float(np.sum(snap["status"] == 1)),
        "charging_uavs": float(np.sum(snap["status"] == 2)),
        "stranded_slot_fraction": float(bool(env.last_return_control["stranded_ids_post"])),
        "flow_violation_mbps": float(net["radio_flow_violation_mbps"]),
    }


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    replace_file(temp, path)


def evaluate(learner, p, step, output):
    rows = []
    destination = output / "validation_detail"
    destination.mkdir(exist_ok=True)
    for seed in p["validation_seeds"]:
        env = make_env(p)
        current = env.reset(seed=seed, assignment_seed=seed)
        rng = np.random.default_rng(seed)
        samples, arrays = (
            [],
            {k: [] for k in ["ue_rates", "positions", "roles", "status", "battery", "actions"]},
        )
        for slot in range(p["full_horizon"]):
            actions = learner.act(current["obs"], current["action_mask"], 0.0, rng)
            current = env.step(actions)
            samples.append(dict(slot=slot + 1, **observe_step(env, current["info"])))
            snap = env.last_service_snapshot
            arrays["ue_rates"].append(env.last_v2_network["ue_rate_mbps"].copy())
            for key, source in [
                ("positions", "uav_positions"),
                ("roles", "roles"),
                ("status", "status"),
                ("battery", "battery"),
            ]:
                arrays[key].append(snap[source].copy())
            arrays["actions"].append(actions)
        rates = np.asarray(arrays["ue_rates"])
        average = rates.mean(axis=0)
        qualified_average = np.where(
            (average > 1e-9) & (average >= p["coverage_floor_mbps"] - 1e-9), average, 0.0
        )
        row = dict(
            step=step,
            seed=seed,
            slots=p["full_horizon"],
            **{k: float(np.mean([s[k] for s in samples])) for k in samples[0] if k != "slot"},
            jain_time_average=jain(average),
            jain_qualified_time_average=jain(qualified_average),
            mean_ue_rate_mbps=float(average.mean()),
            p05_ue_mean_rate_mbps=float(np.quantile(average, 0.05)),
            always_outage_ues=int(np.sum(np.all(rates <= 1e-9, axis=0))),
            max_flow_violation_mbps=max(s["flow_violation_mbps"] for s in samples),
        )
        write_csv(destination / f"step{step}_seed{seed}_slots.csv", samples)
        path = destination / f"step{step}_seed{seed}_ue_rates.npz"
        with path.with_suffix(".tmp").open("wb") as stream:
            np.savez_compressed(
                stream, **{k: np.asarray(v) for k, v in arrays.items()}, ue_mean_rates=average
            )
        replace_file(path.with_suffix(".tmp"), path)
        rows.append(row)
    return rows


def checkpoint_payload(learner, p, step, metric, score):
    return dict(
        agent=learner.agent.state_dict(),
        mixer=learner.mixer.state_dict(),
        selected_step=step,
        selection_metric=metric,
        validation_score=score,
        protocol_sha256=digest(p),
        reward_version=VERSION,
        obs_dim=148,
        state_dim=66,
    )


def record_selection(output, learner, p, saved, rows):
    for metric, filename in SELECTIONS.items():
        score = float(np.mean([r[metric] for r in rows]))
        previous = saved["best_by_metric"].get(metric)
        if previous is None or score > previous["score"]:
            saved["best_by_metric"][metric] = dict(score=score, step=saved["step"], file=filename)
            atomic_torch(
                output / filename, checkpoint_payload(learner, p, saved["step"], metric, score)
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
            actions=deepcopy(rng.bit_generator.state),
        ),
    )
    atomic_torch(output / "resume.pt", saved)
    atomic_json(
        output / "status.json",
        dict(
            status=saved["status"],
            committed_training_steps=saved["step"],
            target_steps=saved["protocol"]["total_steps"],
            training_updates=saved["updates"],
            completed_episodes=saved["episode"],
            best_by_metric=saved["best_by_metric"],
            reward_weights=saved["protocol"]["reward_weights"],
            coverage_floor_mbps=saved["protocol"]["coverage_floor_mbps"],
            diagnostic=saved["protocol"]["diagnostic"],
            reward_version=VERSION,
        ),
    )
    write_csv(output / "training.csv", saved["training"])
    write_csv(output / "validation.csv", saved["validation"])


def plot_curves(output, saved):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fields = [
        ("tcf_score", "Normalized TCF score"),
        ("throughput_mbps", "Throughput (Mbps)"),
        ("service_coverage_fraction", "UE fraction >= service threshold"),
        ("qos_fraction", "UE fraction >= 4 Mbps"),
        ("jain_raw", "Raw Jain, all UEs"),
        ("jain_service_qualified", "Service-qualified Jain, all UEs"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    steps = sorted({r["step"] for r in saved["validation"]})
    for axis, (field, label) in zip(axes.flat, fields):
        axis.plot(
            [r["step"] for r in saved["training"]],
            [r[field] for r in saved["training"]],
            alpha=0.35,
            label="Training episode",
        )
        axis.plot(
            steps,
            [np.mean([r[field] for r in saved["validation"] if r["step"] == s]) for s in steps],
            marker="o",
            label="Validation mean",
        )
        axis.set(xlabel="Training steps", ylabel=label)
        if field != "throughput_mbps":
            axis.set_ylim(0, 1)
        axis.grid(alpha=0.2)
    axes.flat[0].legend(fontsize=8)
    p = saved["protocol"]
    fig.suptitle(
        f"{p['arm']} / {p['preset']} / {'96-step functional check' if p['diagnostic'] else str(p['total_steps']) + ' fresh steps'}"
    )
    fig.savefig(output / "learning_curve.png", dpi=160)
    plt.close(fig)


def run_unlocked(args):
    global STOP
    STOP = False
    output = args.output.resolve()
    if output.exists() and not args.resume:
        raise ValueError("Output exists; choose --resume or a fresh output path")
    p = load_protocol(args)
    if p["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
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
            raise RuntimeError(
                "TCF protocol/source/weights/threshold/runtime changed; resume refused"
            )
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
        for metric, record in saved["best_by_metric"].items():
            best = torch.load(output / record["file"], map_location="cpu", weights_only=True)
            if (
                best["protocol_sha256"] != digest(p)
                or best["selected_step"] != record["step"]
                or best["validation_score"] != record["score"]
            ):
                raise RuntimeError("Best checkpoint lineage mismatch: " + metric)
        print(f"Resume {p['arm']} from {saved['step']} committed steps", flush=True)
    else:
        output.mkdir(parents=True)
        atomic_json(output / "protocol.json", p)
        saved = dict(
            protocol=p,
            step=0,
            episode=0,
            updates=0,
            best_by_metric={},
            validated_steps=[],
            training=[],
            validation=[],
            task_signature=None,
            status="prepared",
        )
        save_boundary(output, learner, replay, rng, saved)
    started = time.perf_counter()
    session_episode = saved["episode"]
    with (output / "session_events.jsonl").open("a") as stream:
        stream.write(
            json.dumps(dict(event="start", step=saved["step"], resume=args.resume, utc=time.time()))
            + "\n"
        )
    while True:
        step = saved["step"]
        if step in p["evaluation_steps"] and step not in saved["validated_steps"]:
            rows = evaluate(learner, p, step, output)
            saved["validation"].extend(rows)
            saved["validated_steps"].append(step)
            record_selection(output, learner, p, saved, rows)
            save_boundary(output, learner, replay, rng, saved)
            print(
                f"validation {p['arm']} step={step} TCF={np.mean([r['tcf_score'] for r in rows]):.5f}",
                flush=True,
            )
        if step >= p["total_steps"]:
            saved["status"] = "complete"
            break
        if (
            STOP
            or (args.max_seconds > 0 and time.perf_counter() - started >= args.max_seconds)
            or (
                args.stop_after_episodes > 0
                and saved["episode"] - session_episode >= args.stop_after_episodes
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
            raise RuntimeError("Episode would cross budget")
        episode = saved["episode"]
        scenario = p["train_seed_base"] + episode % 40
        setup_seed = int(
            np.random.SeedSequence([p["training_seed"], episode, 0x5632]).generate_state(1)[0]
        )
        residual = float(np.random.default_rng(setup_seed).uniform(24, 36))
        env = make_env(p, condition=condition, horizon=horizon, residual_battery=residual)
        current = env.reset(seed=scenario, assignment_seed=setup_seed)
        samples, losses = [], []
        episode_start = time.perf_counter()
        for local_step in range(horizon):
            collection_step = step + local_step
            epsilon = max(
                p["epsilon_end"],
                1 - (1 - p["epsilon_end"]) * collection_step / p["epsilon_decay_steps"],
            )
            tic = time.perf_counter()
            actions = learner.act(current["obs"], current["action_mask"], epsilon, rng)
            actor_ms = (time.perf_counter() - tic) * 1000
            assert all(current["action_mask"][a, x] for a, x in enumerate(actions))
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
            samples.append(
                {
                    **observe_step(env, nxt["info"]),
                    "actor_ms": actor_ms,
                    "radio_ms": float(nxt["info"]["radio_runtime_ms"]),
                }
            )
            current = nxt
        saved["step"] = step + horizon
        saved["episode"] += 1
        saved["training"].append(
            dict(
                step=saved["step"],
                episode=episode,
                stage=condition["substage"],
                slots=horizon,
                **{k: float(np.mean([r[k] for r in samples])) for k in samples[0]},
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
            f"{p['arm']} {p['preset']} {saved['step']}/{p['total_steps']} {condition['substage']} TCF={saved['training'][-1]['tcf_score']:.4f}",
            flush=True,
        )
    save_boundary(output, learner, replay, rng, saved)
    plot_curves(output, saved)
    with (output / "session_events.jsonl").open("a") as stream:
        stream.write(
            json.dumps(
                dict(
                    event="finish",
                    step=saved["step"],
                    status=saved["status"],
                    wall_seconds=time.perf_counter() - started,
                )
            )
            + "\n"
        )
    return saved


def run(args):
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(args.output.resolve()) + ".lock", timeout=0):
        return run_unlocked(args)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--arm", choices=["C-R", "C-N"], required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--preset", default="balanced")
    result.add_argument(
        "--weights", type=float, nargs=3, metavar=("THROUGHPUT", "COVERAGE", "FAIRNESS")
    )
    result.add_argument("--coverage-floor", type=float, default=1.0)
    result.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    result.add_argument("--total-steps", type=int, default=120000)
    result.add_argument("--resume", action="store_true")
    result.add_argument("--diagnostic", action="store_true")
    result.add_argument("--max-seconds", type=float, default=0.0)
    result.add_argument("--stop-after-episodes", type=int, default=0)
    return result


if __name__ == "__main__":
    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)
    run(parser().parse_args())
