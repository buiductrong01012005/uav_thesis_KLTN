"""Four fresh CPU CNN/QMIX arms at399600steps,using matched instantaneous fairness and physical energy."""

from __future__ import annotations
from copy import deepcopy
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(name, "1")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import scipy
import torch
from filelock import FileLock
from src.envs.curriculum import condition_for, task_signature
from src.envs.reward import TCFWeights, jain
from src.envs.environment import (
    FairConfig,
    OBS_DIM,
    STATE_DIM,
    VERSION as ENV_VERSION,
    REWARD_VERSION as VERSION,
    linear_learning_rate,
)
from src.diagnostics.bs5_episode_service_metrics import episode_service_metrics
from src.rl.fourarm_qmix import make_learner
from src.diagnostics.bs5_fourarm400k_metrics import (
    observe_step,
    validation_metrics,
    training_metrics,
)
from src.rl.qmix import ReplayBuffer, Transition
from scripts.training.checkpoint_io import (
    atomic_json,
    atomic_torch,
    digest,
    model_payload,
)

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
    "service_score",
    "reward_throughput",
    "reward_coverage",
    "reward_fairness",
    "fairness_used",
    "jain_ema",
    "jain_instant_qualified",
]
SELECTIONS = {
    "reward_total": "best.pt",
    "service_score": "best_service.pt",
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
        ROOT / "scripts/train.py",
        ROOT / "scripts/source_integrity.py",
        ROOT / "scripts/evaluation/actor.py",
        ROOT / "scripts/training/checkpoint_io.py",
        ROOT / "scripts/verification/export_bs5_fourarm400k.py",
        ROOT / "scripts/analysis/plot_bs5_fourarm400k.py",
    ]
    return {f.relative_to(ROOT).as_posix(): sha(f) for f in sorted(files)}


def stop_handler(*_):
    global STOP
    STOP = True
    print("Finishing this episode, then committing optimizer/replay/RNG.", flush=True)


def load_protocol(args):
    from src.experiment import environment_protocol
    from src.envs.config import ARM_ACTIONS

    config = deepcopy(args.resolved_config)
    t = config["training"]
    env = config["environment"]
    p = dict(t, **environment_protocol(env))
    p.update(
        version=ENV_VERSION,
        training_env_version=ENV_VERSION,
        reward_version=VERSION,
        preset="service_fair",
        obs_dim=OBS_DIM,
        state_dim=STATE_DIM,
        policy_actions=ARM_ACTIONS[t["arm"]],
        full_horizon=env["simulation"]["horizon"],
        reserved_test_seeds=config["evaluation"]["test_seeds"],
        resolved_config=config,
        resolved_config_sha256=digest(config),
        architecture=deepcopy(t["architecture"]),
        source_sha256=source_lock(),
        torch_version=str(torch.__version__),
        runtime=dict(
            python=sys.version.split()[0],
            numpy=np.__version__,
            scipy=scipy.__version__,
            threads=1,
            cuda=torch.version.cuda,
            cudnn=torch.backends.cudnn.version(),
            torch_build_sha256=hashlib.sha256(torch.__config__.show().encode()).hexdigest(),
        ),
    )
    if p["device"] != "cpu":
        raise ValueError("This reproducible training runner uses CPU")
    TCFWeights(**p["reward_weights"])
    FairConfig(**p["fair_config"])
    common = {
        k: v
        for k, v in p.items()
        if k not in {"arm", "policy_actions", "runtime", "torch_version", "comparison_fingerprint"}
    }
    p["comparison_fingerprint"] = digest(common)
    return p


def make_env(p, *, condition=None, horizon=None, residual_battery=None):
    from src.experiment import make_environment

    return make_environment(
        p["resolved_config"]["environment"],
        arm=p["arm"],
        condition=condition,
        horizon=horizon,
        residual_battery=residual_battery,
    )


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def evaluate(learner, p, step, output):
    rows = []
    destination = output / "validation_detail"
    destination.mkdir(exist_ok=True)
    for seed in p["validation_seeds"]:
        env = make_env(p)
        current = env.reset(seed=seed, assignment_seed=seed)
        initial_battery = float(env.battery.sum())
        cycles, pending = [], {}
        charge_rejoins = 0
        energy_audits = []
        collision_audits = []
        rng = np.random.default_rng(seed)
        samples, arrays = (
            [],
            {
                k: []
                for k in [
                    "ue_rates",
                    "ema_rates",
                    "history_map",
                    "ue_positions",
                    "positions",
                    "roles",
                    "status",
                    "battery",
                    "actions",
                    "proxy_actions",
                    "post_status",
                    "post_battery",
                    "path_wireless_hops",
                ]
            },
        )
        for slot in range(p["full_horizon"]):
            before = env.status.copy()
            actions = learner.act(current["obs"], current["action_mask"], 0.0, rng)
            current = env.step(actions)
            samples.append(dict(slot=slot + 1, **observe_step(env, current["info"])))
            energy_audits.append(deepcopy(env.last_energy_audit))
            collision_audits.append(deepcopy(env.last_collision_audit))
            snap = env.last_service_snapshot
            arrays["ue_rates"].append(env.last_v2_network["ue_rate_mbps"].copy())
            arrays["ema_rates"].append(env.ema_rates.copy())
            arrays["history_map"].append(env.history_reports.vector(env.t).copy())
            for key, source in [
                ("positions", "uav_positions"),
                ("roles", "roles"),
                ("status", "status"),
                ("battery", "battery"),
            ]:
                arrays[key].append(snap[source].copy())
            arrays["actions"].append(actions)
            arrays["proxy_actions"].append(env._slot_actions)
            arrays["ue_positions"].append(snap["ue_positions"].copy())
            arrays["post_status"].append(env.status.copy())
            arrays["post_battery"].append(env.battery.copy())
            arrays["path_wireless_hops"].append(
                [env.last_v2_network["path_wireless_hops"].get(a, 0) for a in range(5)]
            )
            net = env.last_v2_network
            served = {
                a: bool(np.any(net["ue_rate_mbps"][users] > 1e-6))
                for a, users in net["association"].items()
            }
            for relay, flow in net["forwarded_mbps_by_uav"].items():
                if flow > 1e-6:
                    served[relay] = True
            for aid in range(5):
                if before[aid] == 2 and env.status[aid] == 0:
                    charge_rejoins += 1
                    if aid in pending:
                        pending[aid]["rejoin_slot"] = env.t
                if (
                    aid in pending
                    and pending[aid]["rejoin_slot"] is not None
                    and served.get(aid, False)
                ):
                    pending[aid].update(served_again_slot=env.t, outcome="served_again")
                    cycles.append(pending.pop(aid))
                if before[aid] == 0 and env.status[aid] == 1:
                    if aid in pending:
                        pending[aid]["outcome"] = "next_return_without_service"
                        cycles.append(pending.pop(aid))
                    pending[aid] = dict(agent=aid, return_slot=env.t, rejoin_slot=None)
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
            **episode_service_metrics(rates, p["coverage_floor_mbps"]),
            jain_qualified_time_average=jain(qualified_average),
            mean_ue_rate_mbps=float(average.mean()),
            p05_ue_mean_rate_mbps=float(np.quantile(average, 0.05)),
            always_outage_ues=int(np.sum(np.all(rates <= 1e-9, axis=0))),
            max_flow_violation_mbps=max(s["flow_violation_mbps"] for s in samples),
            episode_max_wireless_hops_used=max(s["max_wireless_hops_used"] for s in samples),
        )
        used = sum(s["depleted_points"] for s in samples)
        charged = sum(s["charged_points"] for s in samples)
        final_battery = float(env.battery.sum())
        balance = initial_battery + charged - used - final_battery
        assert abs(balance) < 1e-3
        delivered_gbit = float(rates.sum() * env.cfg.slot_s / 1000)
        row.update(
            energy_depleted_points=used,
            energy_charged_points=charged,
            initial_battery_points=initial_battery,
            final_battery_points=final_battery,
            battery_balance_residual_points=balance,
            battery_points_per_gbit=used / delivered_gbit if delivered_gbit > 1e-12 else None,
            charge_rejoin_events=charge_rejoins,
            served_again_cycles=sum(c["outcome"] == "served_again" for c in cycles),
            unfinished_cycles=len(pending),
        )
        atomic_json(
            destination / f"step{step}_seed{seed}_cycles.json",
            dict(completed=cycles, pending=list(pending.values())),
        )
        write_csv(destination / f"step{step}_seed{seed}_slots.csv", samples)
        path = destination / f"step{step}_seed{seed}_ue_rates.npz"
        with path.with_suffix(".tmp").open("wb") as stream:
            np.savez_compressed(
                stream, **{k: np.asarray(v) for k, v in arrays.items()}, ue_mean_rates=average
            )
        path.with_suffix(".tmp").replace(path)
        row.update(
            energy_consumed_wh=sum(x["consumed_wh"] for x in samples),
            swap_events=sum(x["swaps_completed"] for x in samples),
            energy_shortfall_total_j=sum(x["energy_shortfall_j"] for x in samples),
        )
        row["mbit_per_joule"] = (
            delivered_gbit * 1000 / (row["energy_consumed_wh"] * 3600)
            if row["energy_consumed_wh"] > 0
            else None
        )
        row["joules_per_mbit"] = (
            row["energy_consumed_wh"] * 3600 / (delivered_gbit * 1000)
            if delivered_gbit > 0
            else None
        )
        row.update(
            {
                k.replace("_j", "_wh"): sum(x[k] for x in samples) / 3600
                for k in ["flight_j", "circuit_j", "access_pa_j", "backhaul_pa_j"]
            }
        )
        atomic_json(destination / f"step{step}_seed{seed}_energy_events.json", env.energy_events)
        with (destination / f"step{step}_seed{seed}_energy.npz").open("wb") as stream:
            np.savez_compressed(
                stream, **{k: np.asarray([x[k] for x in energy_audits]) for k in energy_audits[0]}
            )
        row.update(
            active_blocked_total=sum(x["active_blocked_uavs"] for x in samples),
            returning_blocked_total=sum(x["returning_blocked_uavs"] for x in samples),
            policy_move_requests_total=sum(x["policy_move_requests"] for x in samples),
        )
        row["blocked_per_policy_move"] = (
            row["active_blocked_total"] / row["policy_move_requests_total"]
            if row["policy_move_requests_total"] > 0
            else None
        )
        atomic_json(
            destination / f"step{step}_seed{seed}_collision_events.json", env.collision_events
        )
        with (destination / f"step{step}_seed{seed}_collision.npz").open("wb") as stream:
            np.savez_compressed(
                stream,
                **{k: np.asarray([x[k] for x in collision_audits]) for k in collision_audits[0]},
            )
        row.update(validation_metrics(rates, samples, energy_audits, env, collision_audits))
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
        training_env_version=ENV_VERSION,
        obs_dim=OBS_DIM,
        state_dim=STATE_DIM,
        n_actions=p["policy_actions"],
        arm=p["arm"],
        preset=p["preset"],
        fair_config=p["fair_config"],
        resolved_config_sha256=p["resolved_config_sha256"],
        architecture=p["architecture"],
    )


def record_selection(output, learner, p, saved, rows):
    for metric, filename in SELECTIONS.items():
        score = float(np.mean([r[metric] for r in rows]))
        previous = saved["best_by_metric"].get(metric)
        if previous is None or score > previous["score"]:
            # Immutable selection path: an interrupted newer validation cannot
            # overwrite the selected model referenced by the committed resume.
            stable = Path("selected") / f"{Path(filename).stem}_step{saved['step']}.pt"
            (output / "selected").mkdir(exist_ok=True)
            atomic_torch(
                output / stable, checkpoint_payload(learner, p, saved["step"], metric, score)
            )
            saved["best_by_metric"][metric] = dict(
                score=score, step=saved["step"], file=str(stable), public_file=filename
            )


def restore_selected_aliases(output, saved):
    # Public best*.pt files are recoverable aliases, never the source of truth.
    import shutil

    for record in saved["best_by_metric"].values():
        public = output / record["public_file"]
        temp = public.with_suffix(".pt.tmp")
        shutil.copyfile(output / record["file"], temp)
        temp.replace(public)


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
    restore_selected_aliases(output, saved)
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
            training_env_version=ENV_VERSION,
            arm=saved["protocol"]["arm"],
            preset=saved["protocol"]["preset"],
            policy_actions=saved["protocol"]["policy_actions"],
            fair_config=saved["protocol"]["fair_config"],
            obs_dim=OBS_DIM,
            state_dim=STATE_DIM,
            device=saved["protocol"]["device"],
            pause_at_step=saved.get("pause_at_step", 0),
            qos_tolerance_mbps=1e-6,
        ),
    )
    atomic_json(output / "energy_events.json", saved.get("energy_events", []))
    atomic_json(output / "collision_events.json", saved.get("collision_events", []))
    write_csv(output / "training.csv", saved["training"])
    write_csv(output / "validation.csv", saved["validation"])


def plot_curves(output, saved):
    from scripts.analysis.plot_bs5_fourarm400k import plot_run

    plot_run(output, saved)


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
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(p["training_seed"])
    np.random.seed(p["training_seed"])
    torch.manual_seed(p["training_seed"])
    learner = make_learner(
        p["arm"], lr=p["lr"], gamma=p["gamma"], device=p["device"], architecture=p["architecture"]
    )
    replay = ReplayBuffer(p["replay_capacity"], seed=p["training_seed"])
    rng = np.random.default_rng(p["training_seed"])
    if args.resume:
        saved = torch.load(output / "resume.pt", map_location=p["device"], weights_only=False)
        if json.loads((output / "resolved_config.json").read_text()) != p["resolved_config"]:
            raise ValueError("Saved effective configuration changed")
        if saved["protocol"] != p:
            raise RuntimeError(
                "Multi-hop protocol/source/weights/threshold/runtime/actions changed; resume refused"
            )
        for name in ["agent", "mixer", "target_agent", "target_mixer"]:
            getattr(learner, name).load_state_dict(saved["model"][name])
        learner.optimizer.load_state_dict(saved["model"]["optimizer"])
        replay = saved["replay"]
        random.setstate(saved["rng"]["python"])
        np.random.set_state(saved["rng"]["numpy"])
        torch.set_rng_state(saved["rng"]["torch"].cpu())
        if saved["rng"]["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([state.cpu() for state in saved["rng"]["cuda"]])
        rng.bit_generator.state = saved["rng"]["actions"]
        for metric, record in saved["best_by_metric"].items():
            best = torch.load(output / record["file"], map_location="cpu", weights_only=True)
            if (
                best["protocol_sha256"] != digest(p)
                or best["selected_step"] != record["step"]
                or best["validation_score"] != record["score"]
            ):
                raise RuntimeError("Best checkpoint lineage mismatch: " + metric)
        restore_selected_aliases(output, saved)
        print(f"Resume {p['arm']} from {saved['step']} committed steps", flush=True)
    else:
        output.mkdir(parents=True)
        atomic_json(output / "protocol.json", p)
        atomic_json(output / "resolved_config.json", p["resolved_config"])
        saved = dict(
            protocol=p,
            step=0,
            episode=0,
            updates=0,
            best_by_metric={},
            validated_steps=[],
            training=[],
            energy_events=[],
            collision_events=[],
            validation=[],
            task_signature=None,
            status="prepared",
        )
        save_boundary(output, learner, replay, rng, saved)
    saved["pause_at_step"] = args.pause_at_step
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
            plot_curves(output, saved)
            print(
                f"validation {p['arm']} Fair={p['fair_config']['mode']} step={step} service={np.mean([r['service_score'] for r in rows]):.5f}",
                flush=True,
            )
        if step >= p["total_steps"]:
            saved["status"] = "complete"
            break
        if (
            STOP
            or (args.pause_at_step > 0 and step >= args.pause_at_step)
            or (args.max_seconds > 0 and time.perf_counter() - started >= args.max_seconds)
            or (
                args.stop_after_episodes > 0
                and saved["episode"] - session_episode >= args.stop_after_episodes
            )
        ):
            saved["status"] = "paused_at_episode_boundary"
            break
        condition = condition_for(step, p["arm"], p)
        signature = [list(x) for x in task_signature(condition)] + [
            ["S4_entry", condition["stage"] == "S4"]
        ]
        if signature != saved["task_signature"]:
            replay = ReplayBuffer(p["replay_capacity"], seed=p["training_seed"] + saved["episode"])
            saved["task_signature"] = signature
        horizon = (
            p["full_horizon"]
            if condition["stage"] == "S4"
            else (
                p["battery_horizon"]
                if condition["substage"] == "S3d" and not p["diagnostic"]
                else p["early_horizon"]
            )
        )
        if step + horizon > p["total_steps"]:
            raise RuntimeError("Episode would cross budget")
        episode = saved["episode"]
        scenario = p["train_seed_base"] + episode % p["scenario_cycle"]
        setup_seed = int(
            np.random.SeedSequence([p["training_seed"], episode, 0x5632]).generate_state(1)[0]
        )
        residual = float(np.random.default_rng(setup_seed).uniform(*p["residual_battery_range"]))
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
            lr = (
                linear_learning_rate(global_step, p["total_steps"], p["lr"], p["lr_end"])
                if p["lr_schedule"] == "linear"
                else p["lr"]
            )
            for group in learner.optimizer.param_groups:
                group["lr"] = lr
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
        saved["collision_events"].extend(
            dict(episode=episode, scenario_seed=scenario, **event) for event in env.collision_events
        )
        saved["energy_events"].extend(
            dict(episode=episode, scenario_seed=scenario, **event) for event in env.energy_events
        )
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
                learning_rate=lr,
                episode_consumed_wh=sum(r["consumed_wh"] for r in samples),
                episode_swaps=sum(r["swaps_completed"] for r in samples),
                episode_active_blocked=sum(r["active_blocked_uavs"] for r in samples),
                episode_returning_blocked=sum(r["returning_blocked_uavs"] for r in samples),
                episode_policy_move_requests=sum(r["policy_move_requests"] for r in samples),
                episode_energy_shortfall_j=sum(r["energy_shortfall_j"] for r in samples),
                epsilon=epsilon,
                updates=saved["updates"],
                scenario_seed=scenario,
                wall_seconds=time.perf_counter() - episode_start,
            )
        )
        saved["training"][-1].update(training_metrics(samples))
        saved["status"] = "running"
        save_boundary(output, learner, replay, rng, saved)
        print(
            f"{p['arm']} Fair={p['fair_config']['mode']} {saved['step']}/{p['total_steps']} {condition['substage']} service={saved['training'][-1]['service_score']:.4f}",
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
    from scripts.train import parser as shared_parser

    return shared_parser()


if __name__ == "__main__":
    from scripts.train import main

    main()
