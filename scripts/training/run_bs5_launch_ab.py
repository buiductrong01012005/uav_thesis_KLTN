"""Prepare or run the fixed-BS5 launch A/B pilot. Never runs training by default.

--prepare: physics, short debug runs, offline-bundle acceptance, protocol freeze.
--run: execute the frozen eight-run matrix; --resume restarts committed boundaries.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import zipfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.bs5_bundle import (
    aggregate_episode,
    analyze,
    atomic_json,
    canonical_hash,
    file_hash,
    read_json,
)
from src.envs.bs5_launch_ab_env import (
    ARMS,
    LAUNCHES,
    ROUTER_VERSION,
    Geometry,
    make_env,
    shadow_scores,
)
from src.envs.v0_env import MOVE_DELTAS
from src.rl.bs5_launch_qmix import BS5LaunchQMIX
from src.rl.qmix import ReplayBuffer, Transition

PHASE = "after_movement_before_energy_and_ue_mobility"


@contextmanager
def experiment_lock(root):
    """OS lock is released even after a process crash; prevent concurrent runs."""
    with (root / "run.lock").open("a+b") as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def plain(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(type(obj).__name__)


def serial(obj):
    return json.loads(json.dumps(obj, default=plain, allow_nan=False))


def write_trace(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with gzip.open(temp, "wt", encoding="utf-8", compresslevel=3) as stream:
        json.dump(payload, stream, default=plain, separators=(",", ":"), allow_nan=False)
    temp.replace(path)


def source_files():
    skip = {"__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", ".venv", "venv"}
    files = []
    for folder in ("src", "scripts", "tests", "configs"):
        for parent, dirs, names in os.walk(ROOT / folder, followlinks=False):
            dirs[:] = [
                d
                for d in dirs
                if d not in skip
                and not (
                    getattr((Path(parent) / d).lstat(), "st_file_attributes", 0)
                    & stat.FILE_ATTRIBUTE_REPARSE_POINT
                )
                and not (Path(parent) / d).is_symlink()
            ]
            for name in names:
                path = Path(parent) / name
                if path.suffix not in (".pyc", ".pyo") and not path.is_symlink():
                    files.append(path)
    files += [
        ROOT / n
        for n in ("AGENTS.md", "README.md", ".gitignore", "pyproject.toml")
        if (ROOT / n).is_file()
    ]
    files += list(ROOT.glob("requirements*.txt"))
    return sorted(files)


def source_identity():
    hashes = {p.relative_to(ROOT).as_posix(): file_hash(p) for p in source_files()}
    return dict(sha256=canonical_hash(hashes), files=hashes)


def freeze_source(output, identity):
    atomic_json(output / "source_identity.json", identity)
    with zipfile.ZipFile(output / "source_snapshot.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for name in identity["files"]:
            archive.write(ROOT / name, "uav_thesis/" + name)


def make_protocol(config, identity, *, smoke=False):
    config = deepcopy(config)
    if smoke:
        config.update(
            training_seeds=[949451, 949452],
            train_scenario_seeds=[150001, 150004],
            validation_scenario_seeds=[160001, 160002],
            requested_steps_per_run=300,
            warmup=64,
            epsilon_decay_steps=300,
            replay_capacity=2000,
        )
    steps = config.pop("requested_steps_per_run")
    train = list(range(config["train_scenario_seeds"][0], config["train_scenario_seeds"][1] + 1))
    validation = list(
        range(config["validation_scenario_seeds"][0], config["validation_scenario_seeds"][1] + 1)
    )
    development = list(
        range(config["development_scenario_seeds"][0], config["development_scenario_seeds"][1] + 1)
    )
    assert not (
        set(train) & set(validation)
        or set(train) & set(development)
        or set(validation) & set(development)
    )
    assert steps % config["horizon"] == 0
    schedule = {}
    for seed in config["training_seeds"]:
        scenario_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x5343454E]))
        battery_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x42415454]))
        order = []
        while len(order) < steps // config["horizon"]:
            order.extend(map(int, scenario_rng.permutation(train)))
        schedule[str(seed)] = [
            dict(
                scenario_seed=s,
                residual_battery=float(
                    battery_rng.uniform(*config["train_residual_battery_range"])
                ),
            )
            for s in order[: steps // config["horizon"]]
        ]
    config.update(
        scope="debug_smoke_not_main_training"
        if smoke
        else "validation_pilot_not_final_thesis_test",
        source_sha256=identity["sha256"],
        router_version=ROUTER_VERSION,
        steps_per_run=steps,
        evaluation_steps=[0, steps] if smoke else list(range(0, steps + 1, 6000)),
        train_seeds=train,
        validation_seeds=validation,
        development_seeds=development,
        scenario_schedule=schedule,
        arms=serial(ARMS),
        launches=serial(LAUNCHES),
        slot_s=5.0,
        failed_bs_idx=5,
        failure_mode="fixed",
        state_dim=111,
        obs_dim=116,
        n_actions=10,
        no_relay="odd Cartesian actions masked in act and bootstrap, train and validation",
        checkpoint_selection="maximum mean validation utility including step 0; exact tie selects earlier step",
        resume="exact state at committed evaluation/episode boundary; uncommitted work archived and rerun",
        phase=PHASE,
        shadow_coverage="every validation slot; all 3 scalar scores; no training feedback",
        oracle_coverage="slot index divisible by oracle_every_slots, including slot 1; diagnostic only",
        train_trace_coverage="first episode of each scenario plus every Nth episode; failures preserved",
        reward="mean(min(per_UE_rate_mbps / demand_mbps, 1)); unchanged",
        validation_usage="already inspected validation is never called unseen final test",
        initial_weights="same training seed => identical random tensors across all four arms; independent optimizer/replay/exploration thereafter",
        execution_environment=dict(
            python=sys.version,
            torch=torch.__version__,
            numpy=np.__version__,
            cuda_available=torch.cuda.is_available(),
            logical_cpus=os.cpu_count(),
        ),
    )
    return config


def registry_new(protocol):
    return dict(
        status="prepared",
        protocol_sha256=canonical_hash(protocol),
        created_utc=utc_now(),
        runs={
            f"{arm}_seed{seed}": dict(status="not_started", steps=0)
            for seed in protocol["training_seeds"]
            for arm in ARMS
        },
        attempts=[],
    )


def weights_hash(learner):
    digest = hashlib.sha256()
    for prefix, model in (("agent", learner.agent), ("mixer", learner.mixer)):
        for name, tensor in sorted(model.state_dict().items()):
            digest.update((prefix + name).encode())
            digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def capture_decision(env):
    return dict(
        positions=env.uav_positions.copy(),
        status=env.status.copy(),
        battery=env.battery.copy(),
        roles=env.roles.copy(),
    )


def step_record(
    env,
    decision,
    mask,
    actions,
    out,
    geometry,
    *,
    arm,
    seed,
    scenario,
    slot,
    checkpoint_step=None,
    oracle=False,
):
    snap = env.last_service_snapshot
    _, required = geometry.ue_masks(snap["ue_positions"])
    region = geometry.relay & ~geometry.direct
    at = [
        bool(region[int(p[0]) * env.cfg.grid_size + int(p[1])]) and s == 0
        for p, s in zip(snap["uav_positions"], snap["status"])
    ]
    actual = []
    for i, (p, before) in enumerate(zip(snap["uav_positions"], decision["positions"])):
        delta = p - before
        move = next((m for m, d in enumerate(MOVE_DELTAS) if np.array_equal(delta, d)), None)
        actual.append(
            dict(
                move=move,
                role=int(snap["roles"][i]),
                autonomous=bool(decision["status"][i] != 0),
                requested_move_executed=bool(np.array_equal(delta, MOVE_DELTAS[actions[i] // 2])),
            )
        )
    row = dict(
        arm=arm,
        seed=seed,
        scenario_seed=scenario,
        slot=slot,
        checkpoint_step=checkpoint_step,
        service_phase=PHASE,
        raw_actions=actions,
        action_mask=mask,
        decoded_actions=[[a // 2, a % 2] for a in actions],
        applied_actions=actual,
        decision=decision,
        service=snap,
        reward=float(out["reward"]),
        post=dict(
            positions=env.uav_positions.copy(),
            status=env.status.copy(),
            roles=env.roles.copy(),
            battery=env.battery.copy(),
            ue_positions=env.ue_positions.copy(),
        ),
        relay_required_ue=required,
        uav_in_relay_required_access_region=at,
        nearest_station_steps=[int(env._nearest_station(p)[1]) for p in snap["uav_positions"]],
    )
    if checkpoint_step is not None:
        row["shadow"] = shadow_scores(env, snap, oracle=oracle)
    return serial(row)


def evaluate(learner, protocol, output, arm, seed, step):
    started = time.monotonic()
    env = make_env(
        arm, horizon=protocol["horizon"], residual_battery=protocol["validation_residual_battery"]
    )
    geometry = Geometry(env)
    means = []
    learner.agent.eval()
    for scenario in protocol["validation_seeds"]:
        out = env.reset(seed=scenario)
        residual = dict(
            id=int(env.residual_uav_idx), energy=float(env.battery[env.residual_uav_idx])
        )
        rng = np.random.default_rng(np.random.SeedSequence([scenario, 0x4556414C]))
        rows = []
        for t in range(protocol["horizon"]):
            mask, decision = out["action_mask"].copy(), capture_decision(env)
            actions = learner.act(out["obs"], mask, 0.0, rng)
            out = env.step(actions)
            rows.append(
                step_record(
                    env,
                    decision,
                    mask,
                    actions,
                    out,
                    geometry,
                    arm=arm,
                    seed=seed,
                    scenario=scenario,
                    slot=t + 1,
                    checkpoint_step=step,
                    oracle=t % protocol["oracle_every_slots"] == 0,
                )
            )
        metrics = aggregate_episode(rows, protocol["slot_s"])
        means.append(metrics["utility"])
        write_trace(
            output / "evaluations" / f"{step:06d}" / f"{scenario}.json.gz",
            dict(metrics=metrics, trace=rows, initial_residual=residual),
        )
    learner.agent.train()
    result = dict(step=step, utility=float(np.mean(means)), runtime_s=time.monotonic() - started)
    atomic_json(output / "evaluations" / f"{step:06d}" / "summary.json", result)
    return result


def atomic_torch(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    torch.save(data, temp)
    temp.replace(path)


def save_state(output, learner, replay, exploration, commit):
    state = dict(
        commit=commit,
        agent=learner.agent.state_dict(),
        mixer=learner.mixer.state_dict(),
        target_agent=learner.target_agent.state_dict(),
        target_mixer=learner.target_mixer.state_dict(),
        optimizer=learner.optimizer.state_dict(),
        replay=replay,
        exploration_rng=exploration.bit_generator.state,
        python_rng=random.getstate(),
        numpy_rng=np.random.get_state(),
        torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        environment="next episode reset from frozen scenario/residual schedule; no in-flight episode",
        scheduler="fixed LR; epsilon determined by committed environment step",
    )
    atomic_torch(
        output / "checkpoints" / f"{commit['steps']:06d}.pt",
        dict(agent=state["agent"], mixer=state["mixer"], step=commit["steps"]),
    )
    atomic_torch(output / "resume.pt", state)
    # JSON is the final commit marker. resume.pt embeds the authoritative marker
    # too, so a crash between replaces can be reconciled on resume.
    atomic_json(output / "commit.json", commit)


def load_state(path, learner, exploration):
    state = torch.load(path, map_location=learner.device, weights_only=False)
    for key in ("agent", "mixer", "target_agent", "target_mixer"):
        getattr(learner, key).load_state_dict(state[key])
    learner.optimizer.load_state_dict(state["optimizer"])
    exploration.bit_generator.state = state["exploration_rng"]
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"].cpu())
    if state["cuda_rng"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])
    return state["commit"], state["replay"]


def archive_uncommitted(directory, commit):
    candidates = []
    for sub in ("episodes", "updates", "train_traces"):
        for path in (directory / sub).glob("*"):
            if int(path.name.split(".")[0]) >= commit["episodes"]:
                candidates.append(path)
    for path in (directory / "evaluations").glob("*"):
        if int(path.name) > commit["steps"]:
            candidates.append(path)
    for path in (directory / "checkpoints").glob("*.pt"):
        if int(path.stem) > commit["steps"]:
            candidates.append(path)
    if candidates:
        archive = (
            directory / "aborted_attempts" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        )
        for path in candidates:
            assert path.resolve().is_relative_to(directory.resolve())
            target = archive / path.relative_to(directory)
            target.parent.mkdir(parents=True, exist_ok=True)
            path.rename(target)


def run_one(root, protocol, registry, arm, seed, *, resume=False, export=True):
    name = f"{arm}_seed{seed}"
    output = root / "runs" / name
    if registry["runs"][name]["status"] == "complete":
        return
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    env = make_env(arm, horizon=protocol["horizon"])
    geometry = Geometry(env)
    learner = BS5LaunchQMIX(
        env, lr=protocol["lr"], gamma=protocol["gamma"], device=protocol["device"]
    )
    exploration = np.random.default_rng(np.random.SeedSequence([seed, 0x504F4C49]))
    replay = ReplayBuffer(protocol["replay_capacity"], seed + 20000)
    meta = dict(
        run=name,
        arm=arm,
        seed=seed,
        source_sha256=protocol["source_sha256"],
        resolved_config=serial(asdict(env.cfg)),
        initial_weights_sha256=weights_hash(learner),
        state_dim=env.state_dim,
        obs_dim=env.obs_dim,
        n_actions=env.n_actions,
        parameter_count=sum(p.numel() for p in learner.agent.parameters())
        + sum(p.numel() for p in learner.mixer.parameters()),
    )
    meta["config_sha256"] = canonical_hash(meta["resolved_config"])
    if (output / "meta.json").exists():
        assert read_json(output / "meta.json") == meta, "run identity changed"
    else:
        atomic_json(output / "meta.json", meta)
    commit = dict(
        steps=0,
        episodes=0,
        updates=0,
        best_step=0,
        best_utility=-1.0,
        train_seconds=0.0,
        update_seconds=0.0,
        eval_seconds=0.0,
        io_seconds=0.0,
    )
    if (output / "resume.pt").exists():
        if not resume:
            raise ValueError("Run exists; use --resume to continue")
        commit, replay = load_state(output / "resume.pt", learner, exploration)
        atomic_json(output / "commit.json", commit)
        archive_uncommitted(output, commit)
    else:
        ev = evaluate(learner, protocol, output, arm, seed, 0)
        commit.update(best_utility=ev["utility"], eval_seconds=ev["runtime_s"])
        save_state(output, learner, replay, exploration, commit)
    registry["runs"][name] = dict(status="running", steps=commit["steps"])
    atomic_json(root / "run_registry.json", registry)
    attempt = dict(
        run=name,
        started_utc=utc_now(),
        from_step=commit["steps"],
        executed_train_steps=0,
        learner_updates=0,
        status="running",
    )
    registry["attempts"].append(attempt)
    atomic_json(root / "run_registry.json", registry)
    try:
        while commit["steps"] < protocol["steps_per_run"]:
            eid = commit["episodes"]
            schedule = protocol["scenario_schedule"][str(seed)]
            spec = schedule[eid]
            env.cfg.residual_initial_battery = spec["residual_battery"]
            out = env.reset(seed=spec["scenario_seed"])
            first = spec["scenario_seed"] not in {s["scenario_seed"] for s in schedule[:eid]}
            trace_episode = first or eid % protocol["train_trace_every_episodes"] == 0
            rows, updates = [], []
            started = time.monotonic()
            eps_start = None
            for t in range(protocol["horizon"]):
                eps = 1 + (protocol["epsilon_end"] - 1) * min(
                    commit["steps"] / protocol["epsilon_decay_steps"], 1
                )
                if eps_start is None:
                    eps_start = eps
                mask, decision = out["action_mask"].copy(), capture_decision(env)
                actions = learner.act(out["obs"], mask, eps, exploration)
                nxt = env.step(actions)
                replay.add(
                    Transition(
                        out["obs"],
                        out["state"],
                        np.asarray(actions, dtype=np.int64),
                        nxt["reward"],
                        nxt["obs"],
                        nxt["state"],
                        nxt["action_mask"],
                        float(nxt["truncated"]),
                    )
                )
                commit["steps"] += 1
                attempt["executed_train_steps"] += 1
                if (
                    len(replay) >= max(protocol["warmup"], protocol["batch_size"])
                    and commit["steps"] % protocol["update_every"] == 0
                ):
                    update_started = time.monotonic()
                    metrics = learner.train_batch(replay.sample(protocol["batch_size"]))
                    commit["update_seconds"] += time.monotonic() - update_started
                    commit["updates"] += 1
                    attempt["learner_updates"] += 1
                    updates.append(
                        dict(
                            step=commit["steps"],
                            update=commit["updates"],
                            replay_size=len(replay),
                            run=name,
                            **metrics,
                        )
                    )
                if commit["steps"] % protocol["target_every"] == 0:
                    learner.update_targets()
                rows.append(
                    step_record(
                        env,
                        decision,
                        mask,
                        actions,
                        nxt,
                        geometry,
                        arm=arm,
                        seed=seed,
                        scenario=spec["scenario_seed"],
                        slot=t + 1,
                    )
                )
                out = nxt
            assert out["truncated"]
            metrics = aggregate_episode(rows, protocol["slot_s"])
            record = dict(
                run=name,
                arm=arm,
                seed=seed,
                scenario_seed=spec["scenario_seed"],
                episode_id=eid,
                source_sha256=meta["source_sha256"],
                config_sha256=meta["config_sha256"],
                env_steps=commit["steps"],
                learner_updates=commit["updates"],
                epsilon_start=eps_start,
                epsilon_end=eps,
                metrics=metrics,
                residual_uav_id=int(env.residual_uav_idx),
                residual_battery=spec["residual_battery"],
                completion_status="complete",
                first_scenario_encounter=first,
            )
            atomic_json(output / "episodes" / f"{eid:06d}.json", record)
            atomic_json(output / "updates" / f"{eid:06d}.json", updates)
            if trace_episode:
                write_trace(
                    output / "train_traces" / f"{eid:06d}.json.gz",
                    dict(trace=rows, metrics=metrics),
                )
            commit["train_seconds"] += time.monotonic() - started
            commit["episodes"] += 1
            if commit["steps"] in protocol["evaluation_steps"]:
                ev = evaluate(learner, protocol, output, arm, seed, commit["steps"])
                commit["eval_seconds"] += ev["runtime_s"]
                if ev["utility"] > commit["best_utility"]:
                    commit.update(best_step=commit["steps"], best_utility=ev["utility"])
                io_started = time.monotonic()
                save_state(output, learner, replay, exploration, commit)
                commit["io_seconds"] += time.monotonic() - io_started
                done = commit["steps"] == protocol["steps_per_run"]
                registry["runs"][name] = dict(
                    status="complete" if done else "running", steps=commit["steps"]
                )
                atomic_json(root / "run_registry.json", registry)
                print(
                    json.dumps(
                        dict(
                            event="validation",
                            run=name,
                            step=commit["steps"],
                            utility=ev["utility"],
                            best_step=commit["best_step"],
                        )
                    ),
                    flush=True,
                )
                if export:
                    export_bundles(root)
        attempt.update(status="complete", ended_utc=utc_now())
        atomic_json(root / "run_registry.json", registry)
    except BaseException as exc:
        durable = read_json(output / "commit.json")
        attempt.update(
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            ended_utc=utc_now(),
            reason=f"{type(exc).__name__}: {exc}",
            rollback_to_step=durable["steps"],
        )
        registry["runs"][name] = dict(status="incomplete", steps=durable["steps"])
        registry["status"] = "incomplete"
        atomic_json(root / "run_registry.json", registry)
        if export:
            export_bundles(root)
        raise


def bundle_members(root, *, full):
    for name in ("protocol.json", "schema.json", "source_identity.json", "run_registry.json"):
        yield root / name, name
    for sub in ("preflight", "analysis"):
        for path in sorted((root / sub).rglob("*")):
            if path.is_file():
                yield path, path.relative_to(root).as_posix()
    registry = read_json(root / "run_registry.json")
    for run, entry in registry["runs"].items():
        if entry["status"] == "not_started":
            continue
        directory = root / "runs" / run
        commit = read_json(directory / "commit.json")
        for name in ("meta.json", "commit.json"):
            yield directory / name, f"runs/{run}/{name}"
        for sub in ("episodes", "updates", "train_traces"):
            for path in sorted((directory / sub).glob("*")):
                if int(path.name.split(".")[0]) < commit["episodes"]:
                    yield path, path.relative_to(root).as_posix()
        for step in [
            s for s in read_json(root / "protocol.json")["evaluation_steps"] if s <= commit["steps"]
        ]:
            for path in sorted((directory / "evaluations" / f"{step:06d}").glob("*")):
                yield path, path.relative_to(root).as_posix()
        if full:
            yield directory / "resume.pt", f"runs/{run}/resume.pt"
            for path in sorted((directory / "checkpoints").glob("*.pt")):
                if int(path.stem) <= commit["steps"]:
                    yield path, path.relative_to(root).as_posix()
            for path in sorted((directory / "aborted_attempts").rglob("*")):
                if path.is_file():
                    yield path, path.relative_to(root).as_posix()
    if full:
        yield root / "source_snapshot.zip", "source/uav_thesis_source.zip"


def export_bundles(root):
    root = Path(root)
    analyze(root, root / "analysis", check_manifest=False)
    destination = root / "exports"
    destination.mkdir(exist_ok=True)
    analyzer = (ROOT / "scripts/analysis/bs5_bundle.py").read_bytes()
    validator = b'from analyze_bundle import main\nif __name__ == "__main__":\n    main()\n'
    for full, name in ((False, "audit_light.zip"), (True, "repro_full.zip")):
        target = destination / name
        temp = target.with_name(name + ".tmp")
        hashes = {}
        with zipfile.ZipFile(
            temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3
        ) as archive:
            for path, relative in bundle_members(root, full=full):
                # Traces and source ZIP are already compressed.
                archive.write(
                    path,
                    relative,
                    compress_type=zipfile.ZIP_STORED
                    if path.suffix in (".gz", ".zip")
                    else zipfile.ZIP_DEFLATED,
                )
                hashes[relative] = file_hash(path)
            for relative, data in (
                ("analyze_bundle.py", analyzer),
                ("validate_bundle.py", validator),
            ):
                archive.writestr(relative, data)
                hashes[relative] = hashlib.sha256(data).hexdigest()
            archive.writestr("SHA256SUMS.json", json.dumps(hashes, indent=2))
        with zipfile.ZipFile(temp) as archive:
            assert archive.testzip() is None
        temp.replace(target)


def topology_svg(env):
    # Coordinates are derived from the source; row increases upward in this map.
    def xy(cell):
        return 40 + float(cell[1]) * 25, 530 - float(cell[0]) * 25

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="760" height="600" viewBox="0 0 760 600">',
        '<rect width="760" height="600" fill="white"/>',
    ]
    for r, c in env.affected_cells:
        x, y = xy((r, c))
        parts.append(f'<rect x="{x - 12.5}" y="{y - 12.5}" width="25" height="25" fill="#fee2e2"/>')
    for i, cell in enumerate(env.bs_cells):
        x, y = xy(cell)
        parts.append(
            f'<circle cx="{x}" cy="{y}" r="8" fill="{"#dc2626" if i == 5 else "#111827"}"/><text x="{x + 10}" y="{y}">BS{i}</text>'
        )
    for cell in env.charging_stations:
        x, y = xy(cell)
        parts.append(f'<rect x="{x - 6}" y="{y - 6}" width="12" height="12" fill="#16a34a"/>')
    for label, launch in LAUNCHES.items():
        for cell in launch:
            x, y = xy(cell)
            parts.append(
                f'<circle cx="{x}" cy="{y}" r="5" fill="{"#2563eb" if label == "A" else "#f59e0b"}"/>'
            )
    parts += [
        '<text x="560" y="40">BS5 failed (red region)</text>',
        '<text x="560" y="70">A: blue; B: orange</text>',
        '<text x="560" y="100">Charging: green</text>',
        '<text x="40" y="575">20×20 cells, 50 m; cell centers; x=column, y=row</text></svg>',
    ]
    return "".join(parts)


def preflight(root, config):
    directory = root / "preflight"
    directory.mkdir()
    env = make_env("A-R")
    env.reset(seed=140001)
    assert env.cfg.failed_bs_idx == 5 and np.array_equal(env.failed_bs_xy, [925, 925])
    assert np.array_equal(env.bs_xy[4], [425, 725])
    assert not any(np.array_equal(bs, env.bs_xy[5]) for bs in env.surviving_bs_xy)
    geometry = Geometry(env)
    # Closed unit-cell regions, not just their grid-index points.
    lo = env.affected_cells[None]
    hi = lo + 1.0
    distance = np.linalg.norm(
        np.maximum(np.maximum(lo - geometry.cells[:, None], geometry.cells[:, None] - hi), 0),
        axis=-1,
    )
    possible = (distance <= env.cfg.access_radius_cells + 1e-9).any(axis=1)
    launches = []
    for label, launch in LAUNCHES.items():
        assert len(set(launch)) == 5
        e = make_env(label + "-R")
        e.reset(seed=140001)
        d = np.abs(geometry.cells[:, None] - np.asarray(launch)[None]).sum(axis=-1).min(axis=1)
        launches.append(
            dict(
                launch=label,
                cells=launch,
                initial_roles=e.roles.tolist(),
                lower_bound_any_region_moves=int(d[possible & geometry.relay].min()),
                direct_only_lower_bound_moves=int(d[possible & geometry.direct].min()),
                station_steps=[int(e._nearest_station(p)[1]) for p in e.uav_positions],
            )
        )
    shares = []
    for scenario in range(
        *[config["development_scenario_seeds"][0], config["development_scenario_seeds"][1] + 1]
    ):
        env.reset(seed=scenario)
        for t in range(config["horizon"]):
            assert all(env._ue_position_is_valid(p) for p in env.ue_positions)
            if t % 15 == 0:
                direct, required = geometry.ue_masks(env.ue_positions)
                shares.append(
                    dict(
                        scenario=scenario,
                        slot=t + 1,
                        direct_fraction=float(direct.mean()),
                        relay_required_fraction=float(required.mean()),
                    )
                )
            env._advance_mobile_ues()
    # Same action proposal stream; valid-mask fallback registered before runs.
    probe_rows = []
    for scenario in range(
        config["development_scenario_seeds"][0], config["development_scenario_seeds"][0] + 3
    ):
        for arm in ARMS:
            e = make_env(arm)
            out = e.reset(seed=scenario)
            rng = np.random.default_rng(scenario + 900000)
            rewards, digest = [], hashlib.sha256()
            for _ in range(config["horizon"]):
                digest.update(e.ue_positions.tobytes())
                actions = []
                for mask in out["action_mask"]:
                    proposal = int(rng.integers(0, 10))
                    # Map odd to Access for no-relay, then boundary/status Stay.
                    if arm.endswith("N"):
                        proposal -= proposal % 2
                    actions.append(proposal if mask[proposal] else 8)
                out = e.step(actions)
                rewards.append(float(out["reward"]))
            probe_rows.append(
                dict(
                    arm=arm,
                    scenario=scenario,
                    ue_hash=digest.hexdigest(),
                    first_service_slot=next(
                        (i + 1 for i, u in enumerate(rewards) if u > 1e-12), None
                    ),
                    zero_reward_slots=sum(u <= 1e-12 for u in rewards),
                    utility=float(np.mean(rewards)),
                )
            )
    for scenario in {r["scenario"] for r in probe_rows}:
        assert len({r["ue_hash"] for r in probe_rows if r["scenario"] == scenario}) == 1
    # Positive control: globally direct-unreachable UE, one Access plus one Relay.
    env.reset(seed=140001)
    env.uav_positions[:2] = [[19, 16], [16, 11]]
    env.roles[:2] = [0, 1]
    env.status[2:] = 2
    env.ue_positions[:] = [19, 19]
    direct, required = geometry.ue_masks(env.ue_positions)
    assert not direct.any() and required.all()
    from src.envs.bs5_launch_ab_env import score_network

    actual, no_relay = score_network(env), score_network(env, False)
    assert actual["utility"] > 0 and no_relay["utility"] == 0
    atomic_json(
        directory / "physics.json",
        serial(
            dict(
                executed_probe_env_steps=len(probe_rows) * config["horizon"],
                geometry_mobility_advances=len(
                    range(
                        config["development_scenario_seeds"][0],
                        config["development_scenario_seeds"][1] + 1,
                    )
                )
                * config["horizon"],
                failed_bs_idx=5,
                bs_xy_m=env.bs_xy,
                charging_cells=env.charging_stations,
                launches=launches,
                geometry_samples=shares,
                random_probe=probe_rows,
                positive_control=dict(
                    ue=[19, 19],
                    access=[19, 16],
                    relay=[16, 11],
                    active_uavs=2,
                    global_direct_reachability_ceiling=0,
                    service=actual,
                    caution="synthetic physics check, not a learned full-scene result",
                ),
                geometry_caution="reachability only; ignores simultaneous fleet capacity, collision, battery and real service onset",
                review_probe_history="previous read-only review inspected geometry on 120001..120010 and action traces on 130001; these are not unseen final test scenarios",
            )
        ),
    )
    (directory / "topology.svg").write_text(topology_svg(env), encoding="utf-8")


def offline_acceptance(root):
    """Copy light export outside the experiment; block torch/src imports."""
    block = """import importlib.abc,sys,runpy
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in {'torch','src'}: raise ImportError('blocked '+fullname)
sys.meta_path.insert(0,Block())
sys.argv=['analyze_bundle.py','--root','.', '--output', 'offline_rebuilt', '--require-complete']
runpy.run_path('analyze_bundle.py',run_name='__main__')
"""
    with tempfile.TemporaryDirectory(prefix="bs5_acceptance_") as tmp:
        with zipfile.ZipFile(root / "exports/audit_light.zip") as archive:
            assert not any(n.endswith(".pt") for n in archive.namelist())
            archive.extractall(tmp)
        process = subprocess.run(
            [sys.executable, "-B", "-c", block],
            cwd=tmp,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        if process.returncode:
            raise RuntimeError(process.stdout + process.stderr)
        return dict(
            checkpoint_free=True, torch_env_imports_blocked=True, output=process.stdout.strip()
        )


def run_matrix(root, *, resume=False, export=True):
    protocol = read_json(root / "protocol.json")
    assert source_identity()["sha256"] == protocol["source_sha256"], (
        "Source changed after protocol freeze; prepare a new output directory"
    )
    registry = read_json(root / "run_registry.json")
    assert canonical_hash(protocol) == registry["protocol_sha256"]
    torch.set_num_threads(protocol["torch_threads"])
    registry["status"] = "running"
    atomic_json(root / "run_registry.json", registry)
    try:
        for seed in protocol["training_seeds"]:
            for arm in ARMS:
                run_one(root, protocol, registry, arm, seed, resume=resume, export=export)
        registry["status"] = "complete"
        registry["completed_utc"] = utc_now()
        atomic_json(root / "run_registry.json", registry)
        export_bundles(root)
    except BaseException:
        registry["status"] = "incomplete"
        atomic_json(root / "run_registry.json", registry)
        raise


def schema():
    return dict(
        schema_version=1,
        units=dict(
            positions="row,col grid coordinates",
            battery="normalized energy",
            rates="Mbps",
            slot="1-based, 5 seconds",
            utility="0..1 fraction of demand",
        ),
        phase=PHASE,
        action_encoding="move*2+role; Access=0, Relay=1; moves N,S,E,W,Stay",
        trace="one gzip JSON packet per validation episode; contains complete ordered slot records",
        censoring="first_service_slot=null with first_service_censored=true means not recovered",
        gradients="before_clip is clip_grad_norm_ return; after_clip is measured norm; clipping threshold 10",
        resume="resume.pt includes online/targets/Adam/replay/all policy/global RNG; committed evaluation boundaries only",
        oracle="all Active role assignments; instantaneous upper reference, never a policy action",
        failure="one BS, ID 5; full mobility and battery; no curriculum or pretrained weights",
    )


def prepare(output, config_path):
    if output.exists():
        raise FileExistsError(
            f"Output already exists; preserve it and choose a new --outdir: {output}"
        )
    output.mkdir(parents=True)
    config = read_json(config_path)
    identity = source_identity()
    freeze_source(output, identity)
    preflight(output, config)
    atomic_json(output / "schema.json", schema())
    smoke = output / "smoke"
    smoke.mkdir()
    freeze_source(smoke, identity)
    shutil.copytree(output / "preflight", smoke / "preflight")
    atomic_json(smoke / "schema.json", schema())
    protocol = make_protocol(config, identity, smoke=True)
    atomic_json(smoke / "protocol.json", protocol)
    atomic_json(smoke / "run_registry.json", registry_new(protocol))
    started = time.monotonic()
    run_matrix(smoke, export=False)
    acceptance = offline_acceptance(smoke)
    elapsed = time.monotonic() - started
    commits = [read_json(p) for p in (smoke / "runs").glob("*/commit.json")]
    train_seconds = sum(c["train_seconds"] for c in commits)
    update_seconds = sum(c["update_seconds"] for c in commits)
    eval_seconds = sum(c["eval_seconds"] for c in commits)
    update_count = sum(c["updates"] for c in commits)

    def estimate(steps):
        train = max(0, train_seconds - update_seconds) / (300 * 8) * (steps * 8)
        updates = (
            len(
                [
                    s
                    for s in range(config["update_every"], steps + 1, config["update_every"])
                    if s >= max(config["warmup"], config["batch_size"])
                ]
            )
            * 8
        )
        learning = update_seconds / max(1, update_count) * updates
        evslots = (steps // 6000 + 1) * 10 * config["horizon"] * 8
        evaluation = eval_seconds / (2 * 2 * config["horizon"] * 8) * evslots
        overhead = max(30, elapsed - train_seconds - eval_seconds) * max(1, steps / 6000)
        return (train + learning + evaluation + overhead) * config["runtime_safety_factor"] / 3600

    requested = config["requested_steps_per_run"]
    estimated = estimate(requested)
    if estimated > config["max_estimated_hours"]:
        config["requested_steps_per_run"] = config["fallback_steps_per_run"]
    main = make_protocol(config, identity)
    main["runtime_estimate"] = dict(
        smoke_elapsed_s=elapsed,
        estimated_24k_hours=estimated,
        estimated_selected_hours=estimate(main["steps_per_run"]),
        caveat=f"short {config['device']} smoke extrapolation, including shadow/oracle logging and export allowance; not a runtime guarantee",
    )
    acceptance.update(
        smoke_training_steps=2400,
        smoke_learner_updates=update_count,
        main_training_steps_executed=0,
        smoke_runtime_s=elapsed,
    )
    acceptance["physics_probe_env_steps"] = read_json(output / "preflight/physics.json")[
        "executed_probe_env_steps"
    ]
    atomic_json(output / "preflight/offline_acceptance.json", acceptance)
    atomic_json(output / "protocol.json", main)
    atomic_json(output / "run_registry.json", registry_new(main))
    export_bundles(output)
    print(
        json.dumps(
            dict(
                event="PREPARED_NOT_TRAINED",
                outdir=str(output),
                steps_per_run=main["steps_per_run"],
                total_main_steps=main["steps_per_run"] * 8,
                estimated_hours=main["runtime_estimate"]["estimated_selected_hours"],
            )
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--export", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/bs5_launch_ab_v1.json")
    parser.add_argument("--outdir", type=Path, default=ROOT / "results/bs5_launch_ab_20261022")
    args = parser.parse_args()
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    output = args.outdir.resolve()
    if args.prepare:
        prepare(output, args.config)
    elif args.run:
        run_matrix(output, resume=args.resume)
    else:
        export_bundles(output)


if __name__ == "__main__":
    main()
