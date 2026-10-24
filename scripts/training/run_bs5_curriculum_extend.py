"""Authorized full-state 24k -> 60k continuation; never starts a main run from zero.

Prepare reads trusted local PyTorch checkpoints (pickle), copies immutable parents,
and freezes an explicit continuation authorization. Main and diagnostic paths are
disjoint. No scientific src module is modified by this runner.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from scripts.analysis.bs5_bundle import atomic_json, canonical_hash, file_hash, read_json
from scripts.analysis.analyze_bs5_curriculum_bundle import episode_metrics
from scripts.training.run_bs5_curriculum_pilot import (
    act,
    archive_uncommitted,
    condition_for,
    evaluate,
    make_env,
    record,
)
from scripts.training.run_bs5_launch_ab import (
    atomic_torch,
    capture_decision,
    experiment_lock,
    freeze_source,
    load_state,
    save_state,
    serial,
    source_identity,
    utc_now,
    write_trace,
)
from scripts.training.run_bs5_direct_replay_reset import Timers
from src.envs.bs5_curriculum_env import task_signature
from src.envs.bs5_launch_ab_env import Geometry
from src.rl.bs5_curriculum_qmix import BS5CurriculumQMIX
from src.rl.bs5_curriculum_replay import CurriculumTransition

DEFAULT = ROOT / "configs/bs5_curriculum_extend_60k_v1.json"
FRESH_ORIGIN = "fresh_0_to_60k_v1"
STATE_KEYS = {
    "agent",
    "mixer",
    "target_agent",
    "target_mixer",
    "optimizer",
    "replay",
    "exploration_rng",
    "python_rng",
    "numpy_rng",
    "torch_rng",
    "cuda_rng",
    "commit",
}


def runtime():
    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        driver = None
    return dict(
        python=sys.version,
        torch=torch.__version__,
        numpy=np.__version__,
        cuda=torch.version.cuda,
        cuda_available=torch.cuda.is_available(),
        driver=driver,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        torch_threads=torch.get_num_threads(),
    )


def names(policy):
    return [f"{arm}_seed{seed}" for seed in policy["training_seeds"] for arm in policy["arms"]]


def io_count(timer, kind, paths):
    item = timer.data["io"].setdefault(kind, dict(calls=0, bytes=0))
    for path in paths:
        item["calls"] += 1
        item["bytes"] += Path(path).stat().st_size


def check_policy(policy):
    assert policy["arms"] == ["C-R", "C-N"]
    if policy.get("diagnostic"):
        assert all(s >= 900000 for s in policy["training_seeds"])
        assert 0 < policy["parent_steps"] < policy["total_steps"] <= 1000
        assert policy["scope"].startswith("diagnostic")
    else:
        locked = read_json(DEFAULT)
        if policy.get("origin") == FRESH_ORIGIN:
            locked.update(
                origin=FRESH_ORIGIN,
                scope="fresh_training_validation_pilot_n2_not_historical_continuation",
                parent_source_sha256=policy["parent_source_sha256"],
            )
            assert len(policy["parent_source_sha256"]) == 64
        assert policy == locked, "main continuation policy is locked at four lineages / total60k"


def extend_protocol(parent, policy, identity):
    """Copy frozen prefix; never regenerate the old shared RNG at count=400."""
    p = deepcopy(parent)
    start, end, horizon = policy["parent_steps"], policy["total_steps"], p["horizon"]
    assert parent["steps_per_run"] == start and start < end
    assert start % horizon == end % horizon == 0
    p.update(
        experiment="bs5_curriculum_extend_60k_v1",
        scope=policy["scope"],
        steps_per_run=end,
        arms=policy["arms"],
        training_seeds=policy["training_seeds"],
        source_sha256=identity["sha256"],
        parent_source_sha256=parent["source_sha256"],
        parent_protocol_sha256=canonical_hash(parent),
        continuation=deepcopy(policy),
    )
    p["stage_ends"][-1] = end
    p["common_evaluation_steps"] = list(range(0, end + 1, p["eval_every"]))
    p["evaluation_steps"] = sorted(
        set(parent["evaluation_steps"])
        | set(range(start + p["eval_every"], end + 1, p["eval_every"]))
    )
    p["training_conditions"] = {
        arm: deepcopy(parent["training_conditions"][arm]) for arm in p["arms"]
    }
    for spans in p["training_conditions"].values():
        assert spans[-1]["condition"]["stage"] == "S4"
        spans[-1]["end"] = end
    p["scenario_schedule"] = {}
    for seed in p["training_seeds"]:
        prefix = deepcopy(parent["scenario_schedule"][str(seed)])
        assert len(prefix) == start // horizon
        rngs = [
            np.random.default_rng(np.random.SeedSequence([seed, 0x36304B, i])) for i in range(4)
        ]
        count = (end - start) // horizon
        orders = []
        for rng, pool in zip(rngs, (p["train_pool"], p["fixture_pool"])):
            order = []
            while len(order) < count:
                order.extend(map(int, rng.permutation(pool)))
            orders.append(order)
        suffix = [
            dict(
                scenario_seed=orders[0][i],
                fixture_seed=orders[1][i],
                residual_battery=float(rngs[2].uniform(*p["train_residual_battery_range"])),
                assignment_seed=int(rngs[3].integers(0, 2**32, dtype=np.uint64)),
            )
            for i in range(count)
        ]
        p["scenario_schedule"][str(seed)] = prefix + suffix
    return p


def check_science(parent, policy, identity, snapshot):
    assert parent["source_sha256"] == identity["sha256"] == canonical_hash(identity["files"])
    assert identity["sha256"] == policy["parent_source_sha256"], "wrong parent source"
    changed = []
    with zipfile.ZipFile(snapshot) as z:
        for name, digest in identity["files"].items():
            data = z.read("uav_thesis/" + name)
            assert hashlib.sha256(data).hexdigest() == digest, f"parent snapshot hash: {name}"
            if name.startswith("src/"):
                current = (ROOT / name).read_bytes()
                # Cross-platform checkout newline conversion only, never scientific edits.
                assert data.replace(b"\r\n", b"\n") == current.replace(b"\r\n", b"\n"), (
                    f"scientific source drift: {name}"
                )
                if data != current:
                    changed.append(name)
    if policy.get("diagnostic"):
        return changed
    expected = dict(
        horizon=150,
        stage_ends=[3000, 6000, 12000, 24000],
        lr=0.0003,
        gamma=0.99,
        replay_capacity=30000,
        batch_size=64,
        warmup=1000,
        update_every=4,
        target_every=1000,
        epsilon_end=0.05,
        epsilon_decay_steps=18000,
        state_dim=111,
        obs_dim=116,
        n_actions=10,
        condition_boundaries=[3000, 4500, 6000, 7500, 9000, 10500, 12000],
        train_pool=list(range(120001, 120041)),
        validation_pool=list(range(130001, 130011)),
    )
    for key, value in expected.items():
        assert parent[key] == value, f"parent protocol mismatch: {key}"
    # 12k is a label boundary; it is not a physical task/replay-clear boundary.
    return changed


def check_state(path, meta, marker, p, policy, arm, seed):
    state = torch.load(path, map_location="cpu", weights_only=False)
    assert STATE_KEYS <= state.keys(), f"not full resume: {STATE_KEYS - state.keys()}"
    c, replay = state["commit"], state["replay"]
    assert c == marker, "checkpoint/JSON commit disagreement"
    assert c["steps"] == policy["parent_steps"]
    assert c["episodes"] == policy["parent_steps"] // p["horizon"]
    assert c["updates"] == policy["parent_updates"]
    assert meta["arm"] == arm and meta["seed"] == seed
    assert meta["source_sha256"] == p["source_sha256"]
    assert meta["config_sha256"] == canonical_hash(meta["resolved_config"])
    assert (meta["state_dim"], meta["obs_dim"], meta["n_actions"]) == (111, 116, 10)
    expected_config = serial(asdict(make_env(condition_for(0, arm, p), p, arm == "C-R").cfg))
    assert meta["resolved_config"] == expected_config, "environment/config drift"
    assert c["current_task"] == serial(task_signature(condition_for(c["steps"], arm, p)))
    assert replay.capacity == p["replay_capacity"] and len(replay) >= p["warmup"]
    assert sum(replay.composition().values()) == len(replay)
    assert 0 <= replay.pos < replay.capacity and replay.rng.getstate()
    if not policy.get("diagnostic"):
        assert c["lineage"] == 6 and len(replay) == replay.pos == 13500
        assert replay.composition() == {"S3d": 1500, "S4": 12000}
        if policy.get("origin") != FRESH_ORIGIN or p["device"].startswith("cuda"):
            assert len(state["cuda_rng"]) > 0, "CUDA parent requires CUDA RNG states"
    for item in replay.data:
        assert item.lineage == c["lineage"] and item.collection_step < c["steps"]
        assert np.isfinite(item.obs).all() and np.isfinite(item.state).all()
        assert all(item.current_action_mask[i, a] for i, a in enumerate(item.actions))
        if arm == "C-N":
            assert not item.current_action_mask[:, 1::2].any()
            assert not item.next_action_mask[:, 1::2].any()
            assert not (item.actions % 2).any()
    assert len({item.collection_step for item in replay.data}) == len(replay)
    assert max(item.collection_step for item in replay.data) == c["steps"] - 1
    assert min(item.collection_step for item in replay.data) == c["steps"] - len(replay)
    probe_env = make_env(condition_for(0, arm, p), p, arm == "C-R")
    probe = BS5CurriculumQMIX(probe_env, lr=p["lr"], gamma=p["gamma"], device="cpu")
    for group in ("agent", "mixer", "target_agent", "target_mixer"):
        assert state[group] and all(torch.isfinite(t).all() for t in state[group].values())
        getattr(probe, group).load_state_dict(state[group], strict=True)
    probe.optimizer.load_state_dict(state["optimizer"])
    assert state["optimizer"]["state"], "missing Adam moments"
    for group in state["optimizer"]["param_groups"]:
        assert group["lr"] == p["lr"]
    for item in state["optimizer"]["state"].values():
        assert int(item["step"]) == c["updates"]
        assert torch.isfinite(item["exp_avg"]).all() and torch.isfinite(item["exp_avg_sq"]).all()
    for param, item in probe.optimizer.state.items():
        assert param.shape == item["exp_avg"].shape == item["exp_avg_sq"].shape
    random.Random().setstate(state["python_rng"])
    np.random.RandomState().set_state(state["numpy_rng"])
    np.random.default_rng().bit_generator.state = state["exploration_rng"]
    random.Random().setstate(replay.rng.getstate())
    torch.Generator().set_state(state["torch_rng"])
    assert all(torch.is_tensor(s) and s.dtype == torch.uint8 for s in state["cuda_rng"])
    return dict(
        steps=c["steps"],
        episodes=c["episodes"],
        updates=c["updates"],
        replay_size=len(replay),
        replay_pos=replay.pos,
        composition=replay.composition(),
        lineage=c["lineage"],
        cuda_rng_devices=len(state["cuda_rng"]),
    )


def preflight(parent, policy):
    check_policy(policy)
    parent = Path(parent)
    missing = [
        str(parent / f)
        for f in ("protocol.json", "source_identity.json", "source_snapshot.zip")
        if not (parent / f).is_file()
    ]
    for name in names(policy):
        missing += [
            str(parent / "runs" / name / f)
            for f in ("meta.json", "commit.json", "resume.pt")
            if not (parent / "runs" / name / f).is_file()
        ]
    report = dict(
        ready=False,
        missing=missing,
        errors=[],
        runtime=runtime(),
        runs={},
        main_steps_executed=0,
        required_runs=names(policy),
    )
    if missing:
        return report
    p = read_json(parent / "protocol.json")
    try:
        report["newline_only_src_differences"] = check_science(
            p, policy, read_json(parent / "source_identity.json"), parent / "source_snapshot.zip"
        )
    except (AssertionError, KeyError, ValueError, OSError, RuntimeError) as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    for name in names(policy):
        try:
            arm, seed = name.split("_seed")
            d = parent / "runs" / name
            report["runs"][name] = check_state(
                d / "resume.pt",
                read_json(d / "meta.json"),
                read_json(d / "commit.json"),
                p,
                policy,
                arm,
                int(seed),
            )
        except (AssertionError, KeyError, ValueError, OSError, RuntimeError) as exc:
            report["errors"].append(f"{name}: {type(exc).__name__}: {exc}")
    report["ready"] = not report["errors"]
    try:
        probe = deepcopy(p)
        probe["parent_execution_environment"] = p["execution_environment"]
        check_runtime(policy, probe, report["runtime"])
        report["runtime_ready"] = True
    except AssertionError as exc:
        report["runtime_ready"] = False
        report["runtime_blocker"] = str(exc)
    return report


def check_runtime(policy, p, actual):
    if policy.get("diagnostic"):
        assert p["scope"].startswith("diagnostic"), "diagnostic policy cannot run main"
        return
    if policy.get("origin") == FRESH_ORIGIN:
        parent_runtime = p["parent_execution_environment"]
        for key in (
            "python",
            "torch",
            "numpy",
            "cuda",
            "cuda_available",
            "gpu",
            "torch_threads",
            "deterministic_algorithms",
            "cudnn_deterministic",
            "cudnn_benchmark",
            "matmul_allow_tf32",
            "cudnn_allow_tf32",
            "cublas_workspace_config",
        ):
            assert actual[key] == parent_runtime[key], f"fresh lineage runtime drift: {key}"
        return
    assert actual["torch"].startswith(policy["required_torch_prefix"]), "requires parent Torch2.11"
    assert actual["numpy"] == policy["required_numpy"], "requires parent NumPy2.1.3"
    assert actual["cuda_available"] and p["device"].startswith("cuda"), (
        "requires CUDA parent runtime"
    )
    parent_runtime = p["parent_execution_environment"]
    for key in (
        "deterministic_algorithms",
        "cudnn_deterministic",
        "cudnn_benchmark",
        "matmul_allow_tf32",
        "cudnn_allow_tf32",
        "cublas_workspace_config",
    ):
        if key in parent_runtime:
            assert parent_runtime[key] == actual[key], f"numerical flag drift: {key}"


def prepare(parent, output, policy, *, evidence=None):
    parent, output = Path(parent).resolve(), Path(output).resolve()
    assert not output.exists(), "new output required; use run --resume on an existing extension"
    assert not output.is_relative_to(parent) and not parent.is_relative_to(output)
    output.mkdir(parents=True)
    timer = Timers(output, "prepare-extension")
    with timer.phase("import_parent"):
        report = preflight(parent, policy)
        atomic_json(output / "preflight.json", report)
        if not report["ready"]:
            timer.flush(status="blocked_no_rollout")
            raise RuntimeError("Missing/incompatible full parent states; see preflight.json")
        identity = source_identity()
        freeze_source(output, identity)
        old = read_json(parent / "protocol.json")
        p = extend_protocol(old, policy, identity)
        p["parent_execution_environment"] = old["execution_environment"]
        p["scope"] = "diagnostic_smoke_not_main" if policy.get("diagnostic") else policy["scope"]
        atomic_json(output / "protocol.json", p)
        refs = dict(
            version=policy["version"],
            parent_protocol_sha256=canonical_hash(old),
            parent_source_sha256=old["source_sha256"],
            current_source_sha256=identity["sha256"],
            rule="same src text (newline conversion allowed); separate authorized runner/config",
            files={},
        )
        refs["origin"] = policy.get("origin", "historical_full_state_continuation")
        refs["source_diff"] = {
            "added": sorted(
                set(identity["files"]) - set(read_json(parent / "source_identity.json")["files"])
            ),
            "changed": sorted(
                k
                for k, v in read_json(parent / "source_identity.json")["files"].items()
                if k in identity["files"] and identity["files"][k] != v
            ),
            "removed": sorted(
                set(read_json(parent / "source_identity.json")["files"]) - set(identity["files"])
            ),
        }
        refs["schedule_generation_runtime"] = runtime()
        refs["runtime_flags_missing_in_parent"] = sorted(
            set(
                (
                    "deterministic_algorithms",
                    "cudnn_deterministic",
                    "cudnn_benchmark",
                    "matmul_allow_tf32",
                    "cudnn_allow_tf32",
                    "cublas_workspace_config",
                )
            )
            - set(old["execution_environment"])
        )
        refs["runtime_limitation"] = (
            "Same versions preferred; GPU/driver/unspecified flags prevent a bitwise continuation guarantee. No flags changed by this runner."
        )

        def copy_file(src, rel):
            dst = output / "parents" / rel
            if dst.exists():
                assert file_hash(dst) == file_hash(src), f"conflicting parent evidence: {rel}"
                return
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            io_count(timer, "copy_parent", [dst])
            digest = file_hash(src)
            assert digest == file_hash(dst)
            refs["files"][rel] = dict(sha256=digest, bytes=dst.stat().st_size)

        for rel in ("protocol.json", "source_identity.json", "source_snapshot.zip"):
            copy_file(parent / rel, rel)
        if policy.get("origin") == FRESH_ORIGIN:
            copy_file(parent / "run_registry.json", "run_registry.json")
            for src in (parent / "timing").glob("*.json"):
                copy_file(src, "timing/" + src.name)
        registry = dict(status="prepared", protocol_sha256=canonical_hash(p), runs={}, attempts=[])
        for name in names(policy):
            d = parent / "runs" / name
            for rel in ("meta.json", "commit.json", "resume.pt"):
                copy_file(d / rel, f"runs/{name}/{rel}")
            c = read_json(d / "commit.json")
            best = d / "checkpoints" / f"{c['best_step']:06d}.pt"
            if best.exists():
                copy_file(best, f"runs/{name}/checkpoints/{best.name}")
            report["runs"][name]["parent_best_weights_available"] = best.exists()
            run = output / "runs" / name
            run.mkdir(parents=True)
            shutil.copyfile(d / "resume.pt", run / "resume.pt")
            io_count(timer, "import_working_state", [run / "resume.pt"])
            atomic_json(run / "commit.json", c)
            meta = read_json(d / "meta.json")
            atomic_json(
                run / "meta.json",
                dict(
                    meta,
                    source_sha256=identity["sha256"],
                    parent_meta_sha256=file_hash(d / "meta.json"),
                    continuation_version=policy["version"],
                ),
            )
            registry["runs"][name] = dict(
                status="imported_pending_tail",
                steps=policy["parent_steps"],
                imported_steps=policy["parent_steps"],
                pending_steps=policy["total_steps"] - policy["parent_steps"],
            )
        # Copy evidence once, retaining bytes; never synthesize absent historical packets.
        for base in dict.fromkeys([parent, Path(evidence).resolve() if evidence else parent]):
            if (base / "protocol.json").exists():
                assert canonical_hash(read_json(base / "protocol.json")) == canonical_hash(old)
            for name in names(policy):
                for sub in (
                    "episodes",
                    "updates",
                    "events",
                    "train_traces",
                    "evaluations",
                    "stage_evaluations",
                ):
                    for src in sorted((base / "runs" / name / sub).rglob("*")):
                        if src.is_file():
                            copy_file(
                                src,
                                f"runs/{name}/{sub}/{src.relative_to(base / 'runs' / name / sub).as_posix()}",
                            )
            table = base / "analysis/validation_episodes.csv"
            if table.exists():
                copy_file(table, "analysis/validation_episodes.csv")
        refs["parent_common_raw_coverage"] = {
            name: sum(
                (
                    output / "parents/runs" / name / "evaluations" / f"{s:06d}" / f"{v}.json.gz"
                ).exists()
                for s in old["common_evaluation_steps"]
                for v in old["validation_pool"]
            )
            for name in names(policy)
        }
        atomic_json(output / "parent_refs.json", refs)
        atomic_json(output / "preflight.json", report)
        atomic_json(output / "run_registry.json", registry)
        atomic_json(
            output / "budget_ledger.json",
            dict(
                imported_steps=policy["parent_steps"] * len(names(policy)),
                new_executed_steps=0,
                planned_new_steps=(policy["total_steps"] - policy["parent_steps"])
                * len(names(policy)),
                attempts=[],
                durable_new_steps=0,
                rollback_steps=0,
            ),
        )
        if policy.get("origin") == FRESH_ORIGIN:
            ledger(output, registry, policy)
    timer.flush(status="prepared_no_rollout")
    return p


def ledger(root, registry, policy):
    attempts = registry["attempts"]
    executed = sum(a["executed_train_steps"] for a in attempts)
    durable = sum(max(0, e["steps"] - policy["parent_steps"]) for e in registry["runs"].values())
    data = dict(
        imported_steps=policy["parent_steps"] * len(registry["runs"]),
        new_executed_steps=executed,
        durable_new_steps=durable,
        rollback_steps=executed - durable,
        attempts=attempts,
        uncertain_unflushed_steps_upper_bound=sum(
            a.get("uncertain_unflushed_steps_upper_bound", 0) for a in attempts
        ),
        count_semantics="exact for caught interruptions; flushed completed episodes lower bound after abrupt kill; uncertainty separate",
        planned_new_steps=(policy["total_steps"] - policy["parent_steps"]) * len(registry["runs"]),
    )
    if policy.get("origin") == FRESH_ORIGIN:
        fresh_prefix = read_json(root / "parents/run_registry.json")
        prefix_executed = sum(a["executed_train_steps"] for a in fresh_prefix["attempts"])
        data.update(
            origin=FRESH_ORIGIN,
            historical_imported_steps=0,
            prefix_executed_steps=prefix_executed,
            tail_executed_steps=executed,
            whole_fresh_executed_steps=prefix_executed + executed,
            whole_fresh_durable_steps=data["imported_steps"] + durable,
            planned_whole_fresh_steps=policy["total_steps"] * len(registry["runs"]),
            prefix_attempts=fresh_prefix["attempts"],
            prefix_uncertain_unflushed_steps_upper_bound=sum(
                a.get("uncertain_unflushed_steps_upper_bound", 0) for a in fresh_prefix["attempts"]
            ),
            whole_fresh_rollback_steps=prefix_executed
            + executed
            - data["imported_steps"]
            - durable,
        )
    atomic_json(root / "budget_ledger.json", data)


def tail(root, p, registry, arm, seed, timer, *, stop_after=None):
    """Only full-task tail. Save full RNG/Adam/replay state without touching lineage."""
    name = f"{arm}_seed{seed}"
    output = root / "runs" / name
    policy = p["continuation"]
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    env = make_env(condition_for(policy["parent_steps"], arm, p), p, arm == "C-R")
    learner = BS5CurriculumQMIX(env, lr=p["lr"], gamma=p["gamma"], device=p["device"])
    exploration = np.random.default_rng()
    with timer.phase("persistence"):
        commit, replay = load_state(output / "resume.pt", learner, exploration)
        io_count(timer, "load_full_state", [output / "resume.pt"])
        for prior in registry["attempts"]:
            if prior["run"] == name and prior["status"] == "running":
                prior.update(
                    status="abrupt_restart",
                    ended_utc=utc_now(),
                    uncertain_unflushed_steps_upper_bound=p["horizon"],
                    reason="process ended without flush; count is lower bound, at most one in-flight episode unaccounted",
                )
        atomic_json(output / "commit.json", commit)
        archive_uncommitted(output, commit)
        index = commit.setdefault("extension_packet_index", {})
        # Also quarantine leftover atomic-write temporaries, including events/*.tmp.
        for sub in ("episodes", "updates", "events", "train_traces", "evaluations"):
            for path in sorted((output / sub).rglob("*")):
                if path.is_file() and path.relative_to(output).as_posix() not in index:
                    target = (
                        output
                        / "aborted_attempts"
                        / f"reconcile_{time.time_ns()}"
                        / path.relative_to(output)
                    )
                    assert path.resolve().is_relative_to(output.resolve()) and not path.is_symlink()
                    target.parent.mkdir(parents=True, exist_ok=True)
                    path.rename(target)
        for rel, entry in index.items():
            path = (output / rel).resolve()
            assert path.is_relative_to(output.resolve())
            assert file_hash(path) == entry["sha256"], f"committed tail packet changed: {rel}"
        atomic_json(output / "packet_index.json", index)
        registry["runs"][name].update(
            steps=commit["steps"],
            status="complete" if commit["steps"] == p["steps_per_run"] else "running",
        )
    attempt = dict(
        run=name,
        from_step=commit["steps"],
        executed_train_steps=0,
        learner_updates=0,
        started_utc=utc_now(),
        status="running",
        runtime=runtime(),
    )
    registry["attempts"].append(attempt)
    atomic_json(root / "run_registry.json", registry)
    meta = read_json(output / "meta.json")

    def packet(rel, data, compressed=False):
        path = output / rel
        assert not path.exists(), f"packet would overwrite: {path}"
        (write_trace if compressed else atomic_json)(path, data)
        io_count(timer, "packet_write", [path])
        index[rel] = dict(sha256=file_hash(path), bytes=path.stat().st_size)

    try:
        while commit["steps"] < p["steps_per_run"]:
            timer.switch("train")
            eid, start = commit["episodes"], commit["steps"]
            c = condition_for(start, arm, p)
            assert c["stage"] == "S4"
            assert commit["current_task"] == serial(task_signature(c)), "tail cannot clear replay"
            spec = p["scenario_schedule"][str(seed)][eid]
            env = make_env(c, p, arm == "C-R", spec["residual_battery"])
            out = env.reset(seed=spec["scenario_seed"], assignment_seed=spec["assignment_seed"])
            geometry = Geometry(env)
            reasons = []
            key = f"S4:{spec['scenario_seed']}"
            if key not in commit["seen"]:
                commit["seen"].append(key)
                reasons.append("first_scenario_in_substage")
            if eid % p["train_trace_every_episodes"] == 0:
                reasons.append("periodic")
            if start == attempt["from_step"]:
                reasons.append("first_after_resume")
            rows, updates = [], []
            for t in range(p["horizon"]):
                epsilon = 1 + (p["epsilon_end"] - 1) * min(
                    commit["steps"] / p["epsilon_decay_steps"], 1
                )
                mask, decision = out["action_mask"].copy(), capture_decision(env)
                actions, q = act(learner, out, epsilon, exploration)
                nxt = env.step(actions)
                replay.add(
                    CurriculumTransition(
                        out["obs"],
                        out["state"],
                        np.asarray(actions),
                        nxt["reward"],
                        nxt["obs"],
                        nxt["state"],
                        nxt["action_mask"].copy(),
                        float(nxt["terminated"] or nxt["truncated"]),
                        mask,
                        "S4",
                        "S4",
                        commit["steps"],
                        commit["lineage"],
                    )
                )
                commit["steps"] += 1
                attempt["executed_train_steps"] += 1
                assert len(replay) >= max(p["warmup"], p["batch_size"]), "unexpected tail warmup"
                if commit["steps"] % p["update_every"] == 0:
                    begin = time.monotonic()
                    update = learner.train_batch(replay.sample(p["batch_size"]))
                    timer.child("update", time.monotonic() - begin)
                    commit["updates"] += 1
                    attempt["learner_updates"] += 1
                    updates.append(
                        dict(
                            step=commit["steps"],
                            update=commit["updates"],
                            epsilon=epsilon,
                            replay_size=len(replay),
                            lineage=commit["lineage"],
                            replay_stage_composition=replay.composition(),
                            target_sync=commit["steps"] % p["target_every"] == 0,
                            **update,
                        )
                    )
                if commit["steps"] % p["target_every"] == 0:
                    learner.update_targets()
                begin = time.monotonic()
                rows.append(
                    record(
                        env,
                        geometry,
                        decision,
                        mask,
                        actions,
                        nxt,
                        p,
                        arm,
                        seed,
                        spec["scenario_seed"],
                        t + 1,
                        commit["steps"],
                        c,
                        q,
                    )
                )
                timer.child("train_shadow_logging", time.monotonic() - begin)
                out = nxt
            assert out["truncated"]
            commit["episodes"] += 1
            for field in ("stage_budgets", "substage_budgets"):
                commit[field]["S4"] = commit[field].get("S4", 0) + p["horizon"]
            with timer.phase("persistence"):
                ep = dict(
                    episode_id=eid,
                    start_step=start,
                    end_step=commit["steps"],
                    stage="S4",
                    substage="S4",
                    condition=c,
                    lineage=commit["lineage"],
                    scenario_seed=spec["scenario_seed"],
                    assignment_seed=spec["assignment_seed"],
                    residual=dict(id=int(env.residual_uav_idx), energy=spec["residual_battery"]),
                    capture_reason=reasons,
                    metrics=episode_metrics(rows),
                    source_sha256=p["source_sha256"],
                    config_sha256=meta["config_sha256"],
                    completion_status="complete",
                )
                packet(f"episodes/{eid:06d}.json", ep)
                packet(f"updates/{eid:06d}.json", updates)
                packet(
                    f"events/{eid:06d}.json",
                    [
                        dict(
                            kind="episode_start",
                            episode=eid,
                            global_env_step=start,
                            capture_reason=reasons,
                            replay_clear=False,
                        )
                    ],
                )
                if reasons:
                    packet(
                        f"train_traces/{eid:06d}.json.gz",
                        dict(trace=rows, metrics=ep["metrics"]),
                        True,
                    )
                # Cheap accounting heartbeat, not checkpoint or accumulated export.
                atomic_json(root / "run_registry.json", registry)
                ledger(root, registry, policy)
            step = commit["steps"]
            if step in p["common_evaluation_steps"]:
                timer.switch("eval")
                ev = evaluate(learner, p, output, arm, seed, step)
                timer.child("eval_shadow_logging", ev["shadow_logging_s"])
                commit["evaluations"].append(ev)
                improved = ev["utility"] > commit["best_utility"]
                if improved:
                    commit.update(best_step=step, best_utility=ev["utility"])
                with timer.phase("persistence"):
                    for path in sorted((output / "evaluations" / f"{step:06d}").iterdir()):
                        index[path.relative_to(output).as_posix()] = dict(
                            sha256=file_hash(path), bytes=path.stat().st_size
                        )
                        io_count(timer, "eval_packet_write", [path])
                    # Embedded index + RNG is authoritative if JSON rename is interrupted.
                    if step == p["steps_per_run"]:
                        assert commit["updates"] == policy["final_updates"], (
                            "final update budget mismatch"
                        )
                    save_state(output, learner, replay, exploration, commit)
                    io_count(
                        timer,
                        "checkpoint_write",
                        [
                            output / "resume.pt",
                            output / "commit.json",
                            output / "checkpoints" / f"{step:06d}.pt",
                        ],
                    )
                    if improved:
                        atomic_torch(
                            output / "best.pt",
                            dict(
                                agent=learner.agent.state_dict(),
                                mixer=learner.mixer.state_dict(),
                                step=step,
                            ),
                        )
                    atomic_json(output / "packet_index.json", index)
                    registry["runs"][name].update(
                        steps=step, status="complete" if step == p["steps_per_run"] else "running"
                    )
                    atomic_json(root / "run_registry.json", registry)
                    ledger(root, registry, policy)
                print(
                    json.dumps(
                        dict(
                            run=name,
                            step=step,
                            utility=ev["utility"],
                            best_step=commit["best_step"],
                            updates=commit["updates"],
                        )
                    ),
                    flush=True,
                )
                if stop_after == step:
                    raise KeyboardInterrupt("bounded diagnostic interruption")
        assert commit["updates"] == policy["final_updates"], "final update budget mismatch"
        attempt.update(status="complete", ended_utc=utc_now())
    except BaseException as exc:
        durable = read_json(output / "commit.json")
        attempt.update(
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            reason=f"{type(exc).__name__}: {exc}",
            ended_utc=utc_now(),
            durable_step=durable["steps"],
        )
        registry["runs"][name].update(status="incomplete", steps=durable["steps"])
        registry["status"] = "incomplete"
        raise
    finally:
        with timer.phase("persistence"):
            atomic_json(root / "run_registry.json", registry)
            ledger(root, registry, policy)
        timer.flush(run=name, status=attempt["status"])


def run(root, *, resume=False):
    root = Path(root).resolve()
    p, registry = read_json(root / "protocol.json"), read_json(root / "run_registry.json")
    assert source_identity()["sha256"] == p["source_sha256"], "extension source changed"
    check_policy(p["continuation"])
    assert registry["protocol_sha256"] == canonical_hash(p)
    assert set(registry["runs"]) == set(names(p["continuation"]))
    torch.set_num_threads(p["torch_threads"])
    check_runtime(p["continuation"], p, runtime())
    refs = read_json(root / "parent_refs.json")
    for rel, item in refs["files"].items():
        # Resume states/protocol/source checked each invocation; raw packet checks deferred to final analysis.
        if not rel.endswith(".json.gz"):
            assert file_hash(root / "parents" / rel) == item["sha256"], f"parent changed: {rel}"
    with experiment_lock(root):
        for name in names(p["continuation"]):
            entry = registry["runs"][name]
            assert resume or entry["status"] == "imported_pending_tail", "use --resume"
            if entry["status"] != "complete":
                arm, seed = name.split("_seed")
                timer = Timers(root, f"tail_{name}", p["device"])
                tail(root, p, registry, arm, int(seed), timer)
                if all(e["status"] == "complete" for e in registry["runs"].values()):
                    registry["status"] = "complete"
                    atomic_json(root / "run_registry.json", registry)
                finalize(root)
        registry["status"] = "complete"
        atomic_json(root / "run_registry.json", registry)
        finalize(root)


def finalize(root):
    """Idempotent per durable snapshot, no analysis/export inside the episode loop."""
    from scripts.analysis.analyze_bs5_curriculum_extend import analyze, export_bundles

    root = Path(root).resolve()
    registry = read_json(root / "run_registry.json")
    token = canonical_hash(
        dict(
            protocol=read_json(root / "protocol.json"),
            commits={n: read_json(root / "runs" / n / "commit.json") for n in registry["runs"]},
        )
    )
    receipt = root / "exports" / f"receipt_{token}.json"
    if receipt.exists():
        prior = read_json(receipt)
        if all(
            (root / item["path"]).exists() and file_hash(root / item["path"]) == item["sha256"]
            for item in prior["artifacts"]
        ):
            return prior
    timer = Timers(root, "finalize")
    analysis = root.parent / f"{root.name}_analysis_{token[:12]}"
    with timer.phase("analysis"):
        analyze(
            root,
            analysis,
            require_complete=all(e["status"] == "complete" for e in registry["runs"].values()),
        )
    timer.flush(status="before_export")
    with timer.phase("export"):
        artifacts = export_bundles(root, analysis, token)
        io_count(timer, "export_write", [root / a["path"] for a in artifacts])
    timer.flush(status="export_closed")
    result = dict(snapshot=token, artifacts=artifacts, closing_timer=read_json(timer.path))
    atomic_json(receipt, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["preflight", "prepare-extension", "run", "analyze", "export"]
    )
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--analysis-output", type=Path)
    args = parser.parse_args()
    if args.command == "preflight":
        report = preflight(args.parent, read_json(args.config))
        atomic_json(args.output / "preflight.json", report)
        print(json.dumps(report, indent=2))
        return 0 if report["ready"] else 2
    if args.command == "prepare-extension":
        prepare(args.parent, args.output, read_json(args.config), evidence=args.evidence)
    elif args.command == "run":
        run(args.output, resume=args.resume)
    elif args.command == "export":
        finalize(args.output)
    else:
        from scripts.analysis.analyze_bs5_curriculum_extend import analyze

        assert args.analysis_output, "--analysis-output outside evidence required"
        analyze(args.output, args.analysis_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
