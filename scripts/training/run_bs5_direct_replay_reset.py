"""Two paired direct/full-task runs, six replay clears, synchronous deferred exports."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
import time
import zipfile

PROCESS_ENTRY = time.monotonic()  # module entry, before NumPy/Torch/scientific imports

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from scripts.analysis.bs5_bundle import atomic_json as bundle_atomic_json
from scripts.analysis.bs5_bundle import canonical_hash, file_hash, read_json
from scripts.analysis.analyze_bs5_curriculum_bundle import episode_metrics
from scripts.training.run_bs5_curriculum_pilot import act, make_env, record, protocol_new
from scripts.training.run_bs5_launch_ab import (
    capture_decision,
    experiment_lock,
    freeze_source,
    load_state,
    save_state,
    serial,
    source_identity,
    utc_now,
    weights_hash,
    write_trace as bundle_write_trace,
)
from src.envs.bs5_curriculum_env import condition_for
from src.envs.bs5_launch_ab_env import Geometry
from src.rl.bs5_curriculum_qmix import BS5CurriculumQMIX
from src.rl.bs5_curriculum_replay import CurriculumReplay, CurriculumTransition

ARM = "D-R-reset"
RESET_STEPS = [3000, 4500, 6000, 7500, 9000, 10500]
WINDOWS_IO_ATTEMPTS = 6 if os.name == "nt" else 1


def retry_atomic_io(operation, *args):
    # Cloud-sync/antivirus readers can briefly lock os.replace on Windows.
    # Retry only permission errors, bounded to 1.55 seconds; never ignore denial.
    for attempt in range(WINDOWS_IO_ATTEMPTS):
        try:
            return operation(*args)
        except PermissionError:
            if attempt + 1 == WINDOWS_IO_ATTEMPTS:
                raise
            time.sleep(0.05 * 2**attempt)


def atomic_json(path, data):
    return retry_atomic_io(bundle_atomic_json, path, data)


def write_trace(path, data):
    return retry_atomic_io(bundle_write_trace, path, data)


def update_schedule(p, end=None):
    """Compute expectations from collection and buffer warmup, independently of learner."""
    size, lineage, rows = 0, 0, []
    for before in range(p["steps_per_run"] if end is None else end):
        if before in p["reset_steps"]:
            size, lineage = 0, lineage + 1
        size = min(p["replay_capacity"], size + 1)
        step = before + 1
        if size >= max(p["warmup"], p["batch_size"]) and step % p["update_every"] == 0:
            rows.append((step, lineage, size))
    return rows


def resolved_protocol(config, identity, baseline, *, smoke=False, device=None):
    assert config["allow_relay"] is True and config["training_mode"] == "direct"
    assert config["training_seeds"] == [551, 552]
    assert config["reset_steps"] == RESET_STEPS and config["steps_per_run"] == 24000
    old = baseline["protocol"]
    assert old["source_sha256"] == baseline["source_identity"]["sha256"]
    scientific = {
        n: h for n, h in baseline["source_identity"]["files"].items() if n.startswith("src/")
    }
    assert all(identity["files"].get(n) == h for n, h in scientific.items()), (
        "scientific source changed"
    )
    p = deepcopy(old)
    if smoke:
        # Reuse only the audited schedule generator; fresh debug pools and seeds.
        cfg = read_json(ROOT / "configs/bs5_curriculum_fixed_v1.json")
        p = protocol_new(cfg, identity, smoke=True)
    p.update(
        version=config["version"],
        experiment=config["version"],
        arms=[ARM],
        allow_relay=True,
        training_mode="direct",
        variant="replay_reset",
        source_sha256=identity["sha256"],
        baseline_sha256=canonical_hash(baseline),
        historical_source_sha256=old["source_sha256"],
        reset_steps=[48, 72, 96, 120, 144, 168] if smoke else RESET_STEPS,
        common_evaluation_steps=list(range(0, 385, 48))
        if smoke
        else config["common_evaluation_steps"],
        export_policy=config["export_policy"],
        device=device or config["device"],
        torch_threads=config["torch_threads"],
        scope="smoke_not_main" if smoke else "paired_validation_pilot",
        replay_rule="initial seed+20000; clear k uses seed+20000+k, source C-R physical task changes",
        checkpoint_rule="before_clear durable, then after_clear durable; authoritative resume embeds applied marker",
        max_rollback_steps=48 if smoke else 3000,
        timer_definition="exclusive synchronized phase wall; packet logging inside train/eval; update and shadow are children; process active excludes notebook idle",
        baseline_verification_scope=baseline["verification_scope"],
        execution_environment=dict(
            python=sys.version,
            torch=torch.__version__,
            numpy=np.__version__,
            device=device or config["device"],
        ),
    )
    p["evaluation_steps"] = p["common_evaluation_steps"]
    assert p["common_evaluation_steps"] == list(range(0, p["steps_per_run"] + 1, p["eval_every"]))
    assert all(x % p["horizon"] == 0 for x in p["reset_steps"] + p["evaluation_steps"])
    counts = [sum(lineage == k for _, lineage, _ in update_schedule(p)) for k in range(7)]
    if not smoke:
        assert counts == [501, 126, 126, 126, 126, 126, 3126]
        assert p["scenario_schedule"] == old["scenario_schedule"]
    p["expected_updates_by_lineage"] = counts
    p["expected_updates"] = sum(counts)
    p["full_condition"] = condition_for(0, "D-R", p)
    p["training_conditions"] = {
        ARM: [dict(start=0, end=p["steps_per_run"], condition=p["full_condition"])]
    }
    for key in list(p):
        if (
            key.startswith("gate_")
            or key.startswith("stage_validation")
            or key
            in {
                "stage_ends",
                "condition_boundaries",
                "stage_validation_pool",
                "fixture_pool",
                "fixture_scenario_seeds",
            }
        ):
            p.pop(key)
    return p


class Timers:
    """Switch phases, never add nested outer wall intervals twice."""

    def __init__(self, root, operation, device="cpu", start_at=None):
        self.root, self.device = Path(root), device
        self.started = start_at if start_at is not None else time.monotonic()
        self.tick = time.monotonic()
        self.current = "other"
        self.data = dict(
            operation=operation,
            started_utc=utc_now(),
            outer={"setup": self.tick - self.started},
            children={},
            child_calls={},
            calls={},
            io={},
            scope="one active invocation; CLI includes module imports from PROCESS_ENTRY; test/function calls start at invocation; no notebook idle; final timing JSON write/process teardown excluded",
            children_scope="CPU call wall, including blocking tensor reads; no extra CUDA synchronize per update/slot; parent phases synchronized",
            cuda_boundary_sync=device.startswith("cuda"),
        )
        self.path = self.root / "timing" / f"{time.time_ns()}_{os.getpid()}.json"

    def switch(self, phase):
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        now = time.monotonic()
        outer = self.data["outer"]
        outer[self.current] = outer.get(self.current, 0.0) + now - self.tick
        self.tick, self.current = now, phase

    @contextmanager
    def phase(self, name):
        previous = self.current
        self.switch(name)
        self.data["calls"][name] = self.data["calls"].get(name, 0) + 1
        try:
            yield
        finally:
            self.switch(previous)

    def child(self, name, elapsed):
        self.data["children"][name] = self.data["children"].get(name, 0.0) + elapsed
        self.data["child_calls"][name] = self.data["child_calls"].get(name, 0) + 1

    def flush(self, **extra):
        self.switch(self.current)
        self.data.update(extra, measured_wall_s=self.tick - self.started, ended_utc=utc_now())
        self.data["residual_wall_s"] = self.data["measured_wall_s"] - sum(
            self.data["outer"].values()
        )
        atomic_json(self.path, self.data)


def packet(output, relative, data, index, *, compressed=False):
    path = output / relative
    assert not path.exists(), f"immutable packet exists: {path}"
    (write_trace if compressed else atomic_json)(path, data)
    index[relative] = dict(sha256=file_hash(path), bytes=path.stat().st_size)
    atomic_json(output / "packet_index.json", index)


def reconcile(output, commit):
    """Authoritative embedded packet set wins after any interrupted JSON/checkpoint replace."""
    keep = commit["packet_index"]
    orphan = []
    for folder in ("episodes", "updates", "train_traces", "events", "evaluations"):
        orphan += [
            f
            for f in (output / folder).rglob("*")
            if f.is_file() and f.relative_to(output).as_posix() not in keep
        ]
    orphan += [f for f in (output / "checkpoints").glob("*.pt") if int(f.stem) > commit["steps"]]
    destination = output / "aborted_attempts" / str(time.time_ns())
    for path in orphan:
        assert path.resolve().is_relative_to(output.resolve()) and not path.is_symlink()
        target = destination / path.relative_to(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        path.rename(target)
    for name, info in keep.items():
        assert file_hash(output / name) == info["sha256"], f"committed packet mutated: {name}"
    atomic_json(output / "packet_index.json", keep)
    atomic_json(output / "commit.json", commit)
    atomic_json(output / "reset_ledger.json", commit["resets"])
    return deepcopy(keep)


def initialize(p, seed, baseline):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    env = make_env(p["full_condition"], p, p["allow_relay"])
    learner = BS5CurriculumQMIX(env, lr=p["lr"], gamma=p["gamma"], device=p["device"])
    digest = weights_hash(learner)
    if p["scope"] != "smoke_not_main":
        expected = baseline["metas"][f"D-R_seed{seed}"]
        assert digest == expected["initial_weights_sha256"], (
            f"Initial weights mismatch seed{seed}: {digest}; baseline {expected['initial_weights_sha256']}. "
            "Use historical PyTorch 2.11.0 runtime; no main rollout executed."
        )
        assert serial(asdict(env.cfg)) == expected["resolved_config"], "full-target config differs"
    return env, learner


def preserve_hash(learner, exploration):
    def digest(value):
        h = hashlib.sha256()

        def visit(x):
            if torch.is_tensor(x):
                h.update(str((x.dtype, tuple(x.shape))).encode())
                h.update(x.detach().cpu().numpy().tobytes())
            elif isinstance(x, np.ndarray):
                h.update(str((x.dtype, x.shape)).encode())
                h.update(x.tobytes())
            elif isinstance(x, dict):
                for k in sorted(x, key=str):
                    h.update(str(k).encode())
                    visit(x[k])
            elif isinstance(x, (tuple, list)):
                for v in x:
                    visit(v)
            else:
                h.update(repr(x).encode())

        visit(value)
        return h.hexdigest()

    return dict(
        weights=weights_hash(learner),
        targets=digest([learner.target_agent.state_dict(), learner.target_mixer.state_dict()]),
        optimizer=digest(learner.optimizer.state_dict()),
        exploration=canonical_hash(exploration.bit_generator.state),
        global_rng=digest(
            [
                random.getstate(),
                np.random.get_state(),
                torch.get_rng_state(),
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            ]
        ),
    )


def apply_reset(p, seed, learner, replay, exploration, commit):
    step = commit["steps"]
    if step not in p["reset_steps"] or step in commit["applied_reset_steps"]:
        return replay
    assert commit["reset_phase"] == "before_clear"
    before = preserve_hash(learner, exploration)
    discarded = len(replay)
    commit["lineage"] += 1
    replay = CurriculumReplay(p["replay_capacity"], seed + 20000 + commit["lineage"])
    after = preserve_hash(learner, exploration)
    assert before == after
    commit["applied_reset_steps"].append(step)
    commit["resets"][-1].update(
        kind="reset_applied",
        requested=True,
        discarded=discarded,
        lineage=commit["lineage"],
        replay_rng_seed=seed + 20000 + commit["lineage"],
        preserved_before=before,
        preserved_after=after,
        epsilon_global_step=step,
        target_schedule="global step unchanged",
        applied_exactly_once=True,
    )
    commit["reset_phase"] = "after_clear"
    return replay


def evaluate(learner, p, output, seed, step, index, timers):
    means = []
    env = make_env(p["full_condition"], p, p["allow_relay"], p["validation_residual_battery"])
    geometry = Geometry(env)
    previous = learner.agent.training
    learner.agent.eval()
    try:
        for scenario in p["validation_pool"]:
            out = env.reset(seed=scenario, assignment_seed=scenario, diagnostic=True)
            residual = dict(
                id=int(env.residual_uav_idx), energy=float(env.battery[env.residual_uav_idx])
            )
            rows = []
            rng = np.random.default_rng(np.random.SeedSequence([scenario, 0x4556414C]))
            for t in range(p["horizon"]):
                mask, decision = out["action_mask"].copy(), capture_decision(env)
                actions, q = act(learner, out, 0.0, rng)
                nxt = env.step(actions)
                began = time.monotonic()
                rows.append(
                    record(
                        env,
                        geometry,
                        decision,
                        mask,
                        actions,
                        nxt,
                        p,
                        ARM,
                        seed,
                        scenario,
                        t + 1,
                        step,
                        p["full_condition"],
                        q,
                        evaluation=True,
                        kind="full",
                    )
                )
                timers.child("eval_shadow_oracle_record", time.monotonic() - began)
                out = nxt
            metrics = episode_metrics(rows)
            packet(
                output,
                f"evaluations/{step:06d}/{scenario}.json.gz",
                dict(
                    trace=rows,
                    metrics=metrics,
                    initial_residual=residual,
                    condition=p["full_condition"],
                    kind="full",
                ),
                index,
                compressed=True,
            )
            means.append(metrics["utility"])
    finally:
        learner.agent.train(previous)
    result = dict(
        step=step,
        utility=float(np.mean(means)),
        slots=p["horizon"] * len(means),
        eligible_for_best=True,
    )
    packet(output, f"evaluations/{step:06d}/summary.json", result, index)
    return result


def durable(output, learner, replay, exploration, commit, index, timers, registry, root, name):
    with timers.phase("persistence"):
        commit["packet_index"] = deepcopy(index)
        save_state(output, learner, replay, exploration, commit)
        written = sum(
            (output / relative).stat().st_size
            for relative in ("resume.pt", "commit.json", f"checkpoints/{commit['steps']:06d}.pt")
        )
        timers.data["io"]["checkpoint_bytes_written"] = (
            timers.data["io"].get("checkpoint_bytes_written", 0) + written
        )
        timers.data["io"].setdefault("committed_packet_bytes_by_run", {})[name] = sum(
            info["bytes"] for info in index.values()
        )
        atomic_json(output / "reset_ledger.json", commit["resets"])
        registry["runs"][name] = dict(
            status="complete"
            if commit["steps"] == read_json(root / "protocol.json")["steps_per_run"]
            else "running",
            steps=commit["steps"],
        )
        registry["status"] = (
            "complete"
            if all(v["status"] == "complete" for v in registry["runs"].values())
            else "incomplete"
        )
        atomic_json(root / "run_registry.json", registry)
    timers.flush(last_durable_step=commit["steps"], run=name)


def run_one(root, p, registry, seed, timers, *, resume=False, stop_at=None, cadence_hook=None):
    """stop_at=(step, before_clear/after_clear/committed) is a bounded resume test hook."""
    name = f"{ARM}_seed{seed}"
    if registry["runs"][name]["status"] == "complete":
        return
    output = root / "runs" / name
    output.mkdir(parents=True, exist_ok=True)
    with timers.phase("setup"):
        baseline = read_json(root / "baseline_refs.json")
        env, learner = initialize(p, seed, baseline)
        exploration = np.random.default_rng(np.random.SeedSequence([seed, 0x504F4C49]))
        replay = CurriculumReplay(p["replay_capacity"], seed + 20000)
        meta = dict(
            run=name,
            arm=ARM,
            seed=seed,
            allow_relay=True,
            training_mode="direct",
            variant="replay_reset",
            initial_weights_sha256=weights_hash(learner),
            source_sha256=p["source_sha256"],
            resolved_config=serial(asdict(env.cfg)),
            state_dim=env.state_dim,
            obs_dim=env.obs_dim,
            n_actions=env.n_actions,
        )
        meta["config_sha256"] = canonical_hash(meta["resolved_config"])
        if (output / "meta.json").exists():
            assert read_json(output / "meta.json") == meta, "run identity drift"
        else:
            atomic_json(output / "meta.json", meta)
        commit = dict(
            steps=0,
            episodes=0,
            updates=0,
            best_step=0,
            best_utility=-1.0,
            lineage=0,
            applied_reset_steps=[],
            reset_phase="initialized",
            resets=[],
            evaluations=[],
            seen=[],
            packet_index={},
        )
        if (output / "resume.pt").exists():
            assert resume, "Use --resume"
            commit, replay = load_state(output / "resume.pt", learner, exploration)
        index = reconcile(output, commit)
        # A crash can replace resume.pt before commit.json/registry. Repair even
        # when the authoritative state is already the final step (no loop runs).
        registry["runs"][name] = dict(
            status="complete" if commit["steps"] == p["steps_per_run"] else "running",
            steps=commit["steps"],
        )
        registry["status"] = (
            "complete"
            if all(e["status"] == "complete" for e in registry["runs"].values())
            else "incomplete"
        )
        atomic_json(root / "run_registry.json", registry)
    attempt = dict(
        run=name,
        from_step=commit["steps"],
        started_utc=utc_now(),
        status="running",
        train_steps=0,
        updates=0,
        event="resume_attempt" if resume and (output / "resume.pt").exists() else "fresh_attempt",
        reset_phase=commit["reset_phase"],
        max_rollback_steps=p["max_rollback_steps"],
    )
    registry["attempts"].append(attempt)
    try:
        if not commit["evaluations"]:
            with timers.phase("eval"):
                ev = evaluate(learner, p, output, seed, 0, index, timers)
            commit.update(evaluations=[ev], best_utility=ev["utility"])
            durable(
                output, learner, replay, exploration, commit, index, timers, registry, root, name
            )
        while commit["steps"] < p["steps_per_run"]:
            if (
                commit["steps"] in p["reset_steps"]
                and commit["steps"] not in commit["applied_reset_steps"]
            ):
                with timers.phase("persistence"):
                    replay = apply_reset(p, seed, learner, replay, exploration, commit)
                durable(
                    output,
                    learner,
                    replay,
                    exploration,
                    commit,
                    index,
                    timers,
                    registry,
                    root,
                    name,
                )
                if stop_at == (commit["steps"], "after_clear"):
                    raise KeyboardInterrupt("bounded after-clear test")
            with timers.phase("train_loop"):
                eid, start = commit["episodes"], commit["steps"]
                spec = p["scenario_schedule"][str(seed)][eid]
                env = make_env(p["full_condition"], p, p["allow_relay"])
                env.cfg.residual_initial_battery = spec["residual_battery"]
                out = env.reset(seed=spec["scenario_seed"], assignment_seed=spec["assignment_seed"])
                residual = dict(id=int(env.residual_uav_idx), energy=spec["residual_battery"])
                if p["scope"] != "smoke_not_main":
                    reference = baseline["training_references"][str(seed)][eid]
                    assert residual == reference["residual"], "training residual pairing mismatch"
                geometry = Geometry(env)
                reasons = []
                if spec["scenario_seed"] not in commit["seen"]:
                    reasons.append("first_scenario")
                    commit["seen"].append(spec["scenario_seed"])
                if eid % p["train_trace_every_episodes"] == 0:
                    reasons.append("periodic10episodes")
                if start in [0] + p["reset_steps"]:
                    reasons.append("first_after_reset_or_initial")
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
                            "D",
                            "D",
                            commit["steps"],
                            commit["lineage"],
                        )
                    )
                    commit["steps"] += 1
                    attempt["train_steps"] += 1
                    if (
                        len(replay) >= max(p["warmup"], p["batch_size"])
                        and commit["steps"] % p["update_every"] == 0
                    ):
                        began = time.monotonic()
                        update = learner.train_batch(replay.sample(p["batch_size"]))
                        timers.child("train_update", time.monotonic() - began)
                        commit["updates"] += 1
                        attempt["updates"] += 1
                        updates.append(
                            dict(
                                step=commit["steps"],
                                update=commit["updates"],
                                epsilon=epsilon,
                                replay_size=len(replay),
                                replay_stage_composition=replay.composition(),
                                lineage=commit["lineage"],
                                target_sync=commit["steps"] % p["target_every"] == 0,
                                **update,
                            )
                        )
                    if commit["steps"] % p["target_every"] == 0:
                        learner.update_targets()
                    rows.append(
                        record(
                            env,
                            geometry,
                            decision,
                            mask,
                            actions,
                            nxt,
                            p,
                            ARM,
                            seed,
                            spec["scenario_seed"],
                            t + 1,
                            commit["steps"],
                            p["full_condition"],
                            q,
                        )
                    )
                    out = nxt
                assert out["truncated"]
                if p["scope"] != "smoke_not_main" and "exogenous" in reference:
                    assert [r["exogenous_sha256"] for r in rows] == reference["exogenous"], (
                        "training exogenous pairing mismatch"
                    )
                ep = dict(
                    episode_id=eid,
                    start_step=start,
                    end_step=commit["steps"],
                    lineage=commit["lineage"],
                    scenario_seed=spec["scenario_seed"],
                    assignment_seed=spec["assignment_seed"],
                    residual=residual,
                    condition=p["full_condition"],
                    source_sha256=p["source_sha256"],
                    config_sha256=meta["config_sha256"],
                    capture_reason=reasons,
                    metrics=episode_metrics(rows),
                    completion_status="complete",
                    trajectory_sha256=canonical_hash(
                        [[r["raw_actions"], r["reward"]] for r in rows]
                    ),
                )
                packet(output, f"episodes/{eid:06d}.json", ep, index)
                packet(output, f"updates/{eid:06d}.json", updates, index)
                packet(
                    output,
                    f"events/{eid:06d}.json",
                    dict(
                        kind="episode_complete",
                        step=commit["steps"],
                        terminal=True,
                        starts=rows[0]["decision"]["positions"],
                        capture_reason=reasons,
                        lineage=commit["lineage"],
                    ),
                    index,
                )
                if reasons:
                    packet(
                        output,
                        f"train_traces/{eid:06d}.json.gz",
                        dict(trace=rows, metrics=ep["metrics"]),
                        index,
                        compressed=True,
                    )
                commit["episodes"] += 1
            step = commit["steps"]
            if step in p["evaluation_steps"]:
                with timers.phase("eval"):
                    ev = evaluate(learner, p, output, seed, step, index, timers)
                commit["evaluations"].append(ev)
                if ev["utility"] > commit["best_utility"]:
                    commit.update(best_step=step, best_utility=ev["utility"])
            if step in p["evaluation_steps"] or step in p["reset_steps"]:
                if step in p["reset_steps"]:
                    commit["reset_phase"] = "before_clear"
                    commit["resets"].append(dict(kind="reset_requested", step=step))
                else:
                    commit["reset_phase"] = "committed"
                commit["final_weights_sha256"] = weights_hash(learner)
                durable(
                    output,
                    learner,
                    replay,
                    exploration,
                    commit,
                    index,
                    timers,
                    registry,
                    root,
                    name,
                )
                print(
                    json.dumps(
                        dict(
                            run=name,
                            step=step,
                            updates=commit["updates"],
                            reset_phase=commit["reset_phase"],
                            utility=commit["evaluations"][-1]["utility"],
                        )
                    ),
                    flush=True,
                )
                if stop_at == (step, "before_clear") or stop_at == (step, "committed"):
                    raise KeyboardInterrupt("bounded durable-boundary test")
                if cadence_hook:
                    cadence_hook()
        assert commit["updates"] == p["expected_updates"]
        attempt["status"] = "complete"
    except BaseException as exc:
        attempt.update(
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            reason=f"{type(exc).__name__}: {exc}",
        )
        # Persist reason only; latest embedded durable state remains authoritative.
        durable_commit = (
            read_json(output / "commit.json")
            if (output / "commit.json").exists()
            else dict(steps=0)
        )
        registry["runs"][name] = dict(status="incomplete", steps=durable_commit["steps"])
        registry["status"] = "incomplete"
        raise
    finally:
        attempt["ended_utc"] = utc_now()
        atomic_json(root / "run_registry.json", registry)
        timers.flush(run=name, status=attempt["status"])


def prepare(root, config_path, *, smoke=False, device=None, start_at=None):
    started = start_at if start_at is not None else time.monotonic()
    assert not root.exists(), "Preserve prior results: choose a new --outdir"
    config = read_json(config_path)
    baseline = read_json(ROOT / config["baseline"])
    identity = source_identity()
    p = resolved_protocol(config, identity, baseline, smoke=smoke, device=device)
    torch.set_num_threads(p["torch_threads"])
    # Fail before creating main evidence if runtime cannot reproduce paired initialization.
    for seed in p["training_seeds"]:
        initialize(p, seed, baseline)
    root.mkdir(parents=True)
    timers = Timers(root, "prepare", p["device"], start_at=started)
    with timers.phase("setup"):
        freeze_source(root, identity)
        timers.data["io"].update(
            source_members=len(identity["files"]),
            source_hash_count=len(identity["files"]),
            source_snapshot_bytes=(root / "source_snapshot.zip").stat().st_size,
        )
        atomic_json(root / "protocol.json", p)
        atomic_json(root / "baseline_refs.json", baseline)
        atomic_json(
            root / "source_diff.json",
            dict(
                historical=baseline["source_identity"]["sha256"],
                current=identity["sha256"],
                changed=[
                    n
                    for n, h in identity["files"].items()
                    if baseline["source_identity"]["files"].get(n) != h
                ],
                scientific_src_unchanged=True,
            ),
        )
        atomic_json(
            root / "schema.json",
            dict(
                version=1,
                service_phase="after_movement_before_energy_and_ue_mobility",
                reward="mean(min(rate/demand,1))",
                throughput="360*utility Mbps; not independent evidence",
                empty_relay="empty active Relay UAV-slots / active Relay UAV-slots",
                jain="per-UE episode-mean rates",
                tail="last three common evaluations",
                oracle="slots1/76 diagnostics only",
                checkpoint="checkpoints/*.pt inference; resume.pt optimizer/replay/RNG authoritative",
                baseline_scope=baseline["verification_scope"],
            ),
        )
        atomic_json(
            root / "run_registry.json",
            dict(
                status="prepared",
                created_utc=utc_now(),
                protocol_sha256=canonical_hash(p),
                runs={
                    f"{ARM}_seed{s}": dict(status="not_started", steps=0)
                    for s in p["training_seeds"]
                },
                attempts=[],
            ),
        )
    timers.flush(status="prepared", main_training_steps_executed=0)
    print(
        json.dumps(
            dict(
                event="PREPARED",
                scope=p["scope"],
                runs=list(p["training_seeds"]),
                updates_per_run=p["expected_updates"],
                eval_slots=len(p["training_seeds"])
                * len(p["evaluation_steps"])
                * len(p["validation_pool"])
                * p["horizon"],
            )
        ),
        flush=True,
    )


def zip_snapshot(destination, files, base):
    temporary = destination.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w") as archive:
        for path in sorted(files):
            compression = (
                zipfile.ZIP_STORED if path.suffix in {".gz", ".zip"} else zipfile.ZIP_DEFLATED
            )
            archive.write(path, path.relative_to(base).as_posix(), compress_type=compression)
    with zipfile.ZipFile(temporary) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == len(set(archive.namelist()))
    temporary.replace(destination)


def seed_shards(audit, exports, prefix, limit=480 * 1024**2):
    """Large raw evidence: merge seed shards before auditing; each is explicitly partial."""
    if audit.stat().st_size <= limit:
        return []
    shards = []
    with zipfile.ZipFile(audit) as original:
        names = original.namelist()
        runs = sorted({n.split("/")[1] for n in names if n.startswith("runs/")})
        common = [n for n in names if not n.startswith("runs/")]
        for run in runs:
            selected = common + [n for n in names if n.startswith(f"runs/{run}/")]
            destination = exports / f"{prefix}_audit_{run}.zip"
            temporary = destination.with_suffix(".zip.tmp")
            with zipfile.ZipFile(temporary, "w") as archive:
                for name in selected:
                    archive.writestr(
                        name,
                        original.read(name),
                        compress_type=zipfile.ZIP_STORED
                        if name.endswith(".gz")
                        else zipfile.ZIP_DEFLATED,
                    )
            with zipfile.ZipFile(temporary) as archive:
                assert archive.testzip() is None and len(archive.namelist()) == len(
                    set(archive.namelist())
                )
            temporary.replace(destination)
            shards.append(
                dict(
                    name=destination.name,
                    sha256=file_hash(destination),
                    bytes=destination.stat().st_size,
                    partial=True,
                    rule="Extract ALL seed shards into one folder, then verify the full SHA256SUMS.json. One shard is not a complete audit.",
                )
            )
    return shards


def export_bundles(root, timers=None, *, start_at=None):
    from scripts.analysis.analyze_bs5_direct_replay_reset_bundle import analyze

    own = timers is None
    timers = timers or Timers(root, "manual_export", start_at=start_at)
    assert source_identity()["sha256"] == read_json(root / "protocol.json")["source_sha256"], (
        "export using the frozen source snapshot"
    )
    registry = read_json(root / "run_registry.json")
    commits = {
        name: read_json(root / "runs" / name / "commit.json")
        for name, entry in registry["runs"].items()
        if entry["status"] != "not_started"
    }
    snapshot = canonical_hash(dict(protocol=read_json(root / "protocol.json"), commits=commits))
    exports = root / "exports"
    exports.mkdir(exist_ok=True)
    previous = read_json(exports / "latest.json") if (exports / "latest.json").exists() else {}
    # Verify bytes even when returning an unchanged snapshot; do not trust cached hashes.
    with timers.phase("analysis"):
        analysis_result = analyze(root, root / "analysis", check_manifest=False)
        for key, value in (
            ("analysis_packet_bytes_read", analysis_result["verified_packet_bytes"]),
            ("analysis_packet_hash_count", analysis_result["verified_packet_count"]),
        ):
            timers.data["io"][key] = timers.data["io"].get(key, 0) + value
    if previous.get("snapshot_sha256") == snapshot:
        for name, digest in previous["bundles"].items():
            assert file_hash(exports / name) == digest, "export mutated"
        if own:
            timers.flush(status="unchanged_snapshot_verified")
        return previous
    began = time.monotonic()
    with timers.phase("export"), tempfile.TemporaryDirectory(prefix="bs5-reset-") as temporary:
        stage = Path(temporary)
        static = [
            "protocol.json",
            "baseline_refs.json",
            "source_identity.json",
            "source_diff.json",
            "schema.json",
            "run_registry.json",
        ]
        files = [root / n for n in static]
        files += list((root / "analysis").glob("*")) + list((root / "timing").glob("*.json"))
        for name, commit in commits.items():
            directory = root / "runs" / name
            # Export only authoritative committed packets and matching derived metadata.
            assert read_json(directory / "packet_index.json") == commit["packet_index"], (
                "uncommitted packets: resume first"
            )
            files += [
                directory / n
                for n in ("meta.json", "commit.json", "packet_index.json", "reset_ledger.json")
            ]
            files += [directory / n for n in commit["packet_index"]]
        for path in files:
            target = stage / path.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        for name in (
            "bs5_bundle.py",
            "analyze_bs5_curriculum_bundle.py",
            "analyze_bs5_direct_replay_reset_bundle.py",
        ):
            shutil.copyfile(ROOT / "scripts/analysis" / name, stage / name)
        manifest = {
            p.relative_to(stage).as_posix(): file_hash(p) for p in stage.rglob("*") if p.is_file()
        }
        atomic_json(stage / "SHA256SUMS.json", manifest)
        prefix = f"snapshot_{snapshot[:12]}"
        generation = 1
        while any(exports.glob(prefix + "_*.zip")):
            prefix = f"snapshot_{snapshot[:12]}_g{generation}"
            generation += 1
        audit = exports / f"{prefix}_audit_light.zip"
        zip_snapshot(audit, [p for p in stage.rglob("*") if p.is_file()], stage)
        shards = seed_shards(audit, exports, prefix)
        # Repro references the audit by exact hash rather than duplicating raw evidence.
        repro = stage / "repro"
        repro.mkdir()
        shutil.copyfile(root / "source_snapshot.zip", repro / "source_snapshot.zip")
        atomic_json(
            repro / "companion.json",
            dict(
                audit_light=audit.name,
                sha256=file_hash(audit),
                snapshot_sha256=snapshot,
                seed_shards=shards,
                commands="python -B -m scripts.training.run_bs5_direct_replay_reset --run --resume --outdir <restored-root>",
            ),
        )
        for name, commit in commits.items():
            directory = root / "runs" / name
            target = repro / "runs" / name
            target.mkdir(parents=True)
            shutil.copyfile(directory / "resume.pt", target / "resume.pt")
            for path in sorted((directory / "checkpoints").glob("*.pt")):
                if int(path.stem) <= commit["steps"]:
                    dest = target / "checkpoints" / path.name
                    dest.parent.mkdir(exist_ok=True)
                    shutil.copyfile(path, dest)
        atomic_json(
            repro / "SHA256SUMS.json",
            {
                p.relative_to(repro).as_posix(): file_hash(p)
                for p in repro.rglob("*")
                if p.is_file()
            },
        )
        full = exports / f"{prefix}_repro_full.zip"
        zip_snapshot(full, [p for p in repro.rglob("*") if p.is_file()], repro)
        for key, value in (
            ("export_input_bytes", sum(f.stat().st_size for f in files)),
            ("export_member_hash_count", len(manifest)),
            (
                "export_output_bytes",
                audit.stat().st_size + full.stat().st_size + sum(s["bytes"] for s in shards),
            ),
        ):
            timers.data["io"][key] = timers.data["io"].get(key, 0) + value
        timers.data["io"]["source_members"] = len(read_json(root / "source_identity.json")["files"])
    timers.flush(status="large_exports_complete")
    # Closing receipt lives in small review; large archives explicitly end at their snapshot cutoff.
    receipt = dict(
        snapshot_sha256=snapshot,
        ended_utc=utc_now(),
        large_export_elapsed_s=time.monotonic() - began,
        timer_scope="audit/repro contain pre-compression timing cutoff; this receipt includes their completed compression/hash/CRC; review/receipt final writes excluded from this cutoff",
        seed_shards=shards,
        bundles={
            audit.name: file_hash(audit),
            full.name: file_hash(full),
            **{s["name"]: s["sha256"] for s in shards},
        },
    )
    with timers.phase("export"), tempfile.TemporaryDirectory(prefix="bs5-review-") as temporary:
        review = Path(temporary)
        for folder in ("analysis", "timing"):
            shutil.copytree(root / folder, review / folder)
        for name in static:
            shutil.copyfile(root / name, review / name)
        for name in commits:
            target = review / "runs" / name
            target.mkdir(parents=True)
            for filename in ("reset_ledger.json", "meta.json"):
                shutil.copyfile(root / "runs" / name / filename, target / filename)
        atomic_json(review / "export_receipt.json", receipt)
        atomic_json(
            review / "SHA256SUMS.json",
            {
                p.relative_to(review).as_posix(): file_hash(p)
                for p in review.rglob("*")
                if p.is_file()
            },
        )
        small = exports / f"{prefix}_review.zip"
        zip_snapshot(small, [p for p in review.rglob("*") if p.is_file()], review)
        receipt["bundles"][small.name] = file_hash(small)
    timers.flush(status="export_complete_including_review")
    receipt["completed_timing"] = deepcopy(timers.data)
    atomic_json(exports / "latest.json", receipt)
    print(json.dumps(dict(event="EXPORTED", snapshot=snapshot[:12], review=small.name)), flush=True)
    return receipt


def run_matrix(root, *, resume=False, export_policy=None, start_at=None):
    timers = Timers(root, "run_resume" if resume else "run", start_at=start_at)
    with timers.phase("setup"):
        p = read_json(root / "protocol.json")
        timers.device = p["device"]
        timers.data["cuda_boundary_sync"] = p["device"].startswith("cuda")
        assert source_identity()["sha256"] == p["source_sha256"], (
            "source changed: use frozen source snapshot"
        )
        torch.set_num_threads(p["torch_threads"])
        policy = export_policy or p["export_policy"]
        assert policy in {"per-run", "final", "manual"}
    registry = read_json(root / "run_registry.json")
    try:
        for seed in p["training_seeds"]:
            was_complete = registry["runs"][f"{ARM}_seed{seed}"]["status"] == "complete"
            run_one(root, p, registry, seed, timers, resume=resume)
            if policy == "per-run" and not was_complete:
                export_bundles(root, timers)
        if policy == "final":
            export_bundles(root, timers)
    finally:
        timers.flush(status=read_json(root / "run_registry.json")["status"])


def reconcile_only(root):
    """Recover durable marker from resume without collecting a single transition."""
    p = read_json(root / "protocol.json")
    assert source_identity()["sha256"] == p["source_sha256"], "use frozen source"
    registry = read_json(root / "run_registry.json")
    for name in registry["runs"]:
        directory = root / "runs" / name
        if not (directory / "resume.pt").exists():
            if directory.exists():
                reconcile(directory, dict(steps=0, episodes=0, packet_index={}, resets=[]))
            registry["runs"][name] = dict(status="not_started", steps=0)
            continue
        state = torch.load(directory / "resume.pt", map_location="cpu", weights_only=False)
        commit = state["commit"]
        reconcile(directory, commit)
        registry["runs"][name] = dict(
            steps=commit["steps"],
            status="complete" if commit["steps"] == p["steps_per_run"] else "incomplete",
        )
    registry["status"] = (
        "complete"
        if all(e["status"] == "complete" for e in registry["runs"].values())
        else "incomplete"
    )
    atomic_json(root / "run_registry.json", registry)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    for name in ("prepare", "run", "export", "reconcile"):
        group.add_argument("--" + name, action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--smoke", action="store_true", help="fresh debug seeds; 384 steps/run, never main"
    )
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--export-policy", choices=["per-run", "final", "manual"])
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/bs5_direct_replay_reset_v1.json"
    )
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    root = args.outdir.resolve()
    if args.prepare:
        prepare(root, args.config, smoke=args.smoke, device=args.device, start_at=PROCESS_ENTRY)
    else:
        with experiment_lock(root):
            if args.run:
                run_matrix(
                    root,
                    resume=args.resume,
                    export_policy=args.export_policy,
                    start_at=PROCESS_ENTRY,
                )
            elif args.reconcile:
                timers = Timers(root, "reconcile", start_at=PROCESS_ENTRY)
                with timers.phase("persistence"):
                    reconcile_only(root)
                timers.flush(status="reconciled_no_training")
            else:
                export_bundles(root, start_at=PROCESS_ENTRY)


if __name__ == "__main__":
    main()
