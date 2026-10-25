"""Fresh C-N matched-recipe arm; existing C-R is frozen. Separate runner: historical science/runner defaults unchanged.

python -B scripts/training/run_bs5_cn_h1200_return.py prepare --output OUT --device cpu
python -B scripts/training/run_bs5_cn_h1200_return.py run --output OUT [--resume]
python -B scripts/training/run_bs5_cn_h1200_return.py export --output OUT
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, fields
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from scripts.analysis.bs5_bundle import atomic_json, canonical_hash, file_hash, read_json
from scripts.training.run_bs5_curriculum_pilot import act, protocol_new
from scripts.training.run_bs5_launch_ab import (
    atomic_torch, capture_decision, experiment_lock, freeze_source, serial,
    source_identity, step_record, weights_hash, write_trace,
)
from src.envs.bs5_curriculum_env import condition_for, task_signature
from src.envs.bs5_collision_safe_env import COLLISION_VERSION, COLLISION_CONTRACT
from src.envs.bs5_energy_return_env import BS5EnergyReturnEnv, RETURN_VERSION, RETURN_CONTRACT
from scripts.experiments.bs5_crcn_common import TraceReplayReturnEnv, verify_assets
TRACE_HOME = None
from src.envs.bs5_launch_ab_env import Geometry
from src.rl.bs5_curriculum_qmix import BS5CurriculumQMIX
from src.rl.bs5_curriculum_replay import CurriculumReplay, CurriculumTransition

SPEC = ROOT / "configs/bs5_cn_h1200_60k_return_v3.json"


def runtime(device):
    return dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
                device=device, threads=torch.get_num_threads(),
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                cuda=torch.version.cuda if device == "cuda" else None,
                hardware=torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
                platform=sys.platform)


def setup_runtime(device, threads):
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; choose CPU before prepare")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def learning_terminal(stage, terminated, truncated):
    # Final next state is captured BEFORE any reset. Only S4 uses continuing semantics.
    return bool(terminated or (truncated and stage != "S4"))


def build_protocol(identity, device="cpu", *, diagnostic=False):
    spec = read_json(SPEC)
    assert spec["collision_version"] == COLLISION_VERSION
    assert spec["return_version"] == RETURN_VERSION
    base = read_json(ROOT / "configs/bs5_curriculum_fixed_v1.json")
    # Reuse the historical 24k prefix schedule, including its RNG consumption,
    # so changing the number of S4 games does not alter S1-S3 fixtures.
    base.update(requested_steps_per_run=24000, stage_ends=[3000,6000,12000,24000],
                device=device, training_seeds=[551])
    if diagnostic:
        base.update(training_seeds=[955551], horizon=6, stage_ends=[24, 48, 96, 192],
                    requested_steps_per_run=192, warmup=8, batch_size=8,
                    target_every=24, epsilon_decay_steps=144, replay_capacity=256)
    p = protocol_new(base, identity)
    p.update(version=spec["version"], arms=["C-N"], arm="C-N",
             collision_version=COLLISION_VERSION, collision_contract=COLLISION_CONTRACT,
             return_version=RETURN_VERSION, return_contract=RETURN_CONTRACT,
             lineage_policy="fresh C-N Return-v3 only; never initialize from C-R weights/replay/optimizer",
             training_seed=base["training_seeds"][0], total_steps=192 if diagnostic else spec["total_steps"],
             steps_per_run=192 if diagnostic else spec["total_steps"],
             stage_ends=base["stage_ends"] if diagnostic else spec["stage_ends"],
             early_horizon=6 if diagnostic else 150,
             full_horizon=24 if diagnostic else 1200,
             evaluation_steps=[0, 96, 192] if diagnostic else spec["evaluation_steps"],
             validation_pool=[180001] if diagnostic else spec["validation_pool"],
             diagnostic=diagnostic, chunk_slots=6 if diagnostic else spec["chunk_slots"],
             budget_reservation_slots=10, scope=spec["scope"],
             best_selection="max mean full-game validation utility; exact ties earliest",
             time_limit="S1-S3 done=terminated|truncated; S4 done=terminated; final next state before reset",
             replay_lineage="physical-task-change clears, plus explicit clear at S4 entry",
             automatic_extension=False, slot_s=5.0)
    old = p.pop("scenario_schedule")[str(p["training_seed"])]
    schedule, start, eid = [], 0, 0
    s4_rng = np.random.default_rng(np.random.SeedSequence([p["training_seed"], 0x31323030]))
    order = list(map(int, s4_rng.permutation(p["train_pool"])))
    while start < p["total_steps"]:
        c = condition_for(start, "C-N", p)
        h = p["full_horizon"] if c["stage"] == "S4" else p["early_horizon"]
        assert start + h <= next(e for e in p["stage_ends"] if start < e)
        item = deepcopy(old[eid])
        if c["stage"] == "S4":
            k = (start - p["stage_ends"][2]) // h
            item["scenario_seed"] = order[k % len(order)]
        item.update(episode=eid, start=start, horizon=h, condition=c)
        schedule.append(item)
        eid += 1
        start += h
    p["game_schedule"] = schedule
    p["allow_relay"] = False
    p["assets_sha256"] = verify_assets()["manifest_sha256"]
    p["trace_contract"] = "historical C-R training/validation UE traces replayed exactly; no future trace in observations"
    p["planned_training_steps"] = p["total_steps"]
    p["planned_validation_steps"] = len(p["evaluation_steps"]) * len(p["validation_pool"]) * p["full_horizon"]
    p.pop("training_conditions", None)
    p.pop("common_evaluation_steps", None)
    p["validation_pool_usage"] = "previously inspected validation, not unseen final test"
    return p


def prepare(output, device="cpu", *, diagnostic=False):
    output = Path(output).resolve()
    assert not output.exists(), "Use a new output, or run --resume on existing output"
    spec = read_json(SPEC)
    setup_runtime(device, spec["torch_threads"])
    identity = source_identity()
    p = build_protocol(identity, device, diagnostic=diagnostic)
    p["runtime"] = runtime(device)
    output.mkdir(parents=True)
    freeze_source(output, identity)
    atomic_json(output / "protocol.json", p)
    atomic_json(output / "budget_ledger.json", dict(
        protocol_sha256=canonical_hash(p),
        phases={k: dict(cap=cap, charged=0, observed=0) for k, cap in
                (("train", p["planned_training_steps"]), ("eval", p["planned_validation_steps"]))},
        semantics="reserve before physical steps; charged includes uncertain reservations; never erased on resume"))
    atomic_json(output / "status.json", dict(status="prepared", steps=0))
    return p


class Budget:
    def __init__(self, root):
        self.path = Path(root) / "budget_ledger.json"
        self.data = read_json(self.path)

    def reserve(self, phase, count):
        row = self.data["phases"][phase]
        if row["charged"] + count > row["cap"]:
            raise RuntimeError(f"{phase} compute cap reached; no automatic rerun/extension")
        row["charged"] += count
        self.flush()

    def observed(self, phase):
        self.data["phases"][phase]["observed"] += 1

    def flush(self):
        atomic_json(self.path, self.data)


def pack_replay(replay):
    names = [f.name for f in fields(CurriculumTransition)]
    return dict(capacity=replay.capacity, pos=replay.pos, rng=replay.rng.getstate(),
                stage_counts=replay.stage_counts,
                arrays={key: np.stack([getattr(t, key) for t in replay.data])
                        for key in names} if replay.data else {}, count=len(replay))


def unpack_replay(data):
    r = CurriculumReplay(data["capacity"])
    r.pos, r.stage_counts = data["pos"], data["stage_counts"]
    r.rng.setstate(data["rng"])
    r.data = [CurriculumTransition(**{k: values[i] for k, values in data["arrays"].items()})
              for i in range(data["count"])]
    return r


def sample_audit(replay, size):
    indices = replay.rng.sample(range(len(replay.data)), size)
    items = [replay.data[i] for i in indices]
    batch = {k: np.stack([getattr(t, k) for t in items]).astype(dtype) for k, dtype in (
        ("obs", np.float32), ("state", np.float32), ("actions", np.int64),
        ("reward", np.float32), ("next_obs", np.float32), ("next_state", np.float32),
        ("next_action_mask", np.bool_), ("done", np.float32))}
    digest = hashlib.sha256()
    for key in sorted(batch):
        digest.update(key.encode())
        digest.update(batch[key].tobytes())
    return batch, dict(indices=indices, collection_steps=[int(t.collection_step) for t in items],
                       sha256=digest.hexdigest())


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(s):
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"].cpu())
    if s["cuda"]:
        torch.cuda.set_rng_state_all([x.cpu() for x in s["cuda"]])


def save_resume(root, learner, replay, exploration, commit, p):
    started = time.monotonic()
    state = dict(commit=deepcopy(commit), protocol_sha256=canonical_hash(p),
                 models={k: getattr(learner, k).state_dict() for k in
                         ("agent", "mixer", "target_agent", "target_mixer")},
                 optimizer=learner.optimizer.state_dict(), replay=pack_replay(replay),
                 exploration=exploration.bit_generator.state, rng=rng_state(),
                 environment="committed game boundary; no in-flight env; next reset in frozen game_schedule")
    atomic_torch(root / "resume.pt", state)
    commit["timers"]["checkpoint_s"] += time.monotonic() - started
    # resume.pt is the authoritative commit. JSON may be reconstructed after a crash.
    atomic_json(root / "commit.json", commit)


def new_env(p, c, horizon, residual):
    return TraceReplayReturnEnv(c, allow_relay=p.get("allow_relay", False), horizon=horizon,
        residual_battery=residual, fixture=p["fixture"], trace_home=TRACE_HOME)


def append_chunk(root, directory, number, rows, updates):
    path = root / directory / f"{number:04d}.json.gz"
    write_trace(path, dict(rows=rows, updates=updates))
    return dict(path=path.relative_to(root).as_posix(), sha256=file_hash(path),
                first=rows[0]["slot"], last=rows[-1]["slot"], slots=len(rows))


def rollout(root, learner, replay, exploration, p, budget, commit, *, spec, evaluation=False, prefix_validator=None):
    """Exactly one env reset per game. Chunk writes never reset or end transitions."""
    from scripts.analysis.analyze_bs5_cn_h1200_return import metrics
    c, h, scenario = spec["condition"], spec["horizon"], spec["scenario_seed"]
    env = new_env(p, c, h, spec["residual_battery"])
    t0 = time.monotonic()
    out = env.reset(seed=scenario, assignment_seed=spec["assignment_seed"], diagnostic=evaluation)
    commit["timers"]["env_s"] += time.monotonic() - t0
    initial = dict(battery=env.battery.tolist(), residual_uav=int(env.residual_uav_idx),
                   charging_station_cells=env.charging_stations.tolist(),
                   positions=env.uav_positions.tolist(), ue_positions=env.ue_positions.tolist())
    geometry = Geometry(env)
    directory = (f"evaluations/{commit['steps']:06d}/{scenario}" if evaluation
                 else f"training/{spec['episode']:04d}")
    rows, chunk, updates, chunks = [], [], [], []
    phase = "eval" if evaluation else "train"
    env._audit_phase = phase
    reservation_end = 0
    replay_samples_after_rejoin = 0
    rejoined = False
    for t in range(h):
        if t == reservation_end:
            n = min(p["budget_reservation_slots"], h - t)
            budget.reserve(phase, n)
            reservation_end += n
        step = commit["steps"]
        epsilon = 0.0 if evaluation else 1 + (p["epsilon_end"] - 1) * min(step / p["epsilon_decay_steps"], 1)
        decision = capture_decision(env)
        decision["ue_positions"] = env.ue_positions.copy()
        before_rng = canonical_hash(serial(env.rng.bit_generator.state))
        obs, state, mask = out["obs"].copy(), out["state"].copy(), out["action_mask"].copy()
        t0 = time.monotonic()
        actions, q = act(learner, out, epsilon, exploration)
        commit["timers"]["actor_s"] += time.monotonic() - t0
        t0 = time.monotonic()
        nxt = env.step(actions)
        commit["timers"]["env_s"] += time.monotonic() - t0
        budget.observed(phase)
        assert not nxt["terminated"], "Unexpected true terminal; keep partial evidence and audit"
        assert bool(nxt["truncated"]) == (t + 1 == h), "unexpected env reset/time limit"
        terminal = learning_terminal(c["stage"], nxt["terminated"], nxt["truncated"])
        if not evaluation:
            replay.add(CurriculumTransition(obs, state, np.asarray(actions), nxt["reward"],
                nxt["obs"].copy(), nxt["state"].copy(), nxt["action_mask"].copy(), float(terminal),
                mask, c["stage"], c["substage"], step, commit["lineage"]))
            commit["steps"] += 1
            if len(replay) >= max(p["warmup"], p["batch_size"]) and commit["steps"] % p["update_every"] == 0:
                t0 = time.monotonic()
                batch, batch_info = sample_audit(replay, p["batch_size"])
                update = learner.train_batch(batch)
                commit["timers"]["learner_s"] += time.monotonic() - t0
                commit["updates"] += 1
                updates.append(dict(step=commit["steps"], update=commit["updates"], epsilon=epsilon,
                    batch=batch_info, replay_size=len(replay), replay_composition=replay.composition(),
                    lineage=commit["lineage"], **update))
            if commit["steps"] % p["target_every"] == 0:
                learner.update_targets()
        t0 = time.monotonic()
        # No shadow enumeration or policy re-evaluation: delivered network is already in snapshot.
        row = step_record(env, decision, mask, actions, nxt, geometry, arm=p["arm"],
                          seed=p["training_seed"], scenario=scenario, slot=t + 1)
        for agent in env.last_return_control["forced_return_ids"]:
            row["applied_actions"][agent]["autonomous"] = True
        row.update(global_step=step if evaluation else step + 1, checkpoint_step=step if evaluation else None,
            collision=deepcopy(env.last_collision_resolution),
            return_control=deepcopy(env.last_return_control),
            charging_station_cells=env.charging_stations.tolist(),
            stage_id=c["stage"], substage_id=c["substage"], task_condition=c,
            episode=spec.get("episode"), epsilon=epsilon, obs=obs.tolist(), state=state.tolist(),
            next_obs=nxt["obs"].tolist(), next_state=nxt["state"].tolist(),
            next_action_mask=nxt["action_mask"].tolist(), terminated=bool(nxt["terminated"]),
            truncated=bool(nxt["truncated"]), learning_terminal=terminal, bootstrap_mask=int(not terminal),
            env_t_before=t, env_t_after=int(env.t), q_summary=q,
            env_rng_before=before_rng, env_rng_after=canonical_hash(serial(env.rng.bit_generator.state)),
            failed_bs_gateway_excluded=True, surviving_bs_ids=[0, 1, 2, 3, 4],
            lifecycle_transitions=[dict(uav=a, before=int(b), after=int(s), slot=t+1)
                for a, (b,s) in enumerate(zip(decision["status"], env.status)) if b != s])
        if any(b == 2 and a == 0 for b, a in zip(decision["status"], env.status)):
            rejoined = True
        if rejoined:
            replay_samples_after_rejoin += int(not evaluation)
        commit["timers"]["record_s"] += time.monotonic() - t0
        assert row["env_rng_after"] == canonical_hash(serial(env.rng.bit_generator.state))
        rows.append(row)
        chunk.append(row)
        out = nxt
        if len(chunk) == p["chunk_slots"] or t + 1 == h:
            t0 = time.monotonic()
            chunks.append(append_chunk(root, directory, len(chunks), chunk, updates))
            chunk, updates = [], []
            budget.flush()
            commit["timers"]["raw_io_s"] += time.monotonic() - t0
        if prefix_validator is not None and t + 1 == 1200:
            prefix_validator(rows)
        if evaluation and (t+1)%1000==0:
            print(json.dumps(dict(phase="eval_progress",arm=p["arm"],scenario=scenario,slot=t+1,horizon=h)),flush=True)
    result = dict(spec=spec, initial=initial, resolved_config=serial(asdict(env.cfg)), chunks=chunks,
                  metrics=metrics(rows), after_first_rejoin_transitions=replay_samples_after_rejoin, trace=env.trace_receipt)
    atomic_json(root / directory / "index.json", result)
    return result, directory + "/index.json"


def evaluate(root, learner, p, budget, commit):
    # Restore ALL training RNGs, including the exploration generator, even on failure.
    saved = rng_state()
    mode, before = learner.agent.training, weights_hash(learner)
    learner.agent.eval()
    dummy_rng = np.random.default_rng(0)
    paths, utilities = [], []
    c = condition_for(p["stage_ends"][2], "C-N", p)
    start = time.monotonic()
    try:
        for scenario in p["validation_pool"]:
            spec = dict(condition=c, horizon=p["full_horizon"], scenario_seed=scenario,
                        assignment_seed=scenario, residual_battery=p["validation_residual_battery"])
            result, path = rollout(root, learner, None, dummy_rng, p, budget, commit,
                                   spec=spec, evaluation=True)
            paths.append(path)
            utilities.append(result["metrics"]["utility"])
            print(json.dumps(dict(phase="cn_validation",checkpoint=commit["steps"],scenario=scenario,utility=result["metrics"]["utility"])),flush=True)
    finally:
        restore_rng(saved)
        learner.agent.train(mode)
    assert weights_hash(learner) == before
    ev = dict(step=commit["steps"], utility=float(np.mean(utilities)), games=paths,
              weights_sha256=before, runtime_s=time.monotonic()-start, epsilon=0.0)
    commit["timers"]["eval_inclusive_s"] += ev["runtime_s"]
    commit["evaluations"].append(ev)
    cp = root / "checkpoints" / f"{commit['steps']:06d}.pt"
    atomic_torch(cp, dict(agent=learner.agent.state_dict(), mixer=learner.mixer.state_dict(), step=commit["steps"]))
    ev["checkpoint_sha256"] = file_hash(cp)
    if ev["utility"] > commit["best_utility"]:
        commit["best_utility"], commit["best_step"] = ev["utility"], ev["step"]
    atomic_json(root / "checkpoint_selection.json", dict(rule=p["best_selection"],
        best_step=commit["best_step"], best_utility=commit["best_utility"], last_step=ev["step"],
        curve=commit["evaluations"], protocol_sha256=canonical_hash(p), source_sha256=p["source_sha256"]))


def run(output, *, resume=False, stop_after=None, max_seconds=None):
    global TRACE_HOME
    root = Path(output).resolve()
    TRACE_HOME = root.parent / "traces"
    verify_assets()
    p = read_json(root / "protocol.json")
    assert p.get("return_version") == RETURN_VERSION, "Legacy env output cannot resume in Return-v3"
    assert p.get("collision_version") == COLLISION_VERSION
    assert p["version"] == read_json(SPEC)["version"], "Wrong experiment version"
    setup_runtime(p["device"], p["torch_threads"])
    assert runtime(p["device"]) == p["runtime"], "Runtime changed; do not call this exact resume"
    assert source_identity()["sha256"] == p["source_sha256"], "Frozen source changed"
    with experiment_lock(root):
        # A previous PARTIAL export is immutable in its own directory. The live
        # directory becomes mutable again; validate its old manifest before removing it.
        manifest=root/"SHA256SUMS.json"
        if manifest.exists():
            for rel,digest in read_json(manifest).items():
                assert file_hash(root/rel)==digest, f"changed exported evidence: {rel}"
            repro=root/"REPRO_SHA256SUMS.json"
            if repro.exists():
                for rel,digest in read_json(repro).items():
                    assert file_hash(root/rel)==digest, f"changed exported resume/checkpoint: {rel}"
                # The immutable export retains its hashes. The live root will
                # write new resume weights, so do not leave stale repro hashes
                # for the next export's initial offline analysis.
                repro.unlink()
            manifest.unlink()
        begin=time.monotonic()
        try:
            _run(root, p, resume=resume, stop_after=stop_after, max_seconds=max_seconds)
        finally:
            path=root/"runtime_sessions.json"
            sessions=read_json(path) if path.exists() else []
            sessions.append(dict(wall_s=time.monotonic()-begin,resume=resume,
                status=read_json(root/"status.json"),
                note="includes initialization/load, train, eval, logging and checkpoints; eval timer is nested"))
            atomic_json(path,sessions)


def _run(root, p, *, resume, stop_after, max_seconds):
    torch.manual_seed(p["training_seed"])
    random.seed(p["training_seed"])
    np.random.seed(p["training_seed"])
    learner = BS5CurriculumQMIX(new_env(p, condition_for(0, "C-N", p), p["early_horizon"], 30),
                               lr=p["lr"], gamma=p["gamma"], device=p["device"])
    if not p["diagnostic"]:
        assert weights_hash(learner) == verify_assets()["initial_weights_sha256"], "Fresh C-N must start from matching seed551 initialization"
    exploration = np.random.default_rng(np.random.SeedSequence([p["training_seed"], 0x504F4C49]))
    replay = CurriculumReplay(p["replay_capacity"], p["training_seed"] + 20000)
    commit = dict(steps=0, episodes=0, updates=0, lineage=0, signature=None, games=[], events=[],
        evaluations=[], best_step=0, best_utility=-1., timers={k:0. for k in
            ("env_s", "actor_s", "learner_s", "record_s", "raw_io_s", "checkpoint_s", "eval_inclusive_s", "session_wall_s")})
    budget = Budget(root)
    if (root / "resume.pt").exists():
        assert resume, "Existing resume.pt; use --resume"
        data = torch.load(root / "resume.pt", map_location=learner.device, weights_only=False)
        assert data["protocol_sha256"] == canonical_hash(p)
        for key, state in data["models"].items():
            getattr(learner, key).load_state_dict(state)
        learner.optimizer.load_state_dict(data["optimizer"])
        replay = unpack_replay(data["replay"])
        exploration.bit_generator.state = data["exploration"]
        restore_rng(data["rng"])
        commit = data["commit"]
        atomic_json(root / "commit.json", commit)
    elif resume:
        raise RuntimeError("No durable resume.pt; cannot resume an uncommitted run")
    charged = budget.data["phases"]["train"]["charged"]
    eval_used = sum(len(e["games"])*p["full_horizon"] for e in commit["evaluations"])
    assert charged == commit["steps"], "Uncommitted training compute; keep partial/export; no hidden rerun"
    assert budget.data["phases"]["eval"]["charged"] == eval_used, "Uncommitted eval compute; no hidden rerun"
    started = time.monotonic()
    session_started = started
    try:
        if not commit["evaluations"]:
            evaluate(root, learner, p, budget, commit)
            save_resume(root, learner, replay, exploration, commit, p)
        while commit["steps"] < p["total_steps"]:
            if (stop_after is not None and commit["steps"] >= stop_after) or (
                    max_seconds is not None and time.monotonic()-session_started >= max_seconds):
                break
            spec = p["game_schedule"][commit["episodes"]]
            assert spec["start"] == commit["steps"]
            c = spec["condition"]
            signature = serial(task_signature(c)) + [["learning_regime", "continuing1200" if c["stage"] == "S4" else "finite150"]]
            if signature != commit["signature"]:
                before = len(replay)
                if commit["signature"] is not None:
                    commit["lineage"] += 1
                    replay = CurriculumReplay(p["replay_capacity"], p["training_seed"]+20000+commit["lineage"])
                commit["events"].append(dict(step=commit["steps"], kind="task_change_or_s4_entry",
                    discarded=before, signature=signature, lineage=commit["lineage"], epsilon_reset=False))
                commit["signature"] = signature
            result, path = rollout(root, learner, replay, exploration, p, budget, commit, spec=spec)
            commit["episodes"] += 1
            commit["games"].append(path)
            if commit["steps"] in p["evaluation_steps"]:
                evaluate(root, learner, p, budget, commit)
            commit["timers"]["session_wall_s"] += time.monotonic()-started
            save_resume(root, learner, replay, exploration, commit, p)
            started = time.monotonic()
            atomic_json(root / "status.json", dict(status="running", steps=commit["steps"],
                        episodes=commit["episodes"], updates=commit["updates"]))
            if c["stage"] == "S4" or commit["episodes"] % 10 == 0:
                print(json.dumps(dict(step=commit["steps"], stage=c["stage"], horizon=spec["horizon"],
                    utility=result["metrics"]["utility"], updates=commit["updates"],
                    best_step=commit["best_step"])), flush=True)
        status = "complete" if commit["steps"] == p["total_steps"] else "paused_at_game_boundary"
        atomic_json(root / "status.json", dict(status=status, steps=commit["steps"], episodes=commit["episodes"]))
    except BaseException as error:
        budget.flush()
        atomic_json(root / "status.json", dict(status="partial_error", error=repr(error),
            in_memory_steps=commit["steps"], durable_steps=read_json(root/"commit.json")["steps"] if (root/"commit.json").exists() else 0,
            instruction="Preserve this output; no automatic rollback/rerun beyond charged compute"))
        raise


def export(root):
    """Single end-of-run export. Independent ZIP parts, each with ordinary file entries."""
    from scripts.analysis.analyze_bs5_cn_h1200_return import analyze
    root = Path(root).resolve()
    out = root.parent / (root.name + "_export")
    suffix=1
    while out.exists():
        suffix+=1
        out=root.parent/(root.name+f"_export_{suffix}")
    out.mkdir()
    complete = read_json(root / "status.json")["status"] == "complete"
    start = time.monotonic()
    analyze(root, out / "analysis", require_complete=complete, manifest_required=False)
    helpers = ["analyze_bs5_cn_h1200_return.py", "bs5_bundle.py", "bs5_continuous_common.py",
               "analyze_bs5_curriculum_bundle.py", "analyze_bs5_curriculum_extend.py", "bs5_curve_plot.py",
               "bs5_return_audit.py"]
    for name in helpers:
        shutil.copy2(ROOT / "scripts/analysis" / name, root / name)
    # Hash all committed evidence, including resume/checkpoints, but not lock/temp/manifest.
    files = sorted(f for f in root.rglob("*") if f.is_file() and f.name not in
                   {"SHA256SUMS.json", ".experiment.lock", "run.lock"} and not f.name.endswith(".tmp"))
    repro = {f.relative_to(root).as_posix():file_hash(f) for f in files
             if f == root/"resume.pt" or f.parent == root/"checkpoints"}
    atomic_json(root/"REPRO_SHA256SUMS.json",repro)
    files=[f for f in files if f.name != "REPRO_SHA256SUMS.json"]+[root/"REPRO_SHA256SUMS.json"]
    manifest = {f.relative_to(root).as_posix():file_hash(f) for f in files
                if f.relative_to(root).as_posix() not in repro}
    atomic_json(root / "SHA256SUMS.json", manifest)
    files.append(root / "SHA256SUMS.json")
    groups = {"audit_light":[], "repro_full":[]}
    for f in files:
        rel = f.relative_to(root).as_posix()
        groups["repro_full" if rel == "resume.pt" or rel.startswith("checkpoints/") else "audit_light"].append(f)
    receipts = []
    for group, members in groups.items():
        number, archive, size = 0, None, 0
        for f in members:
            # Use uncompressed size for a conservative bound; single large resume stays standalone.
            if f.stat().st_size > 80*1024**2:
                if archive:
                    archive.close(); archive=None
                # Packed replay arrays compress well; try ONE ordinary ZIP first.
                largezip=out/(group+"_"+f.name+".zip")
                with zipfile.ZipFile(largezip,"x",zipfile.ZIP_DEFLATED,compresslevel=3) as z:
                    z.write(f,f.relative_to(root).as_posix())
                if largezip.stat().st_size>80*1024**2:
                    largezip.unlink()
                    target=out/(group+"_"+f.name)
                    shutil.copy2(f,target)
                    receipts.append(dict(path=target.name,sha256=file_hash(target),bytes=target.stat().st_size,
                        restore_to=f.relative_to(root).as_posix(),kind="standalone_large_file"))
                continue
            if archive is None or size + f.stat().st_size > 75*1024**2:
                if archive: archive.close()
                number += 1; size=0
                archive=zipfile.ZipFile(out/f"{group}_part{number:03d}.zip","x",zipfile.ZIP_DEFLATED,compresslevel=3)
            archive.write(f,f.relative_to(root).as_posix()); size+=f.stat().st_size
        if archive: archive.close()
    for f in sorted(out.glob("*.zip")):
        with zipfile.ZipFile(f) as z:
            assert z.testzip() is None
        receipts.append(dict(path=f.name,sha256=file_hash(f),bytes=f.stat().st_size,kind="independent_zip"))
    atomic_json(out/"EXPORT_INDEX.json",dict(complete=complete,files=receipts,
        instruction="Extract ALL audit_light/repro_full ZIPs into ONE empty folder; copy standalone file to restore_to. Run analyzer there.",
        analyzer="python -B analyze_bs5_cn_h1200_return.py --root MERGED --output ANALYSIS --require-complete",
        export_wall_s=time.monotonic()-start))
    # Verify merged transport without loading Torch states or stepping an env.
    merged=out/"_verify_merged"
    merged.mkdir()
    for item in receipts:
        f=out/item["path"]
        if item["kind"] == "independent_zip":
            with zipfile.ZipFile(f) as z: z.extractall(merged)
        else:
            dest=merged/item["restore_to"]; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(f,dest)
    analyze(merged,out/"verified_analysis",require_complete=complete,manifest_required=True)
    shutil.rmtree(merged)
    review=out/"review_small.zip"
    with zipfile.ZipFile(review,"x",zipfile.ZIP_DEFLATED) as z:
        for f in sorted((out/"verified_analysis").rglob("*")):
            if f.is_file(): z.write(f,"analysis/"+f.relative_to(out/"verified_analysis").as_posix())
        for name in ("protocol.json","budget_ledger.json","status.json","checkpoint_selection.json"):
            if (root/name).exists(): z.write(root/name,name)
        z.write(out/"EXPORT_INDEX.json","EXPORT_INDEX.json")
    print(f"Export verified: {out}",flush=True)
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["prepare","run","export"])
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    parser.add_argument("--diagnostic",action="store_true")
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--stop-after",type=int)
    parser.add_argument("--max-seconds",type=float)
    args=parser.parse_args()
    if args.command=="prepare": prepare(args.output,args.device,diagnostic=args.diagnostic)
    elif args.command=="run": run(args.output,resume=args.resume,stop_after=args.stop_after,max_seconds=args.max_seconds)
    else: export(args.output)


if __name__=="__main__": main()
