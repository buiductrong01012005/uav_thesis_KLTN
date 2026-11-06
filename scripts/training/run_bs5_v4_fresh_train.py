"""Fresh C-R/C-N seed551/552 training only. No final benchmark in this runner."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from scripts.experiments.bs5_v4_fresh_common import (
    Budget, BS5CurriculumQMIX, CurriculumReplay, CurriculumTransition, RETURN_CONTRACT,
    RETURN_VERSION, act, atomic_json, atomic_torch, canonical_hash, condition_for, env_for,
    experiment_lock, file_hash, full_condition, pack_replay, read_json, restore_rng,
    rng_state, rollout, sample_audit, save_gz, seed_all, setup, source_identity,
    unpack_replay, weights_hash,
)
from scripts.training.run_bs5_launch_ab import freeze_source
from src.envs.bs5_curriculum_env import task_signature

CONFIG = ROOT / "configs/bs5_return_v4_fresh_train.json"
STOP_REQUESTED = False


def build_protocol(device="auto", diagnostic=False, arms=None, seeds=None):
    p = read_json(CONFIG)
    assert p["return_version"] == RETURN_VERSION
    if diagnostic:
        p.update(training_seeds=[955551, 955552], total_steps=96, stage_ends=[12, 24, 48, 96],
                 early_horizon=6, full_horizon=24, evaluation_steps=[0, 48, 96],
                 validation_pool=[280001, 280002], train_pool=[282001, 282002, 282003, 282004],
                 replay_capacity=128, batch_size=8, warmup=8, target_every=24,
                 epsilon_decay_steps=72, curve_bin_slots=6)
    if arms is not None:
        assert arms and len(set(arms)) == len(arms) and set(arms) <= set(p["arms"]), "Invalid arm partition"
        p["arms"] = [a for a in ["C-R", "C-N"] if a in arms]
    if seeds is not None:
        assert seeds and len(set(seeds)) == len(seeds) and set(seeds) <= set(p["training_seeds"]), "Invalid seed partition"
        p["training_seeds"] = sorted(seeds)
    count = len(p["arms"]) * len(p["training_seeds"])
    p.update(diagnostic=diagnostic, runtime=setup(device), source_sha256=source_identity()["sha256"],
             return_contract=RETURN_CONTRACT, initialization="fresh; same seed => identical initial CR/CN weights",
             best_selection="maximum fixed 5-trace validation mean utility including step0, exact ties earliest",
             checkpoint_selection_uses_test=False, automatic_extension=False,
             time_limit="S1-S3 done=terminated|truncated; S4 done=terminated; next state BEFORE reset",
             replay_policy="clear on physical task change plus explicit S4 entry; retain optimizer/weights/targets/epsilon",
             planned_training_steps=count * p["total_steps"],
             planned_validation_steps=count * len(p["evaluation_steps"]) * len(p["validation_pool"]) * p["full_horizon"])
    p["best_selection"] = p["best_selection"].replace("5-trace", f"{len(p['validation_pool'])}-trace")
    p["schedules"] = {}
    for seed in p["training_seeds"]:
        # Draw all exogenous game parameters before either arm starts.
        scenario_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x5343454E]))
        battery_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x42415454]))
        s4_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x31323030]))
        order, count = [], p["total_steps"] // p["early_horizon"]
        while len(order) < count:
            order.extend(map(int, scenario_rng.permutation(p["train_pool"])))
        s4_order = list(map(int, s4_rng.permutation(p["train_pool"])))
        schedule, step = [], 0
        while step < p["total_steps"]:
            condition = condition_for(step, "C-R", p)
            h = p["full_horizon"] if condition["stage"] == "S4" else p["early_horizon"]
            assert step + h <= next(e for e in p["stage_ends"] if step < e)
            eid = len(schedule)
            scenario = (s4_order[((step - p["stage_ends"][2]) // h) % len(s4_order)]
                        if condition["stage"] == "S4" else order[eid])
            schedule.append(dict(episode=eid, start=step, horizon=h, condition=condition,
                                 scenario_seed=int(scenario), assignment_seed=int(np.random.SeedSequence(
                                     [seed, eid, 0x494453]).generate_state(1)[0]),
                                 residual_battery=float(battery_rng.uniform(*p["train_residual_battery_range"])),
                                 diagnostic=False))
            step += h
        boundaries = {0} | {s["start"] + s["horizon"] for s in schedule}
        assert set(p["evaluation_steps"]) <= boundaries
        p["schedules"][str(seed)] = schedule
    benchmark = read_json(ROOT / "configs/bs5_return_v4_benchmark.json")
    assert not (set(p["train_pool"]) | set(p["validation_pool"])) & set(benchmark["test_scenarios"] + benchmark["reserved_scenarios"])
    p["benchmark_plan_sha256"] = canonical_hash(benchmark)
    return p


def prepare(root, device="auto", diagnostic=False, arms=None, seeds=None):
    root = Path(root).resolve()
    assert not root.exists(), "Existing output: use run/resume, never overwrite."
    p = build_protocol(device, diagnostic, arms, seeds)
    root.mkdir(parents=True)
    atomic_json(root / "protocol.json", p)
    atomic_json(root / "benchmark_plan.json", read_json(ROOT / "configs/bs5_return_v4_benchmark.json"))
    freeze_source(root, source_identity())
    atomic_json(root / "status.json", dict(status="prepared", diagnostic=diagnostic, training_steps=0))
    (root / "runs").mkdir()
    return p


def shard_label(p):
    return "_".join([*(a.replace("-", "") for a in p["arms"]), *(str(s) for s in p["training_seeds"])])


def training_archive_names(p):
    full = len(p["arms"]) == 2 and len(p["training_seeds"]) == 2
    suffix = "" if p["diagnostic"] or full else "__" + shard_label(p)
    return {k: k + suffix + ".zip" for k in ["training_resume", "training_for_benchmark"]}


def models(learner):
    return {k: getattr(learner, k).state_dict() for k in ["agent", "mixer", "target_agent", "target_mixer"]}


def progress_path(run):
    commits = sorted((Path(run) / "commits").glob("*.json"))
    return commits[-1] if commits else Path(run) / "progress.json"


def read_progress(run):
    return read_json(progress_path(run))


def read_selection(run):
    versions = sorted((Path(run) / "selections").glob("*.json"))
    return read_json(versions[-1] if versions else Path(run) / "checkpoint_selection.json")


def commit_progress(run, progress):
    # Immutable transaction records are authoritative; progress.json is a view.
    path = run / "commits" / f"{progress['steps']:06d}_{len(progress['validations']):04d}.json"
    assert not path.exists(), "Duplicate progress transaction."
    atomic_json(path, progress)
    atomic_json(run / "progress.json", progress)


def save_state(run, learner, replay, exploration, progress, p):
    progress["resume_file"] = f"resume/{progress['steps']:06d}.pt"
    data = dict(protocol_sha256=canonical_hash(p), progress=deepcopy(progress), models=models(learner),
                optimizer=learner.optimizer.state_dict(), replay=pack_replay(replay),
                exploration=exploration.bit_generator.state, rng=rng_state())
    started = time.monotonic()
    snapshot = run / progress["resume_file"]
    assert not snapshot.exists(), "Never overwrite a committed resume generation."
    atomic_torch(snapshot, data)
    with snapshot.open("rb") as stream:
        import os
        os.fsync(stream.fileno())
    progress["last_checkpoint_s"] = time.monotonic() - started
    commit_progress(run, progress)
    # Keep the latest two full optimizer/replay states; all inference
    # checkpoints, raw games and immutable commit receipts remain retained.
    for obsolete in sorted((run / "resume").glob("*.pt"))[:-2]:
        obsolete.unlink()


def init_or_load(root, p, arm, seed):
    run = root / "runs" / f"{arm}_seed{seed}"
    run.mkdir(exist_ok=True)
    seed_all(seed)
    spec = p["schedules"][str(seed)][0]
    learner = BS5CurriculumQMIX(env_for(p, spec, root / "traces", arm), lr=p["lr"], gamma=p["gamma"], device=p["runtime"]["device"])
    replay = CurriculumReplay(p["replay_capacity"], seed + 20000)
    exploration = np.random.default_rng(np.random.SeedSequence([seed, 0x504F4C49]))
    progress = dict(arm=arm, seed=seed, steps=0, episodes=0, updates=0, lineage=0, signature=None,
                    games=[], validations=[], best_step=None, best_utility=None,
                    initial_weights_sha256=weights_hash(learner), max_unit_s=0., last_checkpoint_s=0.)
    snapshots = sorted((run / "resume").glob("*.pt"))
    if snapshots:
        # Only load trusted output produced by this runner, never arbitrary third-party pickle.
        state = torch.load(snapshots[-1], map_location=learner.device, weights_only=False)
        assert state["protocol_sha256"] == canonical_hash(p)
        for name, value in state["models"].items():
            getattr(learner, name).load_state_dict(value)
        learner.optimizer.load_state_dict(state["optimizer"])
        replay = unpack_replay(state["replay"])
        exploration.bit_generator.state = state["exploration"]
        restore_rng(state["rng"])
        progress = state["progress"]
        if progress_path(run).exists():
            current = read_progress(run)
            if current["steps"] == progress["steps"]:
                assert current["games"] == progress["games"] and current["updates"] == progress["updates"]
                progress = current  # Validation commits can be newer; weights/replay unchanged.
            else:
                assert current["steps"] < progress["steps"], "Resume checkpoint older than committed training."
                commit_progress(run, progress)
        atomic_json(run / "progress.json", progress)
    else:
        assert not (run / "progress.json").exists(), "Missing resume checkpoint."
        Budget(run / "budget_ledger.json", dict(train=p["total_steps"],
               validation=len(p["evaluation_steps"]) * len(p["validation_pool"]) * p["full_horizon"]))
        save_state(run, learner, replay, exploration, progress, p)
    budget = Budget(run / "budget_ledger.json")
    expected = dict(train=progress["steps"], validation=len(progress["validations"]) * p["full_horizon"])
    for phase, n in expected.items():
        row = budget.data[phase]
        assert row["charged"] == row["observed"] == n, (
            f"Uncommitted {phase} compute: preserve output/ledger; no silent replay. "
            "Soft stops are resumable; a forced kill within a game requires audit.")
    return run, learner, replay, exploration, progress, budget


def checkpoint(run, learner, progress, p):
    cp = run / "checkpoints" / f"{progress['steps']:06d}.pt"
    cp.parent.mkdir(exist_ok=True)
    payload = dict(agent=learner.agent.state_dict(), mixer=learner.mixer.state_dict(),
                   step=progress["steps"], arm=progress["arm"], training_seed=progress["seed"],
                   return_version=RETURN_VERSION, protocol_sha256=canonical_hash(p),
                   weights_sha256=weights_hash(learner))
    if not cp.exists():
        atomic_torch(cp, payload)
    return cp


def validation_unit(root, run, p, learner, progress, budget):
    step = progress["steps"]
    done = [v["scenario"] for v in progress["validations"] if v["step"] == step]
    if len(done) == len(p["validation_pool"]):
        return False
    scenario = p["validation_pool"][len(done)]
    spec = dict(condition=full_condition(p), horizon=p["full_horizon"], scenario_seed=scenario,
                assignment_seed=scenario, residual_battery=p["validation_residual_battery"], diagnostic=True)
    cp = checkpoint(run, learner, progress, p)
    old_rng, old_mode, before = rng_state(), learner.agent.training, weights_hash(learner)
    learner.agent.eval()
    rng = np.random.default_rng(np.random.SeedSequence([scenario, 0x4556414C]))
    directory = run / "validation" / f"{step:06d}" / str(scenario)
    started = time.monotonic()
    try:
        index = rollout(directory, p, spec, progress["arm"], root / "traces", budget, "validation",
                        lambda env, out: (act(learner, out, 0., rng), 0., 0))
    finally:
        restore_rng(old_rng)
        learner.agent.train(old_mode)
    assert weights_hash(learner) == before
    row = dict(step=step, scenario=scenario, index=(directory / "index.json").relative_to(run).as_posix(),
               utility=index["metrics"]["utility"], weights_sha256=before, checkpoint_sha256=file_hash(cp))
    progress["validations"].append(row)
    progress["max_unit_s"] = max(progress["max_unit_s"], time.monotonic() - started)
    own = [r for r in progress["validations"] if r["step"] == step]
    if len(own) == len(p["validation_pool"]):
        value = float(np.mean([r["utility"] for r in own]))
        if progress["best_utility"] is None or value > progress["best_utility"]:
            progress["best_utility"], progress["best_step"] = value, step
        curve = []
        for s in p["evaluation_steps"]:
            rows = [r for r in progress["validations"] if r["step"] == s]
            if len(rows) == len(p["validation_pool"]):
                curve.append(dict(step=s, utility=float(np.mean([r["utility"] for r in rows])),
                                  weights_sha256=rows[0]["weights_sha256"], checkpoint_sha256=rows[0]["checkpoint_sha256"]))
        selection = dict(rule=p["best_selection"], best_step=progress["best_step"],
                         best_utility=progress["best_utility"], last_step=step, curve=curve)
        atomic_json(run / "selections" / f"{step:06d}.json", selection)
        atomic_json(run / "checkpoint_selection.json", selection)
    commit_progress(run, progress)
    assert read_json(run / "progress.json")["validations"] == progress["validations"]
    print(json.dumps(dict(phase="validation_committed", arm=progress["arm"], seed=progress["seed"], **row)), flush=True)
    return True


def train_unit(root, run, p, learner, replay, exploration, progress, budget):
    spec = p["schedules"][str(progress["seed"])][progress["episodes"]]
    assert spec["start"] == progress["steps"]
    c, stage = spec["condition"], spec["condition"]["stage"]
    sig = canonical_hash(task_signature(c))
    if progress["signature"] != sig or spec["start"] == p["stage_ends"][2]:
        replay = CurriculumReplay(p["replay_capacity"], progress["seed"] + 20000 + progress["lineage"])
        progress["lineage"] += 1
        progress["signature"] = sig
    updates = []
    def choose(env, out):
        epsilon = 1 + (p["epsilon_end"] - 1) * min(progress["steps"] / p["epsilon_decay_steps"], 1)
        return act(learner, out, epsilon, exploration), epsilon, 0
    def transition(out, actions, nxt, done, epsilon):
        step = progress["steps"]
        replay.add(CurriculumTransition(out["obs"].copy(), out["state"].copy(), np.asarray(actions),
            nxt["reward"], nxt["obs"].copy(), nxt["state"].copy(), nxt["action_mask"].copy(), float(done),
            out["action_mask"].copy(), stage, c["substage"], step, progress["lineage"]))
        progress["steps"] += 1
        if len(replay) >= max(p["warmup"], p["batch_size"]) and progress["steps"] % p["update_every"] == 0:
            batch, audit = sample_audit(replay, p["batch_size"])
            result = learner.train_batch(batch)
            progress["updates"] += 1
            updates.append(dict(step=progress["steps"], update=progress["updates"], epsilon=epsilon,
                                lineage=progress["lineage"], replay_size=len(replay), batch=audit, **result))
        if progress["steps"] % p["target_every"] == 0:
            learner.update_targets()
    started = time.monotonic()
    directory = run / "training" / f"{spec['episode']:04d}"
    index = rollout(directory, p, spec, progress["arm"], root / "traces", budget, "train", choose, transition)
    save_gz(directory / "updates.json.gz", updates)
    progress["games"].append(dict(index=(directory / "index.json").relative_to(run).as_posix(),
                                  updates_sha256=file_hash(directory / "updates.json.gz"),
                                  start=spec["start"], end=progress["steps"], stage=c["substage"],
                                  utility=index["metrics"]["utility"], runtime_s=time.monotonic() - started))
    progress["episodes"] += 1
    progress["max_unit_s"] = max(progress["max_unit_s"], time.monotonic() - started)
    save_state(run, learner, replay, exploration, progress, p)
    print(json.dumps(dict(phase="training_committed", arm=progress["arm"], seed=progress["seed"],
                         step=progress["steps"], stage=c["substage"], updates=progress["updates"], utility=index["metrics"]["utility"])), flush=True)
    return replay


def all_complete(root, p):
    for seed in p["training_seeds"]:
        for arm in p["arms"]:
            path = root / "runs" / f"{arm}_seed{seed}" / "progress.json"
            if not path.exists():
                return False
            progress = read_progress(path.parent)
            if progress["steps"] != p["total_steps"] or len(progress["validations"]) != len(p["evaluation_steps"]) * len(p["validation_pool"]):
                return False
    return True


def run(root, max_seconds=10 * 3600, max_units=None):
    global STOP_REQUESTED
    root = Path(root).resolve()
    p = read_json(root / "protocol.json")
    assert setup(p["runtime"]["device"]) == p["runtime"], "Runtime/hardware changed: exact resume is blocked."
    assert source_identity()["sha256"] == p["source_sha256"], "Source changed. Use the same Dataset."
    assert p["benchmark_plan_sha256"] == canonical_hash(read_json(root / "benchmark_plan.json"))
    started, units, STOP_REQUESTED = time.monotonic(), 0, False
    def stop_signal(signum, frame):
        global STOP_REQUESTED
        STOP_REQUESTED = True
        print("Stop requested: finishing and committing the current game.", flush=True)
    old_handlers = {}
    for signum in [signal.SIGINT, signal.SIGTERM]:
        old_handlers[signum] = signal.signal(signum, stop_signal)
    status = "paused_at_game_boundary"
    with experiment_lock(root):
        try:
            for seed in p["training_seeds"]:
                for arm in p["arms"]:
                    run_path, learner, replay, exploration, progress, budget = init_or_load(root, p, arm, seed)
                    while True:
                        at_eval = progress["steps"] in p["evaluation_steps"]
                        eval_done = sum(v["step"] == progress["steps"] for v in progress["validations"]) == len(p["validation_pool"])
                        if progress["steps"] == p["total_steps"] and eval_done:
                            break
                        remaining = max_seconds - (time.monotonic() - started)
                        reserve = max(300., 1.5 * progress["max_unit_s"] + 2 * progress.get("last_checkpoint_s", 0))
                        if STOP_REQUESTED or (max_units is not None and units >= max_units) or remaining < reserve:
                            return
                        if at_eval and not eval_done:
                            validation_unit(root, run_path, p, learner, progress, budget)
                        else:
                            replay = train_unit(root, run_path, p, learner, replay, exploration, progress, budget)
                        units += 1
            assert all_complete(root, p), "Final durable run manifests are incomplete."
            status = "complete"
        except BaseException as exc:
            status = "partial_error"
            atomic_json(root / "error.json", dict(error=repr(exc), instruction="Preserve all output and ledger. Never reset counters or silently replay in-flight compute."))
            raise
        finally:
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)
            completed_steps = sum(read_progress(f.parent)["steps"] for f in (root / "runs").glob("*/progress.json"))
            atomic_json(root / "status.json", dict(status=status, diagnostic=p["diagnostic"], training_steps=completed_steps,
                                                  planned_training_steps=p["planned_training_steps"]))
            history = read_json(root / "runtime_sessions.json") if (root / "runtime_sessions.json").exists() else []
            history.append(dict(wall_s=time.monotonic() - started, committed_units=units, status=status))
            atomic_json(root / "runtime_sessions.json", history)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["prepare", "run", "export"])
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--diagnostic", action="store_true")
    ap.add_argument("--arms", help="Comma-separated partition: C-R or C-N; omitted means both")
    ap.add_argument("--seeds", help="Comma-separated partition: 551,552; omitted means both")
    ap.add_argument("--max-seconds", type=float, default=10 * 3600)
    ap.add_argument("--max-units", type=int)
    ap.add_argument("--deep-planner", action="store_true", help="Offline export audit only; no extra rollout")
    a = ap.parse_args()
    if a.command == "prepare":
        prepare(a.output, a.device, a.diagnostic,
                a.arms.split(",") if a.arms else None,
                [int(s) for s in a.seeds.split(",")] if a.seeds else None)
    elif a.command == "run":
        run(a.output, a.max_seconds, a.max_units)
    else:
        from scripts.analysis.bs5_v4_fresh_report import export_training
        print(export_training(a.output, deep=a.deep_planner))


if __name__ == "__main__":
    main()
