"""Fixed-budget curriculum pilot. --prepare runs bounded smoke, never the main pilot."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import itertools
import json
import os
from pathlib import Path
import random
import shutil
import statistics
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

from scripts.analysis.bs5_bundle import atomic_json, canonical_hash, file_hash, read_json
from scripts.analysis.analyze_bs5_curriculum_bundle import analyze, episode_metrics
from scripts.training.run_bs5_launch_ab import (
    capture_decision,
    experiment_lock,
    freeze_source,
    load_state,
    save_state,
    serial,
    source_identity,
    step_record,
    utc_now,
    weights_hash,
    write_trace,
)
from src.envs.bs5_launch_ab_env import Geometry, ROUTER_VERSION, score_network
from src.envs.bs5_curriculum_env import BS5CurriculumEnv, VERSION, condition_for, task_signature
from src.rl.bs5_curriculum_qmix import BS5CurriculumQMIX
from src.rl.bs5_curriculum_replay import CurriculumReplay, CurriculumTransition

ARMS = ("C-R", "D-R", "C-N", "D-N")


def protocol_new(config, identity, *, smoke=False):
    p = deepcopy(config)
    if smoke:
        p.update(
            training_seeds=[955551, 955552],
            horizon=6,
            stage_ends=[48, 96, 192, 384],
            requested_steps_per_run=384,
            eval_every=48,
            warmup=8,
            batch_size=8,
            target_every=24,
            epsilon_decay_steps=288,
            validation_scenario_seeds=[180001, 180002],
            stage_validation_scenario_seeds=[181001, 181002],
            train_scenario_seeds=[182001, 182004],
            fixture_scenario_seeds=[183001, 183004],
            replay_capacity=1000,
        )
    steps = p.pop("requested_steps_per_run")
    assert p["stage_ends"][-1] == steps
    assert all(e % p["horizon"] == 0 for e in p["stage_ends"])
    assert (p["stage_ends"][1] - p["stage_ends"][0]) % (2 * p["horizon"]) == 0
    assert (p["stage_ends"][2] - p["stage_ends"][1]) % (4 * p["horizon"]) == 0
    p.update(
        steps_per_run=steps,
        arms=list(ARMS),
        source_sha256=identity["sha256"],
        version=VERSION,
        router_version=ROUTER_VERSION,
        scope="smoke_not_main" if smoke else "validation_pilot_not_final_test",
        common_evaluation_steps=list(range(0, steps + 1, p["eval_every"])),
        state_dim=111,
        obs_dim=116,
        n_actions=10,
        execution_environment=dict(
            python=sys.version,
            torch=torch.__version__,
            numpy=np.__version__,
            device=p["device"],
            logical_cpus=os.cpu_count(),
        ),
        best_selection="maximum common-cadence full-task mean utility including step0; earlier exact tie",
        replay_lineage="clear at physical task changes only; keep weights/Adam/targets/global epsilon",
        time_limit="source convention: horizon ends set done=1; no bootstrap across reset",
        gate_policy="diagnostic_only_never_changes_fixed_schedule",
    )
    for source, dest in (
        ("train_scenario_seeds", "train_pool"),
        ("validation_scenario_seeds", "validation_pool"),
        ("fixture_scenario_seeds", "fixture_pool"),
        ("stage_validation_scenario_seeds", "stage_validation_pool"),
    ):
        p[dest] = list(range(p[source][0], p[source][1] + 1))
    pools = [
        set(p[k])
        for k in ("train_pool", "validation_pool", "fixture_pool", "stage_validation_pool")
    ]
    assert all(not a & b for a, b in itertools.combinations(pools, 2)), "scenario pool overlap"
    boundaries = []
    previous = condition_for(0, "C-R", p)
    for step in range(p["horizon"], steps, p["horizon"]):
        current = condition_for(step, "C-R", p)
        if current != previous:
            boundaries.append(step)
        previous = current
    p["condition_boundaries"] = boundaries
    p["training_conditions"] = {}
    for arm in ARMS:
        spans = []
        for start in range(0, steps, p["horizon"]):
            c = condition_for(start, arm, p)
            if spans and spans[-1]["condition"] == c:
                spans[-1]["end"] = start + p["horizon"]
            else:
                spans.append(dict(start=start, end=start + p["horizon"], condition=c))
        p["training_conditions"][arm] = spans
    p["evaluation_steps"] = sorted(set(p["common_evaluation_steps"] + boundaries))
    p["scenario_schedule"] = {}
    for seed in p["training_seeds"]:
        schedule_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x5343454E]))
        battery_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x42415454]))
        pools_order = {}
        count = steps // p["horizon"]
        for kind in ("train", "fixture"):
            order = []
            while len(order) < count:
                order.extend(map(int, schedule_rng.permutation(p[f"{kind}_pool"])))
            pools_order[kind] = order
        p["scenario_schedule"][str(seed)] = [
            dict(
                scenario_seed=pools_order["train"][eid],
                fixture_seed=pools_order["fixture"][eid],
                residual_battery=float(battery_rng.uniform(*p["train_residual_battery_range"])),
                assignment_seed=int(
                    np.random.SeedSequence([seed, eid, 0x494453]).generate_state(1)[0]
                ),
            )
            for eid in range(count)
        ]
    return p


def registry_new(p):
    return dict(
        status="prepared",
        protocol_sha256=canonical_hash(p),
        created_utc=utc_now(),
        runs={
            f"{arm}_seed{seed}": dict(status="not_started", steps=0)
            for seed in p["training_seeds"]
            for arm in ARMS
        },
        attempts=[],
    )


def make_env(condition, p, relay, residual=30.0):
    return BS5CurriculumEnv(
        condition,
        allow_relay=relay,
        horizon=p["horizon"],
        residual_battery=residual,
        fixture=p["fixture"],
    )


def act(learner, out, epsilon, rng):
    with torch.no_grad():
        q = learner.agent(torch.as_tensor(out["obs"], device=learner.device)).cpu().numpy()
    actions, entropy = [], []
    for i, mask in enumerate(out["action_mask"]):
        valid = np.flatnonzero(mask)
        values = q[i, valid]
        if rng.random() < epsilon:
            actions.append(int(rng.choice(valid)))
        else:
            actions.append(int(valid[np.argmax(values)]))
        exp = np.exp(values - values.max())
        prob = exp / exp.sum()
        entropy.append(float(-(prob * np.log(np.maximum(prob, 1e-30))).sum()))
    values = q[out["action_mask"].astype(bool)]
    return actions, dict(
        q_valid_mean=float(values.mean()),
        q_valid_min=float(values.min()),
        q_valid_max=float(values.max()),
        q_softmax_entropy=statistics.fmean(entropy),
        agent_forward_calls=1,
        recurrent=False,
    )


def record(
    env,
    geometry,
    decision,
    mask,
    actions,
    nxt,
    p,
    arm,
    seed,
    scenario,
    slot,
    step,
    condition,
    q,
    *,
    evaluation=False,
    kind="train",
):
    row = step_record(
        env,
        decision,
        mask,
        actions,
        nxt,
        geometry,
        arm=arm,
        seed=seed,
        scenario=scenario,
        slot=slot,
        checkpoint_step=step if evaluation else None,
        oracle=evaluation and (slot - 1) % p["oracle_every_slots"] == 0,
    )
    row.update(
        stage_id=condition["stage"],
        substage_id=condition["substage"],
        task_condition=condition,
        active_ids=env.active_ids.tolist(),
        next_action_mask=nxt["action_mask"].tolist(),
        terminated=bool(nxt["terminated"]),
        truncated=bool(nxt["truncated"]),
        q_summary=q,
        eval_kind=kind,
        recurrent_reset=slot == 1,
        exogenous_sha256=canonical_hash(
            [row["service"]["ue_positions"], row["service"]["ue_demand_mbps"]]
        ),
        ue_ids=list(range(90)),
        failed_bs_gateway_excluded=True,
    )
    row["surviving_bs_ids"] = [i for i in range(6) if i != 5]
    row["charging_station_cells"] = env.charging_stations.tolist()
    return row


def evaluate(learner, p, output, arm, seed, step, *, condition=None, label="full"):
    started = time.monotonic()
    full = condition is None
    c = condition_for(0, "D-R", p) if full else deepcopy(condition)
    env = make_env(c, p, arm.endswith("R"), p["validation_residual_battery"])
    geometry = Geometry(env)
    pool = p["validation_pool"] if full else p["stage_validation_pool"]
    directory = output / ("evaluations" if full else "stage_evaluations") / f"{step:06d}"
    if not full:
        directory /= label
    results, gate_rows, shadow_seconds = [], [], 0.0
    previous_mode = learner.agent.training
    learner.agent.eval()
    for scenario in pool:
        out = env.reset(seed=scenario, assignment_seed=scenario, diagnostic=True)
        initial_ue = canonical_hash([env.ue_positions.tolist(), [4.0] * 90])
        initial_positions = env.uav_positions.copy()
        residual = dict(
            id=int(env.residual_uav_idx), energy=float(env.battery[env.residual_uav_idx])
        )
        role_values, role_assignments = [], []
        if not full:
            for roles in itertools.product(
                (0, 1) if arm.endswith("R") else (0,), repeat=len(env.active_ids)
            ):
                env.roles[env.active_ids] = roles
                initial_net = score_network(env, arm.endswith("R"))
                role_values.append(initial_net["utility"])
                role_assignments.append(
                    dict(
                        roles=list(roles),
                        utility=initial_net["utility"],
                        ue_rate_mbps=initial_net["ue_rate_mbps"].tolist(),
                    )
                )
            env.roles[:] = 0
            initial_direct = score_network(env, False)["utility"]
            initial_optimum = max(role_values)
            env._refresh_request_state()
            out.update(obs=env.get_obs(), state=env.get_state(), action_mask=env.get_action_mask())
        rows = []
        rng = np.random.default_rng(np.random.SeedSequence([scenario, 0x4556414C]))
        for t in range(p["horizon"]):
            mask, decision = out["action_mask"].copy(), capture_decision(env)
            actions, q = act(learner, out, 0.0, rng)
            nxt = env.step(actions)
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
                    scenario,
                    t + 1,
                    step,
                    c,
                    q,
                    evaluation=True,
                    kind=label,
                )
            )
            shadow_seconds += time.monotonic() - begin
            out = nxt
        metrics = episode_metrics(rows)
        packet = dict(
            trace=rows,
            metrics=metrics,
            initial_residual=residual,
            initial_exogenous_sha256=initial_ue,
            kind=label,
            condition=c,
            stage_at_checkpoint=condition_for(min(step, p["steps_per_run"] - 1), arm, p),
        )
        if not full:
            packet.update(
                initial_positions=initial_positions.tolist(),
                initial_role_assignments=role_assignments,
            )
            gain = initial_optimum - initial_direct
            valid = gain > p["gate_tolerance"]
            ratio = (metrics["utility"] - initial_direct) / gain if valid else None
            if arm.endswith("N"):
                ratio = (
                    metrics["utility"] / initial_optimum
                    if initial_optimum > p["gate_tolerance"]
                    else None
                )
            moved = any(r["service"]["uav_positions"] != initial_positions.tolist() for r in rows)
            gate = dict(
                scenario=scenario,
                initial_all_access=initial_direct,
                initial_feasible_optimum=initial_optimum,
                attainable_net_gain=gain,
                positive_gain_eligible=valid,
                attainment_ratio=ratio,
                ratio_pass=ratio is not None and ratio >= p["gate_role_fraction"],
                moved=moved,
                movement_pass=moved
                and metrics["utility"] > initial_optimum + p["gate_move_improvement"],
                all_scenarios_retained=True,
            )
            packet["gate"] = gate
            gate_rows.append(gate)
        write_trace(directory / f"{scenario}.json.gz", packet)
        results.append(metrics["utility"])
    learner.agent.train(previous_mode)
    result = dict(
        step=step,
        kind=label,
        utility=statistics.fmean(results),
        runtime_s=time.monotonic() - started,
        shadow_logging_s=shadow_seconds,
        slots=len(pool) * p["horizon"],
        actor_forward_calls=len(pool) * p["horizon"],
        eligible_for_best=full and step in p["common_evaluation_steps"],
    )
    if gate_rows:
        eligible = [g for g in gate_rows if g["attainment_ratio"] is not None]
        fraction = sum(g["ratio_pass"] for g in eligible) / len(eligible) if eligible else None
        result.update(
            gate_rows=gate_rows,
            eligible_episodes=len(eligible),
            total_episodes=len(gate_rows),
            role_pass_fraction=fraction,
            role_gate_pass=fraction is not None and fraction >= p["gate_episode_fraction"],
            movement_pass_fraction=sum(g["movement_pass"] for g in gate_rows) / len(gate_rows),
            gate_controls_transition=False,
            no_relay_s1_gate_is_structural=arm.endswith("N") and c["stage"] == "S1",
        )
    atomic_json(directory / "summary.json", result)
    return result


def archive_uncommitted(output, commit):
    from scripts.training.run_bs5_launch_ab import archive_uncommitted as original

    original(output, commit)
    candidates = []
    for sub in ("events",):
        candidates += [
            p for p in (output / sub).glob("*.json") if int(p.stem) >= commit["episodes"]
        ]
    candidates += [
        p for p in (output / "stage_evaluations").glob("*") if int(p.name) > commit["steps"]
    ]
    if candidates:
        root = output / "aborted_attempts" / str(time.time_ns())
        for path in candidates:
            target = root / path.relative_to(output)
            target.parent.mkdir(parents=True, exist_ok=True)
            path.rename(target)


def run_one(root, p, registry, arm, seed, *, resume=False, export=True, stop_after=None):
    name = f"{arm}_seed{seed}"
    if registry["runs"][name]["status"] == "complete":
        return
    output = root / "runs" / name
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    env = make_env(condition_for(0, arm, p), p, arm.endswith("R"))
    learner = BS5CurriculumQMIX(env, lr=p["lr"], gamma=p["gamma"], device=p["device"])
    exploration = np.random.default_rng(np.random.SeedSequence([seed, 0x504F4C49]))
    replay = CurriculumReplay(p["replay_capacity"], seed + 20000)
    meta = dict(
        run=name,
        arm=arm,
        seed=seed,
        initial_weights_sha256=weights_hash(learner),
        source_sha256=p["source_sha256"],
        resolved_config=serial(asdict(env.cfg)),
        state_dim=111,
        obs_dim=116,
        n_actions=10,
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
        current_task=None,
        seen=[],
        stage_budgets={},
        substage_budgets={},
        train_seconds=0.0,
        update_seconds=0.0,
        eval_seconds=0.0,
        shadow_seconds=0.0,
        io_seconds=0.0,
        evaluations=[],
        stage_evaluations=[],
        gate_history=[],
    )
    if (output / "resume.pt").exists():
        assert resume, "Use --resume"
        commit, replay = load_state(output / "resume.pt", learner, exploration)
        atomic_json(output / "commit.json", commit)
        archive_uncommitted(output, commit)
    else:
        ev = evaluate(learner, p, output, arm, seed, 0)
        commit.update(
            best_utility=ev["utility"],
            evaluations=[ev],
            eval_seconds=ev["runtime_s"],
            shadow_seconds=ev["shadow_logging_s"],
        )
        if arm.startswith("C"):
            gate = evaluate(
                learner,
                p,
                output,
                arm,
                seed,
                0,
                condition=condition_for(0, arm, p),
                label="current",
            )
            commit["stage_evaluations"].append(gate)
            commit["eval_seconds"] += gate["runtime_s"]
            commit["shadow_seconds"] += gate["shadow_logging_s"]
        save_state(output, learner, replay, exploration, commit)
    registry["runs"][name] = dict(status="running", steps=commit["steps"])
    attempt = dict(
        run=name,
        started_utc=utc_now(),
        from_step=commit["steps"],
        status="running",
        executed_train_steps=0,
        learner_updates=0,
    )
    registry["attempts"].append(attempt)
    atomic_json(root / "run_registry.json", registry)
    try:
        while commit["steps"] < p["steps_per_run"]:
            eid, start = commit["episodes"], commit["steps"]
            c = condition_for(start, arm, p)
            signature = serial(task_signature(c))
            events = []

            def event(kind, **extra):
                payload = dict(
                    kind=kind,
                    timestamp=utc_now(),
                    episode=eid,
                    global_env_step=commit["steps"],
                    stage=c["stage"],
                    substage=c["substage"],
                )
                payload.update(extra)
                events.append(payload)

            if commit["current_task"] != signature:
                before = len(replay)
                if commit["current_task"] is not None:
                    commit["lineage"] += 1
                    replay = CurriculumReplay(
                        p["replay_capacity"], seed + 20000 + commit["lineage"]
                    )
                event(
                    "task_change",
                    old_task=commit["current_task"],
                    new_task=signature,
                    replay_discarded=before,
                    replay_lineage=commit["lineage"],
                    keep_weights=True,
                    keep_optimizer=True,
                    keep_targets=True,
                    epsilon_reset=False,
                    condition=c,
                )
                commit["current_task"] = signature
            env = make_env(c, p, arm.endswith("R"))
            spec = p["scenario_schedule"][str(seed)][eid]
            scenario = spec["fixture_seed"] if c["stage"] in {"S1", "S2"} else spec["scenario_seed"]
            env.cfg.residual_initial_battery = spec["residual_battery"]
            out = env.reset(seed=scenario, assignment_seed=spec["assignment_seed"])
            geometry = Geometry(env)
            key = f"{c['substage']}:{scenario}"
            reasons = []
            if key not in commit["seen"]:
                reasons.append("first_scenario_in_substage")
                commit["seen"].append(key)
            if eid % p["train_trace_every_episodes"] == 0:
                reasons.append("periodic")
            if start == 0 or start in p["condition_boundaries"]:
                reasons.append("first_after_boundary")
            initial_residual = dict(id=int(env.residual_uav_idx), energy=spec["residual_battery"])
            event(
                "episode_start",
                condition=c,
                active_ids=env.active_ids.tolist(),
                starts=env.uav_positions.tolist(),
                initial_residual=initial_residual,
                capture_reason=reasons,
                scenario=scenario,
            )
            rows, updates = [], []
            started = time.monotonic()
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
                        c["stage"],
                        c["substage"],
                        commit["steps"],
                        commit["lineage"],
                    )
                )
                commit["steps"] += 1
                attempt["executed_train_steps"] += 1
                if (
                    len(replay) >= max(p["warmup"], p["batch_size"])
                    and commit["steps"] % p["update_every"] == 0
                ):
                    begin = time.monotonic()
                    update = learner.train_batch(replay.sample(p["batch_size"]))
                    commit["update_seconds"] += time.monotonic() - begin
                    commit["updates"] += 1
                    attempt["learner_updates"] += 1
                    updates.append(
                        dict(
                            step=commit["steps"],
                            update=commit["updates"],
                            epsilon=epsilon,
                            replay_size=len(replay),
                            replay_stage_composition=replay.composition(),
                            lineage=commit["lineage"],
                            stage=c["stage"],
                            substage=c["substage"],
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
                        arm,
                        seed,
                        scenario,
                        t + 1,
                        commit["steps"],
                        c,
                        q,
                    )
                )
                out = nxt
            assert out["truncated"]
            commit["train_seconds"] += time.monotonic() - started
            for field, value in (
                ("stage_budgets", c["stage"]),
                ("substage_budgets", c["substage"]),
            ):
                commit[field][value] = commit[field].get(value, 0) + p["horizon"]
            ep = dict(
                episode_id=eid,
                stage=c["stage"],
                substage=c["substage"],
                start_step=start,
                end_step=commit["steps"],
                lineage=commit["lineage"],
                scenario_seed=scenario,
                assignment_seed=spec["assignment_seed"],
                residual=initial_residual,
                condition=c,
                source_sha256=p["source_sha256"],
                config_sha256=meta["config_sha256"],
                capture_reason=reasons,
                metrics=episode_metrics(rows),
                completion_status="complete",
            )
            atomic_json(output / "episodes" / f"{eid:06d}.json", ep)
            atomic_json(output / "updates" / f"{eid:06d}.json", updates)
            if reasons:
                write_trace(
                    output / "train_traces" / f"{eid:06d}.json.gz",
                    dict(trace=rows, metrics=ep["metrics"]),
                )
            commit["episodes"] += 1
            step = commit["steps"]
            if step in p["evaluation_steps"]:
                ev = evaluate(learner, p, output, arm, seed, step)
                commit["evaluations"].append(ev)
                commit["eval_seconds"] += ev["runtime_s"]
                commit["shadow_seconds"] += ev["shadow_logging_s"]
                if ev["eligible_for_best"] and ev["utility"] > commit["best_utility"]:
                    commit.update(best_step=step, best_utility=ev["utility"])
                if arm.startswith("C") and (c["stage"] != "S4" or step == p["steps_per_run"]):
                    gate = evaluate(
                        learner, p, output, arm, seed, step, condition=c, label="current"
                    )
                    commit["stage_evaluations"].append(gate)
                    commit["eval_seconds"] += gate["runtime_s"]
                    commit["shadow_seconds"] += gate["shadow_logging_s"]
                    history = [g for g in commit["gate_history"] if g["substage"] == c["substage"]]
                    consecutive = (
                        gate.get("role_gate_pass", False)
                        and bool(history)
                        and history[-1]["role_gate_pass"]
                    )
                    gate_event = dict(
                        step=step,
                        stage=c["stage"],
                        substage=c["substage"],
                        role_gate_pass=gate.get("role_gate_pass", False),
                        two_consecutive_role_pass=bool(consecutive),
                        movement_pass_fraction=gate.get("movement_pass_fraction"),
                        gate_criterion="movement" if c["stage"] == "S2" else "role",
                        gate_failed=(
                            gate["movement_pass_fraction"] < p["gate_episode_fraction"]
                            if c["stage"] == "S2"
                            else not gate.get("role_gate_pass", False)
                        ),
                        transition_reason="fixed_schedule_not_gate",
                    )
                    commit["gate_history"].append(gate_event)
                    event("gate_diagnostic", **gate_event)
                if arm.startswith("C") and step in {
                    p["stage_ends"][1],
                    p["stage_ends"][2],
                    p["steps_per_run"],
                }:
                    retention = evaluate(
                        learner,
                        p,
                        output,
                        arm,
                        seed,
                        step,
                        condition=condition_for(0, arm, p),
                        label="S1_retention",
                    )
                    commit["stage_evaluations"].append(retention)
                    commit["eval_seconds"] += retention["runtime_s"]
                    commit["shadow_seconds"] += retention["shadow_logging_s"]
                event("evaluation_commit", best_step=commit["best_step"], utility=ev["utility"])
            atomic_json(output / "events" / f"{eid:06d}.json", events)
            if step in p["evaluation_steps"]:
                begin = time.monotonic()
                save_state(output, learner, replay, exploration, commit)
                commit["io_seconds"] += time.monotonic() - begin
                registry["runs"][name] = dict(
                    status="complete" if step == p["steps_per_run"] else "running", steps=step
                )
                atomic_json(root / "run_registry.json", registry)
                print(
                    json.dumps(
                        dict(
                            run=name,
                            step=step,
                            stage=c["stage"],
                            utility=ev["utility"],
                            best_step=commit["best_step"],
                        )
                    ),
                    flush=True,
                )
                if export:
                    export_bundles(root)
                if stop_after == step:
                    raise KeyboardInterrupt("requested interruption at committed boundary")
        attempt.update(status="complete", ended_utc=utc_now())
        atomic_json(root / "run_registry.json", registry)
    except BaseException as exc:
        durable = read_json(output / "commit.json")
        attempt.update(
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            reason=f"{type(exc).__name__}: {exc}",
            ended_utc=utc_now(),
        )
        registry["runs"][name] = dict(status="incomplete", steps=durable["steps"])
        registry["status"] = "incomplete"
        atomic_json(root / "run_registry.json", registry)
        if export:
            export_bundles(root)
        raise


def run_matrix(root, *, resume=False, export=True):
    p = read_json(root / "protocol.json")
    assert source_identity()["sha256"] == p["source_sha256"], (
        "Source changed: prepare a new output; keep old source for resume"
    )
    registry = read_json(root / "run_registry.json")
    assert canonical_hash(p) == registry["protocol_sha256"]
    torch.set_num_threads(p["torch_threads"])
    registry["status"] = "running"
    atomic_json(root / "run_registry.json", registry)
    try:
        for seed in p["training_seeds"]:
            for arm in ARMS:
                run_one(root, p, registry, arm, seed, resume=resume, export=export)
        registry.update(status="complete", completed_utc=utc_now())
        atomic_json(root / "run_registry.json", registry)
        export_bundles(root)
    except BaseException:
        registry["status"] = "incomplete"
        atomic_json(root / "run_registry.json", registry)
        raise


def export_bundles(root):
    analyze(root, root / "analysis", check_manifest=False)
    registry = read_json(root / "run_registry.json")
    common = [
        (root / name, name)
        for name in ("protocol.json", "schema.json", "source_identity.json", "run_registry.json")
    ]
    for sub in ("preflight", "analysis"):
        common += [
            (path, path.relative_to(root).as_posix())
            for path in sorted((root / sub).rglob("*"))
            if path.is_file()
        ]
    full_members = []
    for name, entry in registry["runs"].items():
        if entry["status"] == "not_started":
            continue
        directory = root / "runs" / name
        commit = read_json(directory / "commit.json")
        common += [(directory / n, f"runs/{name}/{n}") for n in ("meta.json", "commit.json")]
        for sub in ("episodes", "updates", "events", "train_traces"):
            common += [
                (path, path.relative_to(root).as_posix())
                for path in sorted((directory / sub).glob("*"))
                if int(path.name.split(".")[0]) < commit["episodes"]
            ]
        for sub in ("evaluations", "stage_evaluations"):
            for step_dir in sorted((directory / sub).glob("*")):
                if int(step_dir.name) <= commit["steps"]:
                    common += [
                        (path, path.relative_to(root).as_posix())
                        for path in sorted(step_dir.rglob("*"))
                        if path.is_file()
                    ]
        full_members.append((directory / "resume.pt", f"runs/{name}/resume.pt"))
        full_members += [
            (path, path.relative_to(root).as_posix())
            for path in sorted((directory / "checkpoints").glob("*.pt"))
            if int(path.stem) <= commit["steps"]
        ]
        full_members += [
            (path, path.relative_to(root).as_posix())
            for path in sorted((directory / "aborted_attempts").rglob("*"))
            if path.is_file()
        ]
    destination = root / "exports"
    destination.mkdir(exist_ok=True)
    code = {
        "bs5_bundle.py": (ROOT / "scripts/analysis/bs5_bundle.py").read_bytes(),
        "analyze_bs5_curriculum_bundle.py": (
            ROOT / "scripts/analysis/analyze_bs5_curriculum_bundle.py"
        ).read_bytes(),
        "analyze_bundle.py": b"from analyze_bs5_curriculum_bundle import main\nif __name__ == '__main__': main()\n",
        "validate_bundle.py": b"from analyze_bs5_curriculum_bundle import main\nif __name__ == '__main__': main()\n",
        "README.md": b"Run: python -B analyze_bundle.py --root . --output ../rebuilt --require-complete\nNo checkpoints, torch, environment or network required. Outputs must be outside this extracted bundle.\nFull resume is at committed evaluation boundaries. Restore source/uav_thesis_source.zip as source_snapshot.zip in the experiment root when restoring repro_full.\n",
    }
    for full, name in ((False, "audit_light.zip"), (True, "repro_full.zip")):
        target = destination / name
        tmp = target.with_suffix(".zip.tmp")
        hashes = {}
        members = common + (
            full_members + [(root / "source_snapshot.zip", "source/uav_thesis_source.zip")]
            if full
            else []
        )
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=3) as z:
            for path, relative in members:
                z.write(
                    path,
                    relative,
                    compress_type=zipfile.ZIP_STORED
                    if path.suffix in {".gz", ".zip"}
                    else zipfile.ZIP_DEFLATED,
                )
                hashes[relative] = file_hash(path)
            import hashlib

            for relative, data in code.items():
                z.writestr(relative, data)
                hashes[relative] = hashlib.sha256(data).hexdigest()
            z.writestr("SHA256SUMS.json", json.dumps(hashes, indent=2))
        with zipfile.ZipFile(tmp) as z:
            assert z.testzip() is None
        tmp.replace(target)
    atomic_json(
        destination / "sizes.json",
        {f.name: dict(bytes=f.stat().st_size) for f in destination.glob("*.zip")},
    )


def preflight(root, p):
    directory = root / "preflight"
    directory.mkdir(parents=True, exist_ok=True)
    results = []
    for scenario in sorted(set(p["fixture_pool"] + p["stage_validation_pool"])):
        env = make_env(condition_for(0, "C-R", p), p, True)
        env.reset(seed=scenario, assignment_seed=scenario)
        values = []
        for roles in itertools.product((0, 1), repeat=2):
            env.roles[env.active_ids] = roles
            net = score_network(env)
            values.append(
                dict(
                    roles=list(roles),
                    utility=net["utility"],
                    throughput_mbps=float(net["ue_rate_mbps"].sum()),
                )
            )
        all_access = next(v["utility"] for v in values if v["roles"] == [0, 0])
        results.append(
            dict(
                scenario=scenario,
                roles=values,
                optimum=max(v["utility"] for v in values),
                all_access=all_access,
                net_gain=max(v["utility"] for v in values) - all_access,
                active_ids=env.active_ids.tolist(),
                positions=env.uav_positions.tolist(),
            )
        )
    assert any(row["net_gain"] > p["gate_tolerance"] for row in results), (
        "Fixture has no net relay benefit"
    )
    atomic_json(
        directory / "fixture_all_scenarios.json",
        dict(
            fixture=p["fixture"],
            cases=results,
            all_cases_retained=True,
            no_positive_case_filtering=True,
            nonpositive_cases=[
                r["scenario"] for r in results if r["net_gain"] <= p["gate_tolerance"]
            ],
        ),
    )
    # Positive physical control, real 90 UE/demand/radio and no controller in training.
    stage2 = condition_for(p["stage_ends"][0], "C-R", p)
    witnesses = []
    for scenario in p["stage_validation_pool"]:
        env = make_env(stage2, p, True)
        env.reset(seed=scenario, assignment_seed=scenario, diagnostic=True)
        ids = env.active_ids
        initial = max_role_utility(env)
        # Three legal steps from held-out diagonal perturbation back to bridge.
        for access_move, relay_move in ((3, 1), (4, 2), (4, 4)):
            actions = [8] * 5
            actions[ids[0]], actions[ids[1]] = access_move * 2, relay_move * 2 + 1
            env.step(actions)
        final = score_network(env)["utility"]
        witnesses.append(
            dict(
                scenario=scenario,
                initial_optimum=initial,
                final_utility=final,
                movement_threshold_attainable=final > initial + p["gate_move_improvement"],
            )
        )
    assert any(w["movement_threshold_attainable"] for w in witnesses), (
        "S2 held-out restoration not feasible"
    )
    atomic_json(directory / "movement_witness.json", witnesses)
    atomic_json(
        directory / "provenance.json",
        dict(
            command=sys.argv,
            source_sha256=p["source_sha256"],
            git_status=git_provenance("status", "--short"),
            git_diff=git_provenance("diff", "--", "src", "scripts", "tests", "configs"),
            random_streams=dict(
                scenario="seed+SCEN",
                battery="seed+BATT",
                assignment="seed+episode+IDS",
                exploration="seed+POLI",
                eval="scenario+EVAL",
                replay="seed+20000+lineage",
            ),
        ),
    )


def git_provenance(*args):
    """Source hashes remain authoritative in extracted Kaggle datasets without Git."""
    try:
        result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    except FileNotFoundError:
        return "unavailable: git executable absent (source inventory retained)"
    return result.stdout if result.returncode == 0 else f"unavailable: {result.stderr.strip()}"


def max_role_utility(env):
    saved = env.roles.copy()
    values = []
    for bits in itertools.product((0, 1), repeat=len(env.active_ids)):
        env.roles[env.active_ids] = bits
        values.append(score_network(env)["utility"])
    env.roles[:] = saved
    return max(values)


def offline_acceptance(root):
    """Run the exported standalone analyzer with no checkpoints or scientific imports."""
    block = """import importlib.abc,sys,runpy
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in {'torch','src'}: raise ImportError('blocked '+fullname)
sys.meta_path.insert(0,Block())
sys.argv=['analyze_bundle.py','--root','.', '--output', '../rebuilt', '--require-complete']
runpy.run_path('analyze_bundle.py',run_name='__main__')
"""
    with tempfile.TemporaryDirectory(prefix="bs5_curriculum_acceptance_") as tmp:
        evidence = Path(tmp) / "evidence"
        evidence.mkdir()
        with zipfile.ZipFile(root / "exports/audit_light.zip") as bundle:
            assert not any(name.endswith(".pt") for name in bundle.namelist())
            bundle.extractall(evidence)
        process = subprocess.run(
            [sys.executable, "-B", "-c", block],
            cwd=evidence,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        if process.returncode:
            raise RuntimeError(process.stdout + process.stderr)
        return dict(
            checkpoint_free=True,
            torch_env_imports_blocked=True,
            cached_tables_rebuilt=True,
            output=process.stdout.strip(),
        )


def prepare(root, config_path):
    assert not root.exists(), "Preserve previous results: choose a new --outdir"
    root.mkdir(parents=True)
    identity = source_identity()
    config = read_json(config_path)
    p = protocol_new(config, identity)
    freeze_source(root, identity)
    atomic_json(root / "protocol.json", p)
    atomic_json(
        root / "schema.json",
        dict(
            version=1,
            inactive_status=3,
            phase="after_movement_before_energy_and_ue_mobility",
            reward="mean(min(rate/demand,1))",
            throughput="sum demand-capped delivered UE rates; Mbps = 360*utility for 90x4Mbps",
            uncapped_throughput="not inferred: radio capacity is not delivered throughput",
            positive_service="rate > 1e-12 Mbps",
            full_qos="rate >= demand - 1e-9 Mbps",
            jain="Jain of per-UE episode-mean rates; not mean of per-slot Jain",
            empty_relay="empty active Relay UAV-slots / all active Relay UAV-slots; None if denominator=0",
            tail="last three common-cadence full-task evaluations",
            events="immutable per-episode atomic packets; uncommitted attempts archived on resume",
            checkpoint="best/last/boundary inference weights; resume.pt resumable at most recent commit only",
        ),
    )
    preflight(root, p)
    smoke = root / "smoke"
    smoke.mkdir()
    freeze_source(smoke, identity)
    sp = protocol_new(config, identity, smoke=True)
    atomic_json(smoke / "protocol.json", sp)
    shutil.copy2(root / "schema.json", smoke / "schema.json")
    shutil.copytree(root / "preflight", smoke / "preflight")
    atomic_json(smoke / "run_registry.json", registry_new(sp))
    began = time.monotonic()
    run_matrix(smoke, export=False)
    acceptance = offline_acceptance(smoke)
    elapsed = time.monotonic() - began
    commits = [read_json(f) for f in (smoke / "runs").glob("*/commit.json")]
    train_rate = sum(c["train_seconds"] for c in commits) / (sp["steps_per_run"] * len(commits))
    eval_rate = sum(c["eval_seconds"] for c in commits) / sum(
        sum(e["slots"] for e in c["evaluations"] + c["stage_evaluations"]) for c in commits
    )
    full_slots = (
        len(p["training_seeds"])
        * len(ARMS)
        * len(p["evaluation_steps"])
        * len(p["validation_pool"])
        * p["horizon"]
    )
    # Same curriculum eval/retention schedule as smoke, scaled by actual pool/horizon.
    stage_evals = sum(len(c["stage_evaluations"]) for c in commits)
    stage_slots = stage_evals * len(p["stage_validation_pool"]) * p["horizon"]
    forecast = (
        (train_rate * p["steps_per_run"] * len(commits) + eval_rate * (full_slots + stage_slots))
        * p["runtime_safety_factor"]
        / 3600
    )
    acceptance.update(
        smoke_elapsed_s=elapsed,
        main_training_steps_executed=0,
        smoke_training_steps=sp["steps_per_run"] * len(commits),
        estimated_hours=forecast,
        full_eval_slots=full_slots,
        stage_eval_slots=stage_slots,
        caveat="bounded smoke extrapolation with full traces/shadows/oracle; not a runtime guarantee; budget never auto-reduced",
    )
    atomic_json(root / "preflight/smoke_acceptance.json", acceptance)
    atomic_json(root / "run_registry.json", registry_new(p))
    export_bundles(root)
    print(
        json.dumps(
            dict(
                event="PREPARED_NOT_MAIN_TRAINED",
                outdir=str(root),
                estimated_hours=forecast,
                total_main_steps=p["steps_per_run"] * len(p["training_seeds"]) * len(ARMS),
            )
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("prepare", "run", "export"):
        mode.add_argument(f"--{name}", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/bs5_curriculum_fixed_v1.json"
    )
    parser.add_argument(
        "--outdir", type=Path, default=ROOT / "results/bs5_curriculum_fixed_20261023"
    )
    args = parser.parse_args()
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    root = args.outdir.resolve()
    if args.prepare:
        prepare(root, args.config)
    else:
        with experiment_lock(root):
            if args.run:
                run_matrix(root, resume=args.resume)
            else:
                export_bundles(root)


if __name__ == "__main__":
    main()
