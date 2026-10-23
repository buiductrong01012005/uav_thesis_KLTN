"""Checkpoint-free BS5 curriculum analysis. Stdlib only, also shipped inside bundles."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import tempfile
import zipfile

try:
    from bs5_bundle import (
        aggregate_episode,
        atomic_json,
        canonical_hash,
        file_hash,
        load_trace,
        rate_utility,
        read_json,
        validate_rates,
        write_csv,
    )
except ImportError:
    from scripts.analysis.bs5_bundle import (
        aggregate_episode,
        atomic_json,
        canonical_hash,
        file_hash,
        load_trace,
        rate_utility,
        read_json,
        validate_rates,
        write_csv,
    )

ARMS = ("C-R", "D-R", "C-N", "D-N")
TOLERANCE = 1e-10


def equivalent(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(equivalent(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(equivalent(x, y) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return (
            math.isfinite(a)
            and math.isfinite(b)
            and math.isclose(a, b, rel_tol=TOLERANCE, abs_tol=TOLERANCE)
        )
    return a == b


def cached_analysis_matches(cached, generated):
    """Compare data across platforms; Windows text output may use CRLF."""
    if not cached.is_file():
        return False
    if cached.suffix == ".json":
        return equivalent(read_json(cached), read_json(generated))
    return cached.read_text(encoding="utf-8") == generated.read_text(encoding="utf-8")


def episode_metrics(rows):
    metrics = aggregate_episode(rows)
    metrics["mean_positive_service_ue"] = statistics.fmean(
        sum(rate > 1e-12 for rate in row["service"]["network"]["ue_rate_mbps"]) for row in rows
    )
    denominator = metrics["relay_empty_slots"] + metrics["relay_traffic_slots"]
    metrics["relay_active_uav_slots"] = denominator
    metrics["empty_relay_fraction"] = (
        metrics["relay_empty_slots"] / denominator if denominator else None
    )
    metrics["actor_forward_calls"] = sum(r["q_summary"]["agent_forward_calls"] for r in rows)
    return metrics


def condition_at(p, arm, step):
    return next(
        span["condition"]
        for span in p["training_conditions"][arm]
        if span["start"] <= step < span["end"]
    )


def signature(condition):
    return {k: v for k, v in condition.items() if k not in {"stage", "substage"}}


def check_rows(rows, p, arm, scenario, *, evaluation=False):
    assert [r["slot"] for r in rows] == list(range(1, p["horizon"] + 1)), (
        "missing/duplicate trace slot"
    )
    for i, row in enumerate(rows):
        assert row["arm"] == arm and row["scenario_seed"] == scenario
        validate_rates(row)
        assert len(row["service"]["ue_positions"]) == 90
        assert row["service"]["ue_demand_mbps"] == [4.0] * 90
        assert row["surviving_bs_ids"] == [0, 1, 2, 3, 4]
        assert row["charging_station_cells"] == [[1, 1], [18, 1], [10, 18]]
        assert row["exogenous_sha256"] == canonical_hash(
            [row["service"]["ue_positions"], row["service"]["ue_demand_mbps"]]
        )
        assert row["terminated"] is False and row["truncated"] == (i == len(rows) - 1)
        assert all(any(mask) for mask in row["next_action_mask"])
        if arm.endswith("N"):
            assert all(not any(mask[1::2]) for mask in row["action_mask"] + row["next_action_mask"])
        inactive = [a for a, s in enumerate(row["service"]["status"]) if s == 3]
        net = row["service"]["network"]
        assert not set(map(int, net["association"])) & set(inactive), "inactive Access contributed"
        assert not {int(r) for r in net["routes"].values() if r is not None} & set(inactive), (
            "inactive Relay contributed"
        )
        for agent in inactive:
            assert row["post"]["status"][agent] == 3
            assert row["decision"]["positions"][agent] == row["post"]["positions"][agent]
            assert row["decision"]["battery"][agent] == row["post"]["battery"][agent]
        if not row["task_condition"]["battery"]:
            assert row["decision"]["battery"] == row["post"]["battery"], "frozen battery changed"
        if not row["task_condition"]["mobility"]:
            assert row["service"]["ue_positions"] == row["post"]["ue_positions"]
        if evaluation:
            sh = row["shadow"]
            actual = net["utility"]
            direct = sh["same_access_direct"]["utility"]
            all_access = sh["all_access"]["utility"]
            assert math.isclose(
                actual - all_access, (actual - direct) - (all_access - direct), abs_tol=TOLERANCE
            )
            oracle = (row["slot"] - 1) % p["oracle_every_slots"] == 0
            assert ("role_enumeration" in sh) == oracle
            if oracle:
                active = sum(s == 0 for s in row["service"]["status"])
                assert len(sh["role_enumeration"]) == 2**active
                for candidate in sh["role_enumeration"]:
                    rate_utility(candidate, row["service"]["ue_demand_mbps"])
                assert max(v["utility"] for v in sh["role_enumeration"]) + TOLERANCE >= actual
        if i + 1 < len(rows):
            nxt = rows[i + 1]
            assert row["post"]["positions"] == nxt["decision"]["positions"]
            assert row["next_action_mask"] == nxt["action_mask"]


def check_gate(packet, p, arm):
    rows, gate = packet["trace"], packet["gate"]
    candidates = packet["initial_role_assignments"]
    for candidate in candidates:
        rate_utility(candidate, [4.0] * 90)
    optimum = max(v["utility"] for v in candidates)
    direct = next(v["utility"] for v in candidates if not any(v["roles"]))
    gain = optimum - direct
    ratio = (packet["metrics"]["utility"] - direct) / gain if gain > p["gate_tolerance"] else None
    if arm.endswith("N"):
        ratio = packet["metrics"]["utility"] / optimum if optimum > p["gate_tolerance"] else None
    assert equivalent(optimum, gate["initial_feasible_optimum"])
    assert equivalent(gain, gate["attainable_net_gain"])
    assert equivalent(ratio, gate["attainment_ratio"])
    moved = any(r["service"]["uav_positions"] != packet["initial_positions"] for r in rows)
    assert gate["moved"] == moved
    assert gate["movement_pass"] == (
        moved and packet["metrics"]["utility"] > optimum + p["gate_move_improvement"]
    )


def validate(root, *, require_complete=False, check_manifest=True):
    root = Path(root)
    if check_manifest:
        sums = read_json(root / "SHA256SUMS.json")
        actual = {
            f.relative_to(root).as_posix()
            for f in root.rglob("*")
            if f.is_file() and f.name != "SHA256SUMS.json"
        }
        assert actual == set(sums), "manifest file set mismatch"
        for name, expected in sums.items():
            path = (root / name).resolve()
            assert path.is_relative_to(root.resolve()) and file_hash(path) == expected, (
                f"hash mismatch: {name}"
            )
    p, registry = read_json(root / "protocol.json"), read_json(root / "run_registry.json")
    assert registry["protocol_sha256"] == canonical_hash(p), "protocol drift"
    identity = read_json(root / "source_identity.json")
    assert identity["sha256"] == p["source_sha256"] == canonical_hash(identity["files"])
    expected_runs = {f"{arm}_seed{seed}" for seed in p["training_seeds"] for arm in ARMS}
    assert set(registry["runs"]) == expected_runs, "missing/extra run"
    if require_complete:
        assert registry["status"] == "complete", "experiment incomplete"
    results, validation, stage_results, initial, exogenous, paired_train = [], [], [], {}, {}, {}
    total_updates = total_train = total_eval = total_stage_eval = 0
    for name in sorted(expected_runs):
        entry = registry["runs"][name]
        if entry["status"] == "not_started":
            assert registry["status"] != "complete"
            continue
        directory = root / "runs" / name
        commit, meta = read_json(directory / "commit.json"), read_json(directory / "meta.json")
        arm, seed = meta["arm"], meta["seed"]
        assert name == f"{arm}_seed{seed}"
        assert meta["source_sha256"] == p["source_sha256"]
        assert meta["config_sha256"] == canonical_hash(meta["resolved_config"])
        assert (meta["state_dim"], meta["obs_dim"], meta["n_actions"]) == (111, 116, 10)
        assert seed not in initial or initial[seed] == meta["initial_weights_sha256"], (
            "unpaired initial weights"
        )
        initial[seed] = meta["initial_weights_sha256"]
        assert entry["steps"] == commit["steps"]
        if require_complete or registry["status"] == "complete":
            assert entry["status"] == "complete" and commit["steps"] == p["steps_per_run"]
        episodes = sorted((directory / "episodes").glob("*.json"))
        episodes = [f for f in episodes if int(f.stem) < commit["episodes"]]
        assert len(episodes) == commit["episodes"], "missing episode"
        seen, budget, subbudget, update_ids, last_task = set(), {}, {}, set(), None
        replay_count, lineage = 0, -1
        for eid, path in enumerate(episodes):
            ep = read_json(path)
            step = eid * p["horizon"]
            condition = condition_at(p, arm, step)
            assert ep["episode_id"] == eid and ep["start_step"] == step
            assert ep["end_step"] == step + p["horizon"] and ep["condition"] == condition
            assert ep["stage"] == condition["stage"] and ep["substage"] == condition["substage"]
            assert ep["completion_status"] == "complete"
            assert (
                ep["source_sha256"] == p["source_sha256"]
                and ep["config_sha256"] == meta["config_sha256"]
            )
            spec = p["scenario_schedule"][str(seed)][eid]
            scenario = (
                spec["fixture_seed"] if ep["stage"] in {"S1", "S2"} else spec["scenario_seed"]
            )
            assert (
                ep["scenario_seed"] == scenario
                and ep["residual"]["energy"] == spec["residual_battery"]
            )
            events = read_json(directory / "events" / path.name)
            assert all(
                e["episode"] == eid and step <= e["global_env_step"] <= ep["end_step"]
                for e in events
            )
            task = signature(condition)
            if task != last_task:
                lineage += 1
                changes = [e for e in events if e["kind"] == "task_change"]
                assert len(changes) == 1 and changes[0]["replay_discarded"] == replay_count
                assert (
                    changes[0]["keep_weights"]
                    and changes[0]["keep_optimizer"]
                    and changes[0]["keep_targets"]
                )
                replay_count, last_task = 0, task
            else:
                assert not any(e["kind"] == "task_change" for e in events)
            assert ep["lineage"] == lineage
            updates = read_json(directory / "updates" / path.name)
            expected_update_steps = []
            for slot in range(1, p["horizon"] + 1):
                replay_count = min(p["replay_capacity"], replay_count + 1)
                s = step + slot
                if replay_count >= max(p["warmup"], p["batch_size"]) and s % p["update_every"] == 0:
                    expected_update_steps.append(s)
            assert [u["step"] for u in updates] == expected_update_steps, (
                "missing/duplicate learner update"
            )
            for update in updates:
                assert update["step"] not in update_ids and update["lineage"] == lineage
                update_ids.add(update["step"])
                expected_epsilon = 1 + (p["epsilon_end"] - 1) * min(
                    (update["step"] - 1) / p["epsilon_decay_steps"], 1
                )
                assert equivalent(update["epsilon"], expected_epsilon), "epsilon schedule drift"
                assert update["update"] == len(update_ids), "update counter drift"
                assert update["target_sync"] == (update["step"] % p["target_every"] == 0)
                assert not update["nonfinite"] and update["grad_norm_after_clip"] <= 10.0001
                assert all(
                    math.isfinite(update[k])
                    for k in ("loss", "td_abs_mean", "q_total_mean", "target_q_mean", "td_abs_p90")
                )
                assert sum(update["replay_stage_composition"].values()) == update["replay_size"]
            key = f"{ep['substage']}:{scenario}"
            reasons = []
            if key not in seen:
                reasons.append("first_scenario_in_substage")
                seen.add(key)
            if eid % p["train_trace_every_episodes"] == 0:
                reasons.append("periodic")
            if step == 0 or step in p["condition_boundaries"]:
                reasons.append("first_after_boundary")
            assert reasons == ep["capture_reason"]
            trace = directory / "train_traces" / f"{eid:06d}.json.gz"
            assert trace.exists() == bool(reasons), "training trace coverage mismatch"
            if reasons:
                packet = load_trace(trace)
                check_rows(packet["trace"], p, arm, scenario)
                assert equivalent(episode_metrics(packet["trace"]), ep["metrics"]), (
                    "train metrics mismatch"
                )
                for row in packet["trace"]:
                    k = seed, eid, canonical_hash(signature(condition)), row["slot"]
                    digest = row["exogenous_sha256"]
                    assert k not in paired_train or paired_train[k] == digest, (
                        "unpaired training exogenous trace"
                    )
                    paired_train[k] = digest
            budget[ep["stage"]] = budget.get(ep["stage"], 0) + p["horizon"]
            subbudget[ep["substage"]] = subbudget.get(ep["substage"], 0) + p["horizon"]
        assert budget == commit["stage_budgets"] and subbudget == commit["substage_budgets"]
        assert sum(budget.values()) == commit["steps"] and len(update_ids) == commit["updates"]
        evaluations = []
        steps = [s for s in p["evaluation_steps"] if s <= commit["steps"]]
        for step in steps:
            location = directory / "evaluations" / f"{step:06d}"
            summary = read_json(location / "summary.json")
            values = []
            for scenario in p["validation_pool"]:
                packet = load_trace(location / f"{scenario}.json.gz")
                assert signature(packet["condition"]) == signature(condition_at(p, "D-R", 0)), (
                    "evaluation task changed"
                )
                rows = packet["trace"]
                check_rows(rows, p, arm, scenario, evaluation=True)
                metrics = episode_metrics(rows)
                assert equivalent(metrics, packet["metrics"]), "trace-summary mismatch"
                assert all(row["checkpoint_step"] == step for row in rows)
                for row in rows:
                    k = scenario, row["slot"]
                    assert k not in exogenous or exogenous[k] == row["exogenous_sha256"], (
                        "unpaired full-task evaluation"
                    )
                    exogenous[k] = row["exogenous_sha256"]
                values.append(metrics["utility"])
                validation.append(
                    dict(run=name, arm=arm, seed=seed, step=step, scenario=scenario, **metrics)
                )
                total_eval += len(rows)
            assert equivalent(statistics.fmean(values), summary["utility"])
            assert summary["eligible_for_best"] == (step in p["common_evaluation_steps"])
            evaluations.append(summary)
        assert equivalent(evaluations, commit["evaluations"])
        stage_specs = []
        if arm.startswith("C"):
            stage_specs.append((0, "current"))
            for step in steps[1:]:
                c = condition_at(p, arm, step - 1)
                if c["stage"] != "S4" or step == p["steps_per_run"]:
                    stage_specs.append((step, "current"))
                if step in {p["stage_ends"][1], p["stage_ends"][2], p["steps_per_run"]}:
                    stage_specs.append((step, "S1_retention"))
        stage_summaries = []
        for step, label in stage_specs:
            location = directory / "stage_evaluations" / f"{step:06d}" / label
            summary = read_json(location / "summary.json")
            values, gates = [], []
            for scenario in p["stage_validation_pool"]:
                packet = load_trace(location / f"{scenario}.json.gz")
                expected_condition = condition_at(
                    p, arm, 0 if label == "S1_retention" else max(0, step - 1)
                )
                assert packet["condition"] == expected_condition, "stage validation task drift"
                check_rows(packet["trace"], p, arm, scenario, evaluation=True)
                metrics = episode_metrics(packet["trace"])
                assert equivalent(metrics, packet["metrics"]), "stage trace-summary mismatch"
                check_gate(packet, p, arm)
                values.append(metrics["utility"])
                gates.append(packet["gate"])
                total_stage_eval += len(packet["trace"])
                stage_results.append(
                    dict(
                        run=name,
                        step=step,
                        label=label,
                        scenario=scenario,
                        **metrics,
                        gate=packet["gate"],
                    )
                )
            assert equivalent(statistics.fmean(values), summary["utility"])
            assert equivalent(gates, summary["gate_rows"])
            eligible = [g for g in gates if g["attainment_ratio"] is not None]
            role_fraction = (
                sum(g["ratio_pass"] for g in eligible) / len(eligible) if eligible else None
            )
            assert equivalent(role_fraction, summary["role_pass_fraction"])
            assert summary["role_gate_pass"] == (
                role_fraction is not None and role_fraction >= p["gate_episode_fraction"]
            )
            assert equivalent(
                sum(g["movement_pass"] for g in gates) / len(gates),
                summary["movement_pass_fraction"],
            )
            assert not summary["gate_controls_transition"]
            stage_summaries.append(summary)
        assert equivalent(stage_summaries, commit["stage_evaluations"])
        common = [e for e in evaluations if e["eligible_for_best"]]
        best = max(common, key=lambda v: (v["utility"], -v["step"]))
        assert best["step"] == commit["best_step"] and equivalent(
            best["utility"], commit["best_utility"]
        )
        auc = sum(
            (a["utility"] + b["utility"]) * 0.5 * (b["step"] - a["step"])
            for a, b in zip(common, common[1:])
        )
        results.append(
            dict(
                run=name,
                arm=arm,
                seed=seed,
                steps=commit["steps"],
                updates=commit["updates"],
                best_step=best["step"],
                best_utility=best["utility"],
                last_utility=common[-1]["utility"],
                tail_mean=statistics.fmean(e["utility"] for e in common[-3:]),
                peak_to_last=best["utility"] - common[-1]["utility"],
                normalized_auc=auc / commit["steps"] if commit["steps"] else 0,
                stage_budgets=budget,
                substage_budgets=subbudget,
                gate_history=commit["gate_history"],
                train_seconds=commit["train_seconds"],
                eval_seconds=commit["eval_seconds"],
                shadow_seconds=commit["shadow_seconds"],
                update_seconds=commit["update_seconds"],
                evaluations=evaluations,
            )
        )
        total_train += commit["steps"]
        total_updates += commit["updates"]
    return dict(
        status=registry["status"],
        training_steps=total_train,
        learner_updates=total_updates,
        validation_slots=total_eval,
        stage_validation_slots=total_stage_eval,
        validated_runs=len(results),
        runs=results,
        validation=validation,
        stage_validation=stage_results,
    )


def analyze(root, output, *, require_complete=False, check_manifest=True):
    root, output = Path(root), Path(output)
    if check_manifest:
        assert not output.resolve().is_relative_to(root.resolve()), (
            "Analysis output must be outside the evidence root"
        )
    result = validate(root, require_complete=require_complete, check_manifest=check_manifest)
    p = read_json(root / "protocol.json")
    output.mkdir(parents=True, exist_ok=True)
    rows, stage_rows = result.pop("validation"), result.pop("stage_validation")
    write_csv(output / "validation_episodes.csv", rows, list(rows[0]) if rows else ["run"])
    write_csv(
        output / "stage_validation_episodes.csv",
        stage_rows,
        list(stage_rows[0]) if stage_rows else ["run"],
    )
    run_lookup = {(r["arm"], r["seed"]): r for r in result["runs"]}
    lookup = {(r["arm"], r["seed"], r["step"], r["scenario"]): r for r in rows}
    contrasts = {
        "curriculum_relay": {"C-R": 1, "D-R": -1},
        "curriculum_vs_direct_no_relay": {"C-R": 1, "D-N": -1},
        "relay_with_curriculum": {"C-R": 1, "C-N": -1},
        "relay_direct": {"D-R": 1, "D-N": -1},
        "interaction": {"C-R": 1, "C-N": -1, "D-R": -1, "D-N": 1},
    }
    effects, paired = [], []
    for seed in p["training_seeds"]:
        if any((arm, seed) not in run_lookup for arm in ARMS):
            continue
        common_end = min(run_lookup[arm, seed]["steps"] for arm in ARMS)
        common_steps = [s for s in p["common_evaluation_steps"] if s <= common_end]
        for selection in ("best", "last_common", "tail_three"):
            selected = {
                arm: (
                    [run_lookup[arm, seed]["best_step"]]
                    if selection == "best"
                    else [common_steps[-1]]
                    if selection == "last_common"
                    else common_steps[-3:]
                )
                for arm in ARMS
            }
            for name, weights in contrasts.items():
                for metric in (
                    "utility",
                    "throughput_mbps",
                    "mean_positive_service_ue",
                    "mean_qos_ue",
                    "relay_required_throughput_mbps",
                    "net_role_gain",
                    "ue_outage_fraction",
                ):
                    diffs, means = [], {arm: [] for arm in weights}
                    for scenario in p["validation_pool"]:
                        values = {
                            arm: statistics.fmean(
                                lookup[arm, seed, step, scenario][metric] for step in selected[arm]
                            )
                            for arm in weights
                        }
                        for arm, v in values.items():
                            means[arm].append(v)
                        delta = sum(weights[arm] * v for arm, v in values.items())
                        diffs.append(delta)
                        paired.append(
                            dict(
                                seed=seed,
                                selection=selection,
                                contrast=name,
                                metric=metric,
                                scenario=scenario,
                                difference=delta,
                            )
                        )
                    effect = dict(
                        seed=seed,
                        selection=selection,
                        contrast=name,
                        metric=metric,
                        difference=statistics.fmean(diffs),
                        scenario_sd=statistics.stdev(diffs) if len(diffs) > 1 else 0,
                    )
                    if len(weights) == 2:
                        positive = next(a for a, w in weights.items() if w == 1)
                        negative = next(a for a, w in weights.items() if w == -1)
                        denominator = statistics.fmean(means[negative])
                        effect.update(
                            ratio_numerator=statistics.fmean(means[positive]),
                            ratio_denominator=denominator,
                            ratio_of_means=statistics.fmean(means[positive]) / denominator
                            if denominator > 1e-6
                            else None,
                        )
                    effects.append(effect)
    fields = [
        "seed",
        "selection",
        "contrast",
        "metric",
        "difference",
        "scenario_sd",
        "ratio_numerator",
        "ratio_denominator",
        "ratio_of_means",
    ]
    write_csv(output / "effects.csv", effects, fields)
    write_csv(
        output / "paired_scenarios.csv",
        paired,
        ["seed", "selection", "contrast", "metric", "scenario", "difference"],
    )
    write_csv(
        output / "runs.csv",
        result["runs"],
        [
            "run",
            "steps",
            "updates",
            "best_step",
            "best_utility",
            "last_utility",
            "tail_mean",
            "peak_to_last",
            "normalized_auc",
            "stage_budgets",
            "substage_budgets",
            "train_seconds",
            "eval_seconds",
            "shadow_seconds",
        ],
    )
    result.update(
        effects=effects,
        caution="Two-seed validation pilot. Curriculum+replay resets are one method; no causal isolation of either component. Best selection excludes boundary-only evaluations. No final test.",
    )
    atomic_json(output / "summary.json", result)
    colors = {"C-R": "#2563eb", "D-R": "#dc2626", "C-N": "#059669", "D-N": "#d97706"}
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="430" viewBox="0 0 900 430"><rect width="900" height="430" fill="white"/><path d="M65 30V350H855" fill="none" stroke="black"/>'
    ]
    for y in (0, 0.1, 0.2, 0.3, 0.4):
        sy = 350 - y * 750
        svg.append(f'<text x="20" y="{sy + 4}">{y:g}</text><path d="M65 {sy}H855" stroke="#eee"/>')
    for i, step in enumerate(p["stage_ends"][:-1]):
        x = 65 + step / p["steps_per_run"] * 790
        svg.append(
            f'<path d="M{x}30V350" stroke="#aaa" stroke-dasharray="3 3"/><text x="{x + 3}" y="25">S{i + 2}</text>'
        )
    for run in result["runs"]:
        points = " ".join(
            f"{65 + e['step'] / p['steps_per_run'] * 790},{350 - e['utility'] * 750}"
            for e in run["evaluations"]
            if e["eligible_for_best"]
        )
        dash = ' stroke-dasharray="5 4"' if run["seed"] == p["training_seeds"][-1] else ""
        svg.append(
            f'<polyline points="{points}" fill="none" stroke="{colors[run["arm"]]}" stroke-width="2"{dash}/>'
        )
    for i, arm in enumerate(ARMS):
        svg.append(f'<text x="{80 + i * 175}" y="390" fill="{colors[arm]}">{arm}</text>')
    svg.append(
        '<text x="65" y="420">Full-task validation utility vs TOTAL training steps; vertical lines: fixed stages.</text></svg>'
    )
    (output / "learning_curves.svg").write_text("".join(svg), encoding="utf-8")
    lines = [
        "# BS5 curriculum pilot — báo cáo",
        "",
        f"Trạng thái: {result['status']}. Train: {result['training_steps']} bước; updates: {result['learner_updates']}; full eval: {result['validation_slots']} slot; stage eval: {result['stage_validation_slots']} slot.",
        "",
        "| Run | Best step | Best utility | Last | Tail 3 | Peak−last |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for run in result["runs"]:
        lines.append(
            f"| {run['run']} | {run['best_step']} | {run['best_utility']:.6f} | {run['last_utility']:.6f} | {run['tail_mean']:.6f} | {run['peak_to_last']:.6f} |"
        )
    lines += [
        "",
        "Đối chiếu từng seed và paired scenario trong effects.csv; gate history, stage budgets và thời gian trong summary.json.",
        "Hai training seeds chỉ là pilot. Curriculum và replay reset là một phương pháp gộp; chưa tách nguyên nhân riêng.",
        "Throughput đã cap demand = 360 × utility; không phải hai bằng chứng độc lập. Không suy ra uncapped throughput từ rate đã cap.",
        "UE rate>1e-12 và UE đạt demand−1e-9 là hai KPI riêng. Jain tính trên rate trung bình mỗi UE trong episode.",
        "Relay rỗng / tổng active Relay UAV-slots; không có Relay thì tỷ lệ là null. First-service không xảy ra được giữ censored.",
        "Oracle chỉ lấy mẫu slot1/76; train traces lấy theo lịch, không suy thành full train coverage. Validation đã được xem; không phải final test.",
    ]
    (output / "report_vi.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if check_manifest:
        for name in (
            "validation_episodes.csv",
            "stage_validation_episodes.csv",
            "effects.csv",
            "paired_scenarios.csv",
            "runs.csv",
            "summary.json",
            "learning_curves.svg",
            "report_vi.md",
        ):
            generated, cached = output / name, root / "analysis" / name
            assert cached_analysis_matches(cached, generated), f"cached analysis mismatch: {name}"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    if args.bundle:
        with tempfile.TemporaryDirectory(prefix="bs5_curriculum_") as tmp:
            with zipfile.ZipFile(args.bundle) as z:
                for name in z.namelist():
                    assert (Path(tmp) / name).resolve().is_relative_to(Path(tmp).resolve()), (
                        "unsafe ZIP path"
                    )
                z.extractall(tmp)
            result = analyze(tmp, args.output, require_complete=args.require_complete)
    else:
        parser.error("--root or --bundle required") if args.root is None else None
        result = analyze(args.root, args.output, require_complete=args.require_complete)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "status",
                    "training_steps",
                    "learner_updates",
                    "validation_slots",
                    "stage_validation_slots",
                    "validated_runs",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
