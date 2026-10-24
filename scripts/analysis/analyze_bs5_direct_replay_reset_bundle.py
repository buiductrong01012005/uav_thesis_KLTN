"""Stdlib-only replay-reset auditor; no torch, environment, weights, or policy rollout."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path
import statistics
import tempfile
import zipfile

try:
    from bs5_bundle import atomic_json, canonical_hash, file_hash, load_trace, read_json, write_csv
    from analyze_bs5_curriculum_bundle import check_rows, episode_metrics, equivalent
except ImportError:
    from scripts.analysis.bs5_bundle import (
        atomic_json,
        canonical_hash,
        file_hash,
        load_trace,
        read_json,
        write_csv,
    )
    from scripts.analysis.analyze_bs5_curriculum_bundle import (
        check_rows,
        episode_metrics,
        equivalent,
    )

ARM = "D-R-reset"


def safe_file(root, name):
    target = (root / name).resolve()
    assert target.is_relative_to(root.resolve()) and target.is_file(), (
        f"missing/unsafe file: {name}"
    )
    return target


def verify_manifest(root):
    sums = read_json(root / "SHA256SUMS.json")
    actual = {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and p != root / "SHA256SUMS.json"
    }
    assert actual == set(sums), "manifest file set mismatch"
    for name, expected in sums.items():
        assert file_hash(safe_file(root, name)) == expected, f"hash mismatch: {name}"


def expected_updates(p, end):
    size, lineage, rows = 0, 0, []
    for before in range(end):
        if before in p["reset_steps"]:
            size, lineage = 0, lineage + 1
        size = min(p["replay_capacity"], size + 1)
        if size >= max(p["warmup"], p["batch_size"]) and (before + 1) % p["update_every"] == 0:
            rows.append((before + 1, lineage, size))
    return rows


def endpoint(name, arm, seed, rows, steps, updates=None):
    curve = []
    for step in sorted({r["step"] for r in rows}):
        episodes = [r for r in rows if r["step"] == step]
        curve.append(dict(step=step, utility=statistics.fmean(r["utility"] for r in episodes)))
    best = max(curve, key=lambda r: r["utility"])
    auc = sum(
        (b["step"] - a["step"]) * (a["utility"] + b["utility"]) / 2
        for a, b in zip(curve, curve[1:])
    )
    last = [r for r in rows if r["step"] == curve[-1]["step"]]
    denominator = sum(r["relay_active_uav_slots"] for r in last)
    return dict(
        run=name,
        arm=arm,
        seed=seed,
        steps=steps,
        updates=updates,
        best_step=best["step"],
        best_utility=best["utility"],
        last_utility=curve[-1]["utility"],
        last_evaluation_step=curve[-1]["step"],
        tail_mean=statistics.fmean(r["utility"] for r in curve[-3:]),
        peak_to_last=best["utility"] - curve[-1]["utility"],
        normalized_auc=auc / curve[-1]["step"] if curve[-1]["step"] else 0.0,
        curve=curve,
        last_relay_empty_slots=sum(r["relay_empty_slots"] for r in last),
        last_relay_active_slots=denominator,
        last_empty_relay_fraction=sum(r["relay_empty_slots"] for r in last) / denominator
        if denominator
        else None,
    )


def selected_metric(episodes, metric):
    if metric == "first_service_slot_restricted":
        # Explicit horizon+1 coding, always accompanied by the censor fraction.
        return statistics.fmean(
            r["first_service_slot"] if r["first_service_slot"] is not None else r["slots"] + 1
            for r in episodes
        )
    if metric == "empty_relay_fraction":
        denominator = sum(r["relay_active_uav_slots"] for r in episodes)
        return sum(r["relay_empty_slots"] for r in episodes) / denominator if denominator else None
    values = [r.get(metric) for r in episodes]
    return (
        statistics.fmean(values)
        if values and all(isinstance(v, (int, float)) for v in values)
        else None
    )


def validate(root, *, require_complete=False, check_manifest=True):
    root = Path(root)
    if check_manifest:
        verify_manifest(root)
    p, registry, identity, baseline = [
        read_json(root / n)
        for n in (
            "protocol.json",
            "run_registry.json",
            "source_identity.json",
            "baseline_refs.json",
        )
    ]
    assert canonical_hash(p) == registry["protocol_sha256"], "protocol drift"
    assert canonical_hash(identity["files"]) == identity["sha256"] == p["source_sha256"]
    assert canonical_hash(baseline) == p["baseline_sha256"], "baseline drift"
    assert p["allow_relay"] is True and p["training_mode"] == "direct" and p["arms"] == [ARM]
    assert p["full_condition"] == dict(
        stage="D",
        substage="D",
        active_count=5,
        mobility=True,
        battery=True,
        starts="B",
        radius=None,
        jitter=0,
        stay_only=False,
    )
    expected_names = {f"{ARM}_seed{s}" for s in p["training_seeds"]}
    assert set(registry["runs"]) == expected_names, "missing/extra run"
    main = p["scope"] != "smoke_not_main"
    if main:
        assert (
            p["training_seeds"] == [551, 552]
            and p["steps_per_run"] == 24000
            and p["horizon"] == 150
        )
        assert p["reset_steps"] == [3000, 4500, 6000, 7500, 9000, 10500]
        assert p["common_evaluation_steps"] == list(range(0, 24001, 3000))
        assert p["validation_pool"] == list(range(130001, 130011))
        assert p["scenario_schedule"] == baseline["protocol"]["scenario_schedule"]
        for n, h in baseline["source_identity"]["files"].items():
            if n.startswith("src/"):
                assert identity["files"][n] == h, "scientific source drift"
    if require_complete:
        assert registry["status"] == "complete", "experiment incomplete"
    results, validation, reset_rows, exogenous = [], [], [], {}
    total_train = total_updates = total_eval = verified_bytes = verified_count = 0
    for name in sorted(expected_names):
        entry = registry["runs"][name]
        if entry["status"] == "not_started":
            assert registry["status"] != "complete"
            continue
        directory = root / "runs" / name
        meta, commit, index = [
            read_json(directory / n) for n in ("meta.json", "commit.json", "packet_index.json")
        ]
        seed = meta["seed"]
        assert meta["arm"] == ARM and name == f"{ARM}_seed{seed}" and meta["allow_relay"] is True
        assert meta["source_sha256"] == p["source_sha256"] and meta[
            "config_sha256"
        ] == canonical_hash(meta["resolved_config"])
        assert (meta["state_dim"], meta["obs_dim"], meta["n_actions"]) == (111, 116, 10)
        assert meta["resolved_config"]["allow_relay"] is True
        if main:
            assert (
                meta["initial_weights_sha256"]
                == baseline["metas"][f"D-R_seed{seed}"]["initial_weights_sha256"]
            )
            assert (
                meta["resolved_config"] == baseline["metas"][f"D-R_seed{seed}"]["resolved_config"]
            )
        assert (
            commit["steps"] == entry["steps"]
            and commit["episodes"] * p["horizon"] == commit["steps"]
        )
        if entry["status"] == "complete":
            assert commit["steps"] == p["steps_per_run"]
        if require_complete or registry["status"] == "complete":
            assert entry["status"] == "complete", "incomplete run"
        assert index == commit["packet_index"], "uncommitted index; resume to reconcile"
        actual = {
            f.relative_to(directory).as_posix()
            for folder in ("episodes", "updates", "events", "train_traces", "evaluations")
            for f in (directory / folder).rglob("*")
            if f.is_file()
        }
        assert set(index) == actual, "packet file set mismatch"
        for relative, info in index.items():
            path = safe_file(directory, relative)
            assert path.stat().st_size == info["bytes"] and file_hash(path) == info["sha256"], (
                f"packet hash mismatch: {relative}"
            )
            verified_bytes += path.stat().st_size
            verified_count += 1
        reset_steps = [
            s
            for s in p["reset_steps"]
            if s < commit["steps"]
            or s == commit["steps"]
            and commit["reset_phase"] == "after_clear"
        ]
        assert commit["applied_reset_steps"] == reset_steps
        assert read_json(directory / "reset_ledger.json") == commit["resets"]
        for k, reset in enumerate(commit["resets"]):
            assert reset["step"] == p["reset_steps"][k]
            if reset["kind"] == "reset_requested":
                assert reset is commit["resets"][-1] and commit["reset_phase"] == "before_clear"
                continue
            previous = p["reset_steps"][k - 1] if k else 0
            assert reset["discarded"] == min(p["replay_capacity"], reset["step"] - previous)
            assert reset["lineage"] == k + 1 and reset["replay_rng_seed"] == seed + 20001 + k
            assert (
                reset["preserved_before"] == reset["preserved_after"]
                and reset["epsilon_global_step"] == reset["step"]
            )
            reset_rows.append(dict(run=name, **reset))
        expected = expected_updates(p, commit["steps"])
        actual_updates, seen = [], set()
        expected_packet_names = set()
        for eid in range(commit["episodes"]):
            ep = read_json(directory / f"episodes/{eid:06d}.json")
            updates = read_json(directory / f"updates/{eid:06d}.json")
            event = read_json(directory / f"events/{eid:06d}.json")
            expected_packet_names.update(
                f"{folder}/{eid:06d}.json" for folder in ("episodes", "updates", "events")
            )
            start = eid * p["horizon"]
            spec = p["scenario_schedule"][str(seed)][eid]
            lineage = sum(s <= start for s in p["reset_steps"])
            assert (
                ep["episode_id"] == eid
                and ep["start_step"] == start
                and ep["end_step"] == start + p["horizon"]
            )
            assert (
                ep["scenario_seed"] == spec["scenario_seed"]
                and ep["assignment_seed"] == spec["assignment_seed"]
            )
            assert ep["residual"]["energy"] == spec["residual_battery"] and ep["lineage"] == lineage
            if main:
                reference = baseline["training_references"][str(seed)][eid]
                assert ep["residual"] == reference["residual"]
            assert ep["condition"] == p["full_condition"] and ep["completion_status"] == "complete"
            assert (
                ep["source_sha256"] == p["source_sha256"]
                and ep["config_sha256"] == meta["config_sha256"]
            )
            assert ep["metrics"]["slots"] == p["horizon"] and event["terminal"] is True
            assert event["starts"] == [[14, 8], [13, 8], [15, 8], [14, 7], [14, 9]]
            reasons = []
            if spec["scenario_seed"] not in seen:
                reasons.append("first_scenario")
                seen.add(spec["scenario_seed"])
            if eid % p["train_trace_every_episodes"] == 0:
                reasons.append("periodic10episodes")
            if start in [0] + p["reset_steps"]:
                reasons.append("first_after_reset_or_initial")
            assert ep["capture_reason"] == reasons
            for u in updates:
                assert u["update"] == len(actual_updates) + 1
                actual_updates.append((u["step"], u["lineage"], u["replay_size"]))
                assert start < u["step"] <= ep["end_step"] and u["lineage"] == lineage
                assert not u["nonfinite"] and all(
                    math.isfinite(v) for v in u.values() if isinstance(v, (int, float))
                )
                assert u["grad_norm_after_clip"] <= 10.0001 and u["lr"] == p["lr"]
                assert equivalent(
                    u["epsilon"],
                    1 + (p["epsilon_end"] - 1) * min((u["step"] - 1) / p["epsilon_decay_steps"], 1),
                )
                assert u["target_sync"] == (u["step"] % p["target_every"] == 0)
                assert u["replay_stage_composition"] == {"D": u["replay_size"]}
            if reasons:
                relative = f"train_traces/{eid:06d}.json.gz"
                expected_packet_names.add(relative)
                packet = load_trace(directory / relative)
                rows = packet["trace"]
                check_rows(rows, p, ARM, spec["scenario_seed"])
                assert all(r["task_condition"] == p["full_condition"] for r in rows)
                assert equivalent(episode_metrics(rows), ep["metrics"]) and equivalent(
                    packet["metrics"], ep["metrics"]
                )
                assert ep["trajectory_sha256"] == canonical_hash(
                    [[r["raw_actions"], r["reward"]] for r in rows]
                )
                if main and "exogenous" in reference:
                    assert [r["exogenous_sha256"] for r in rows] == reference["exogenous"]
        assert actual_updates == expected and len(expected) == commit["updates"], (
            "missing/duplicate/wrong-lineage updates"
        )
        eval_summaries, run_rows = [], []
        for step in [s for s in p["evaluation_steps"] if s <= commit["steps"]]:
            values = []
            expected_packet_names.add(f"evaluations/{step:06d}/summary.json")
            for scenario in p["validation_pool"]:
                relative = f"evaluations/{step:06d}/{scenario}.json.gz"
                expected_packet_names.add(relative)
                packet = load_trace(directory / relative)
                rows = packet["trace"]
                check_rows(rows, p, ARM, scenario, evaluation=True)
                assert all(
                    r["checkpoint_step"] == step
                    and r["seed"] == seed
                    and r["task_condition"] == p["full_condition"]
                    for r in rows
                )
                hashes = [r["exogenous_sha256"] for r in rows]
                assert scenario not in exogenous or hashes == exogenous[scenario], (
                    "unpaired exogenous dynamics"
                )
                exogenous[scenario] = hashes
                if main:
                    assert hashes == baseline["evaluation_exogenous"][str(scenario)]
                    assert (
                        packet["initial_residual"] == baseline["evaluation_residual"][str(scenario)]
                    )
                metrics = episode_metrics(rows)
                assert equivalent(metrics, packet["metrics"]), "tampered cached metrics"
                row = dict(run=name, arm=ARM, seed=seed, step=step, scenario=scenario, **metrics)
                validation.append(row)
                run_rows.append(row)
                values.append(metrics["utility"])
                total_eval += len(rows)
            summary = read_json(directory / f"evaluations/{step:06d}/summary.json")
            assert summary == dict(
                step=step,
                utility=statistics.fmean(values),
                slots=p["horizon"] * len(values),
                eligible_for_best=True,
            ) or equivalent(
                summary,
                dict(
                    step=step,
                    utility=statistics.fmean(values),
                    slots=p["horizon"] * len(values),
                    eligible_for_best=True,
                ),
            )
            eval_summaries.append(summary)
        assert set(index) == expected_packet_names, "missing/extra scheduled packet"
        assert equivalent(commit["evaluations"], eval_summaries)
        result = endpoint(name, ARM, seed, run_rows, commit["steps"], commit["updates"])
        assert commit["best_step"] == result["best_step"] and equivalent(
            commit["best_utility"], result["best_utility"]
        )
        results.append(result)
        total_train += commit["steps"]
        total_updates += commit["updates"]
    return dict(
        status=registry["status"],
        scope=p["scope"],
        training_steps=total_train,
        learner_updates=total_updates,
        validation_slots=total_eval,
        runs=results,
        validation=validation,
        resets=reset_rows,
        verified_packet_bytes=verified_bytes,
        verified_packet_count=verified_count,
        baseline_scope=baseline["verification_scope"],
    )


def baseline_rows(baseline, steps):
    result = []
    for row in csv.DictReader(io.StringIO(baseline["validation_csv"])):
        if int(row["step"]) not in steps:
            continue
        converted = {}
        for k, value in row.items():
            if k in {"seed", "step", "scenario"}:
                converted[k] = int(value)
            elif k in {"arm", "run"}:
                converted[k] = value
            elif value in {"True", "False", "true", "false"}:
                converted[k] = value.lower() == "true"
            else:
                try:
                    converted[k] = float(value) if value not in {"", "None"} else None
                except ValueError:
                    converted[k] = value
        result.append(converted)
    return result


def analyze(root, output, *, require_complete=False, check_manifest=True):
    root, output = Path(root), Path(output)
    if check_manifest:
        assert not output.resolve().is_relative_to(root.resolve()), (
            "output must be outside manifested bundle"
        )
    result = validate(root, require_complete=require_complete, check_manifest=check_manifest)
    p, baseline = read_json(root / "protocol.json"), read_json(root / "baseline_refs.json")
    output.mkdir(parents=True, exist_ok=True)
    rows, resets = result.pop("validation"), result.pop("resets")
    old = [] if p["scope"] == "smoke_not_main" else baseline_rows(baseline, p["evaluation_steps"])
    for arm in ("C-R", "D-R"):
        for seed in p["training_seeds"]:
            new = next((r for r in result["runs"] if r["seed"] == seed and r["arm"] == ARM), None)
            if not new:
                continue
            selected = [
                r
                for r in old
                if r["arm"] == arm and r["seed"] == seed and r["step"] <= new["steps"]
            ]
            if selected:
                historical = next(
                    r for r in baseline["runs"] if r["arm"] == arm and r["seed"] == seed
                )
                result["runs"].append(
                    endpoint(
                        f"{arm}_seed{seed}",
                        arm,
                        seed,
                        selected,
                        new["steps"],
                        historical["updates"] if new["steps"] == p["steps_per_run"] else None,
                    )
                )
    lookup = {(r["arm"], r["seed"], r["step"], r["scenario"]): r for r in rows + old}
    comparisons, pairs = [], []
    metrics = [
        "utility",
        "throughput_mbps",
        "mean_positive_service_ue",
        "mean_qos_ue",
        "jain_mean_rate",
        "relay_required_throughput_mbps",
        "net_role_gain",
        "route_gain",
        "empty_relay_fraction",
        "role_opportunity_cost",
        "role_switches",
        "ue_outage_fraction",
        "first_service_censored",
        "first_service_slot_restricted",
    ]
    for seed in p["training_seeds"]:
        current = {r["arm"]: r for r in result["runs"] if r["seed"] == seed}
        if len(current) != 3:
            continue
        for selection in ("last", "best", "tail"):
            steps = {
                arm: [r["best_step"]]
                if selection == "best"
                else [e["step"] for e in r["curve"]][-3:]
                if selection == "tail"
                else [r["last_evaluation_step"]]
                for arm, r in current.items()
            }
            for arm in ("C-R", "D-R"):
                for metric in metrics:
                    values = {a: [] for a in (ARM, arm)}
                    for scenario in p["validation_pool"]:
                        means = {}
                        for a in values:
                            episodes = [lookup[a, seed, s, scenario] for s in steps[a]]
                            means[a] = selected_metric(episodes, metric)
                            if means[a] is not None:
                                values[a].append(means[a])
                        pairs.append(
                            dict(
                                seed=seed,
                                selection=selection,
                                baseline=arm,
                                scenario=scenario,
                                metric=metric,
                                reset=means[ARM],
                                historical=means[arm],
                                difference=means[ARM] - means[arm]
                                if all(v is not None for v in means.values())
                                else None,
                            )
                        )
                    if all(values.values()):
                        a, b = (
                            selected_metric(
                                [
                                    lookup[x, seed, s, scenario]
                                    for s in steps[x]
                                    for scenario in p["validation_pool"]
                                ],
                                metric,
                            )
                            for x in (ARM, arm)
                        )
                        comparisons.append(
                            dict(
                                seed=seed,
                                selection=selection,
                                baseline=arm,
                                metric=metric,
                                reset=a,
                                historical=b,
                                difference=a - b,
                                ratio_of_means=a / b if abs(b) > 1e-12 else None,
                            )
                        )
        for arm in ("C-R", "D-R"):
            for metric in ("normalized_auc", "peak_to_last"):
                a, b = current[ARM][metric], current[arm][metric]
                comparisons.append(
                    dict(
                        seed=seed,
                        selection="curve",
                        baseline=arm,
                        metric=metric,
                        reset=a,
                        historical=b,
                        difference=a - b,
                        ratio_of_means=a / b if abs(b) > 1e-12 else None,
                    )
                )
    # Across-seed ratios use the ratio of means, never the mean of seed ratios.
    grouped = {}
    for row in comparisons:
        grouped.setdefault((row["selection"], row["baseline"], row["metric"]), []).append(row)
    for (selection, arm, metric), items in grouped.items():
        if len(items) != len(p["training_seeds"]):
            continue
        a, b = (statistics.fmean(r[k] for r in items) for k in ("reset", "historical"))
        comparisons.append(
            dict(
                seed="mean",
                selection=selection,
                baseline=arm,
                metric=metric,
                reset=a,
                historical=b,
                difference=a - b,
                ratio_of_means=a / b if abs(b) > 1e-12 else None,
            )
        )
    write_csv(output / "validation_episodes.csv", rows, list(rows[0]) if rows else ["run"])
    write_csv(output / "baseline_validation_episodes.csv", old, list(old[0]) if old else ["run"])
    write_csv(
        output / "runs.csv",
        result["runs"],
        [
            "run",
            "arm",
            "seed",
            "steps",
            "updates",
            "best_step",
            "best_utility",
            "last_utility",
            "last_evaluation_step",
            "tail_mean",
            "peak_to_last",
            "normalized_auc",
            "last_relay_empty_slots",
            "last_relay_active_slots",
            "last_empty_relay_fraction",
        ],
    )
    write_csv(
        output / "comparisons.csv",
        comparisons,
        [
            "seed",
            "selection",
            "baseline",
            "metric",
            "reset",
            "historical",
            "difference",
            "ratio_of_means",
        ],
    )
    write_csv(
        output / "paired_scenarios.csv",
        pairs,
        [
            "seed",
            "selection",
            "baseline",
            "scenario",
            "metric",
            "reset",
            "historical",
            "difference",
        ],
    )
    atomic_json(output / "reset_summary.json", resets)
    timers = [read_json(f) for f in (root / "timing").glob("*.json")]
    atomic_json(
        output / "timer_summary.json",
        dict(
            attempts=timers,
            measured_active_s=sum(t["measured_wall_s"] for t in timers),
            cutoff="current process timing flush before snapshot; completed export receipt is in review ZIP and exports/latest.json; notebook idle excluded",
        ),
    )
    atomic_json(output / "summary.json", result)
    atomic_json(
        output / "verification.json",
        dict(
            verified=True,
            complete=result["status"] == "complete",
            checkpoint_free=True,
            imports="stdlib only",
            packet_hashes_recomputed=True,
            cached_metrics_recomputed=True,
            baseline_scope=baseline["verification_scope"],
        ),
    )
    colors = {ARM: "#e56b25", "C-R": "#208343", "D-R": "#2868ae"}
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="850" height="350" viewBox="0 0 850 350"><rect width="850" height="350" fill="white"/><text x="25" y="22">Full-task utility; common evaluation cadence</text>'
    ]
    for i, r in enumerate(result["runs"]):
        points = " ".join(
            f"{45 + 760 * e['step'] / p['steps_per_run']:.2f},{310 - 260 * e['utility']:.2f}"
            for e in r["curve"]
        )
        svg.append(
            f'<polyline points="{points}" fill="none" stroke="{colors[r["arm"]]}" stroke-width="2" stroke-dasharray="{4 if r["seed"] == p["training_seeds"][-1] else 0}"/><text x="{45 + (i % 3) * 260}" y="{42 + (i // 3) * 16}" fill="{colors[r["arm"]]}">{r["run"]}</text>'
        )
    svg.append(
        '<path d="M45 80 V310 H805" stroke="black" fill="none"/><text x="45" y="335">0</text><text x="750" y="335">env steps</text></svg>'
    )
    (output / "curves.svg").write_text("\n".join(svg), encoding="utf-8")
    lines = [
        f"# D-R-reset: {result['status']}",
        "",
        f"{result['training_steps']} train steps; {result['learner_updates']} updates; {result['validation_slots']} full-eval slots.",
        "",
        "| Run | Last | Best common | Tail 3 | AUC chuẩn hóa | Peak→last |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in result["runs"]:
        lines.append(
            f"| {r['run']} | {r['last_utility']:.6f} | {r['best_utility']:.6f} | {r['tail_mean']:.6f} | {r['normalized_auc']:.6f} | {r['peak_to_last']:.6f} |"
        )
    lines += [
        "",
        "Primary = last24k. Best chỉ chọn 9 mốc chung; tail = 18k/21k/24k khi complete. Throughput = 360×utility, không phải hai bằng chứng độc lập.",
        "Hai training seeds là pilot; scenario không phải independent training seed. Validation đã xem, không phải final test. Không suy ra S1 đã học vững role.",
        "Replay reset gồm warmup lại và sampling lineage; phép thử này chưa tách mọi nguyên nhân curriculum. Nếu reset gần C-R và hơn D-R: có thể giải thích một phần lợi ích; không chứng minh equivalence/curriculum vô dụng. Nếu C-R hơn reset: bằng chứng cho task schedule trong phạm vi pilot.",
        "Baseline: " + baseline["verification_scope"],
        "Timers: phase outer không chồng nhau; update/shadow là con. Xem export_receipt.json/latest.json cho thời gian export hoàn tất. So runtime cũ cần lưu ý khác thiết bị/runtime; không gán toàn bộ phần dư cũ cho ZIP.",
        "Relay rỗng = Relay UAV-slot active không mang traffic / tổng Relay UAV-slot active. Empty denominator → null. Jain dùng per-UE episode-mean; oracle sampled chỉ diagnostic.",
        "First-service restricted slot: chưa có service được mã hóa horizon+1, luôn đọc cùng first_service_censored (tỷ lệ episode bị censor); không coi đây là thời điểm service thật.",
    ]
    (output / "report_vi.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument(
        "--live",
        action="store_true",
        help="live committed directory without outer archive manifest; packet hashes still required",
    )
    args = parser.parse_args()
    if args.bundle.is_file():
        with tempfile.TemporaryDirectory(prefix="bs5-reset-audit-") as temporary:
            root = Path(temporary)
            with zipfile.ZipFile(args.bundle) as archive:
                assert len(archive.namelist()) == len(set(archive.namelist())), (
                    "duplicate ZIP members"
                )
                assert all(
                    (root / n).resolve().is_relative_to(root.resolve()) for n in archive.namelist()
                ), "unsafe archive"
                archive.extractall(root)
            result = analyze(root, args.output, require_complete=args.require_complete)
    else:
        result = analyze(
            args.bundle,
            args.output,
            require_complete=args.require_complete,
            check_manifest=not args.live,
        )
    print(json.dumps({k: v for k, v in result.items() if k != "runs"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
