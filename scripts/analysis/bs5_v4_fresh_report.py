"""Offline reports/export for fresh training and separate benchmark; no rollouts."""
from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
from scripts.analysis.bs5_bundle import atomic_json, canonical_hash, file_hash, read_json
from scripts.analysis.bs5_return_v4_results import audit_game, life_rows, metric
from scripts.analysis.bs5_continuous_common import lifecycle


def table(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row if not isinstance(row[k], (list, dict)))) or ["empty"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def figure_module():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def checked_game(directory, deep=False):
    index = read_json(directory / "index.json")
    assert file_hash(directory / index["data"]) == index["data_sha256"]
    if index["condition"]["battery"]:
        a = audit_game(directory, index, deep=deep)
    else:
        # audit_game's energy branch assumes full-task drains. Frozen curriculum
        # uses zero drains; audit with that resolved slot configuration.
        adjusted = dict(index, resolved_config=dict(index["resolved_config"]))
        for k in ["hover_drain", "move_extra_drain", "access_radio_drain", "relay_radio_drain",
                  "returning_step_drain", "return_safety_margin", "charge_rate_per_slot"]:
            adjusted["resolved_config"][k] = 0.
        a = audit_game(directory, adjusted, deep=False)
    assert np.allclose(a["rates"].sum(1) / 360., a["rewards"], rtol=0, atol=1e-10)
    assert not a["terminated"].any() and a["truncated"].sum() == 1 and a["truncated"][-1]
    if index["condition"]["stage"] == "S4":
        assert not a["learning_done"].any()
    else:
        assert a["learning_done"].sum() == 1 and a["learning_done"][-1]
    recomputed = metric(a)
    for k, v in index["metrics"].items():
        if isinstance(v, float):
            assert abs(recomputed[k] - v) < 1e-9, k
        else:
            assert recomputed[k] == v, k
    return a, index


def cycle_rows(a, index, run):
    rows, _ = lifecycle(list(life_rows(a)), index["initial"], run=run, scenario=index["scenario"])
    return rows


def fresh_export(root):
    parent = root.parent
    n, out = 1, parent / (root.name + "_export")
    while out.exists():
        n += 1
        out = parent / (root.name + f"_export_{n}")
    out.mkdir()
    (out / "analysis").mkdir()
    return out


def pack(archive, pairs):
    temp = archive.with_name(archive.name + ".tmp")
    with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED, compresslevel=3, allowZip64=True) as z:
        for path, name in pairs:
            z.write(path, name)
    temp.replace(archive)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None


def full_archive(root, archive):
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.name != "run.lock")
    manifest = {p.relative_to(root).as_posix(): file_hash(p) for p in files if p.name != "EXPORT_SHA256SUMS.json"}
    atomic_json(root / "EXPORT_SHA256SUMS.json", manifest)
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.name != "run.lock")
    pack(archive, [(p, root.name + "/" + p.relative_to(root).as_posix()) for p in files])


def finish_export(root, out, report, archive_name):
    # Copy the latest compact report into the resumable root as well.
    shutil.copytree(out / "analysis", root / "analysis", dirs_exist_ok=True)
    full_archive(root, out / archive_name)
    pack(out / "review_small.zip", [(p, "analysis/" + p.relative_to(out / "analysis").as_posix())
                                     for p in sorted((out / "analysis").rglob("*")) if p.is_file()])
    files = [p for p in out.iterdir() if p.is_file()]
    atomic_json(out / "EXPORT_INDEX.json", dict(status=read_json(root / "status.json"),
                files=[dict(path=p.name, sha256=file_hash(p), bytes=p.stat().st_size) for p in files],
                resume_archive=archive_name, report=report))
    return out


def _export_training(root, deep=False):
    from scripts.training.run_bs5_v4_fresh_train import read_progress, read_selection, progress_path, training_archive_names, all_complete
    root = Path(root).resolve()
    p = read_json(root / "protocol.json")
    out = fresh_export(root)
    analysis = out / "analysis"
    bins, episodes, updates, validations, runs, cycle_data = [], [], [], [], [], []
    trace_receipts = {}
    names = training_archive_names(p)
    def record_tape(index):
        spec = {k: index["spec"][k] for k in ["condition", "horizon", "scenario_seed", "assignment_seed", "residual_battery", "diagnostic"]}
        entry = dict(spec=spec, position_sha256=index["trace_sha256"], initial=index["initial"])
        key = index["trace_key"]
        assert key == canonical_hash(entry["spec"])
        if key in trace_receipts:
            assert trace_receipts[key] == entry
        trace_receipts[key] = entry
    audited = 0
    for seed in p["training_seeds"]:
        for arm in p["arms"]:
            label = f"{arm}_seed{seed}"
            run = root / "runs" / label
            if not (run / "progress.json").exists():
                continue
            progress = read_progress(run)
            tags = dict(arm=arm, training_seed=seed)
            for game in progress["games"]:
                directory = (run / game["index"]).parent
                a, index = checked_game(directory, deep)
                record_tape(index)
                audited += len(a["rates"])
                assert file_hash(directory / "updates.json.gz") == game["updates_sha256"]
                own_updates = json.load(gzip.open(directory / "updates.json.gz", "rt"))
                updates += [dict(**tags, **r) for r in own_updates]
                episodes.append({**tags, **game, "episode": index["spec"]["episode"], **index["metrics"]})
                for first in range(0, len(a["rates"]), p["curve_bin_slots"]):
                    last = min(first + p["curve_bin_slots"], len(a["rates"]))
                    bins.append(dict(**tags, stage=game["stage"], training_step=game["start"] + last,
                                     slots=last-first, utility=float(a["rewards"][first:last].mean()),
                                     epsilon=float(a["epsilon"][first:last].mean()),
                                     throughput_mbps=float(a["rates"][first:last].sum(1).mean())))
            for row in progress["validations"]:
                directory = (run / row["index"]).parent
                a, index = checked_game(directory, deep)
                record_tape(index)
                audited += len(a["rates"])
                validations.append(dict(**tags, step=row["step"], scenario=row["scenario"], **index["metrics"]))
                cycle_data += [dict(**tags, checkpoint_step=row["step"], **c)
                               for c in cycle_rows(a, index, label)]
            ledger = read_json(run / "budget_ledger.json")
            for phase, count in [("train", progress["steps"]), ("validation", len(progress["validations"]) * p["full_horizon"])]:
                assert ledger[phase]["charged"] == ledger[phase]["observed"] == count, "Uncommitted compute; export must retain partial-error evidence."
            runs.append(dict(**tags, steps=progress["steps"], updates=progress["updates"],
                             best_step=progress["best_step"], best_utility=progress["best_utility"],
                             validation_games=len(progress["validations"]), initial_weights_sha256=progress["initial_weights_sha256"]))
    for seed in p["training_seeds"]:
        own = [r for r in runs if r["training_seed"] == seed]
        if len(own) == 2:
            assert own[0]["initial_weights_sha256"] == own[1]["initial_weights_sha256"]
    for name, rows in [("training_bins100", bins), ("training_episodes", episodes), ("updates", updates),
                       ("validation_games", validations), ("runs", runs), ("validation_cycles", cycle_data)]:
        table(analysis / (name + ".csv"), rows)
    atomic_json(analysis / "validation_cycles.json", cycle_data)
    curve = []
    for run in runs:
        for step in p["evaluation_steps"]:
            own = [r for r in validations if r["arm"] == run["arm"] and r["training_seed"] == run["training_seed"] and r["step"] == step]
            if len(own) == len(p["validation_pool"]):
                curve.append(dict(arm=run["arm"], training_seed=run["training_seed"], step=step,
                                  utility=float(np.mean([r["utility"] for r in own])), games=len(own)))
    table(analysis / "validation_curve.csv", curve)
    if bins or curve:
        plt = figure_module()
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
        for i, run in enumerate(runs):
            own = [r for r in bins if r["arm"] == run["arm"] and r["training_seed"] == run["training_seed"]]
            color = "#1766a3" if run["arm"] == "C-R" else "#d77722"
            style = "-" if run["training_seed"] == p["training_seeds"][0] else "--"
            x, y = np.array([r["training_step"] for r in own]), np.array([r["utility"] for r in own])
            axes[0].plot(x, y, color=color, alpha=.18, linewidth=.7)
            if len(y):
                # Never smooth across curriculum changes.
                for stage in dict.fromkeys(r["stage"] for r in own):
                    ids = [j for j, r in enumerate(own) if r["stage"] == stage]
                    smooth = np.array([y[ids[max(0, j-4):j+1]].mean() for j in range(len(ids))])
                    axes[0].plot(x[ids], smooth, color=color, linestyle=style,
                                 label=f"{run['arm']} / {run['training_seed']}" if stage == own[0]["stage"] else None)
            val = [r for r in curve if r["arm"] == run["arm"] and r["training_seed"] == run["training_seed"]]
            axes[1].plot([r["step"] for r in val], [r["utility"] for r in val], marker="o", markersize=4,
                         color=color, linestyle=style, label=f"{run['arm']} / {run['training_seed']}")
        for ax in axes:
            ax.set_ylabel("Utility")
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
            for end in p["stage_ends"][:-1]:
                ax.axvline(end, color="gray", alpha=.35, linewidth=.8)
        axes[0].set_title("Training: raw bins + trailing 5-bin mean, reset at each substage")
        axes[1].set_title(f"Validation: measured checkpoints, {len(p['validation_pool'])} fixed traces; epsilon=0")
        axes[1].set_xlabel("Executed training environment steps")
        for ext in ["png", "pdf"]:
            fig.savefig(analysis / f"learning_curves.{ext}", dpi=180)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
        for run in runs:
            own = [r for r in updates if r["arm"] == run["arm"] and r["training_seed"] == run["training_seed"]]
            if own:
                ax.plot([r["step"] for r in own], [r["loss"] for r in own], alpha=.5, linewidth=.7,
                        label=f"{run['arm']} / {run['training_seed']}")
        ax.set(xlabel="Training steps", ylabel="Huber loss", title="Learner diagnostics (all recorded updates)")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
        fig.savefig(analysis / "learner_loss.png", dpi=180)
        plt.close(fig)
    status = read_json(root / "status.json")
    atomic_json(analysis / "summary.json", dict(status=status, audited_slots=audited, runs=runs,
                deep_planner_recomputed=deep, scope=p["scope"], protocol_sha256=canonical_hash(p)))
    lines = ["# Fresh C-R/C-N Return-v4 training", "", f"Status: **{status['status']}**. Diagnostic: {p['diagnostic']}. Audited slots: {audited}.", "",
             f"Training only; final 10k benchmark has NOT run. Assigned arms: {p['arms']}; training seeds: {p['training_seeds']}. Every assigned run starts fresh. Same seed shares initial weights, curriculum game parameters and UE tapes across arms.", "",
             "Training plots show measured 100-slot bins (diagnostic uses shorter bins), plus a trailing five-bin mean reset at each curriculum substage. Validation markers are actual measured checkpoints, not added observations.", "",
             "| Arm | Seed | Train steps | Updates | Best step | Best validation utility |", "|---|---:|---:|---:|---:|---:|"]
    for row in runs:
        value = "pending" if row["best_utility"] is None else f"{row['best_utility']:.6f}"
        lines.append(f"| {row['arm']} | {row['training_seed']} | {row['steps']} | {row['updates']} | {row['best_step']} | {value} |")
    lines += ["", "S1/S2 fixtures favor Relay coordination; C-N is a matched-recipe ablation, not a separately optimized no-Relay curriculum. Two training seeds do not establish convergence or optimality.", "",
              f"Keep {names['training_resume']} for exact resume. {names['training_for_benchmark']} is produced when every run assigned to THIS notebook finishes. Combine all four arm/seed jobs before final benchmark; handoff cannot resume training.", "",
              "Validation is reused for checkpoint selection. Test scenarios 260001–260010 remain separate. Optional 260011–260020 extension must cover every method and be chosen for compute budget, not rankings."]
    (analysis / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    if status["status"] == "complete":
        assert all_complete(root, p)
        atomic_json(analysis / "MATCHED_TRACE_RECEIPT.json", trace_receipts)
        pairs = [(root / name, "bs5_v4_training/" + name) for name in ["protocol.json", "benchmark_plan.json", "status.json", "source_identity.json"]]
        for name in ["MATCHED_TRACE_RECEIPT.json", "training_bins100.csv", "training_episodes.csv", "validation_curve.csv", "runs.csv", "learning_curves.png", "learning_curves.pdf"]:
            if (analysis / name).exists():
                pairs.append((analysis / name, "bs5_v4_training/analysis/" + name))
        for run in runs:
            label = f"{run['arm']}_seed{run['training_seed']}"
            relative = Path("runs") / label
            pairs.append((progress_path(root / relative), (Path("bs5_v4_training") / relative / "progress.json").as_posix()))
            selection_paths = sorted((root / relative / "selections").glob("*.json"))
            pairs.append((selection_paths[-1], (Path("bs5_v4_training") / relative / "checkpoint_selection.json").as_posix()))
            selection = read_selection(root / relative)
            for step in sorted({selection["best_step"], p["total_steps"]}):
                path = relative / "checkpoints" / f"{step:06d}.pt"
                pairs.append((root / path, (Path("bs5_v4_training") / path).as_posix()))
        handoff_manifest = analysis / "HANDOFF_SHA256SUMS.json"
        atomic_json(handoff_manifest, {name.removeprefix("bs5_v4_training/"): file_hash(path) for path, name in pairs})
        pairs.append((handoff_manifest, "bs5_v4_training/HANDOFF_SHA256SUMS.json"))
        pack(out / names["training_for_benchmark"], pairs)
    return finish_export(root, out, "analysis/REPORT.md", names["training_resume"])


def ci(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None, None
    rng = np.random.default_rng(771903)
    means = values[rng.integers(len(values), size=(3000, len(values)))].mean(1)
    return float(np.quantile(means, .025)), float(np.quantile(means, .975))


def _export_benchmark(root, deep=False):
    from scripts.evaluation.run_bs5_v4_benchmark import read_commit
    root = Path(root).resolve()
    p, commit = read_json(root / "protocol.json"), read_commit(root)
    out = fresh_export(root)
    analysis = out / "analysis"
    if (root / "training_analysis").exists():
        shutil.copytree(root / "training_analysis", analysis / "training_learning_curves")
    games, windows, cycles = [], [], []
    traces = {}
    for item in commit["completed"]:
        directory = (root / item["index"]).parent
        a, index = checked_game(directory, deep)
        scenario = index["scenario"]
        if scenario in traces:
            assert traces[scenario] == index["trace_sha256"]
        traces[scenario] = index["trace_sha256"]
        label = index["policy"]
        tag = dict(policy=label, method=label.split("_seed")[0], scenario=scenario, training_seed=index["training_seed"])
        games.append(dict(**tag, **index["metrics"], runtime_s=index["runtime_s"], preview_calls=index["preview_calls"],
                          decision_median_ms=index["decision_median_ms"], decision_p95_ms=index["decision_p95_ms"], privileged=index["privileged"]))
        for start in range(0, len(a["rates"]), 1000):
            windows.append(dict(**tag, first=start+1, last=min(start+1000, len(a["rates"])), **metric(a, start, min(start+1000, len(a["rates"])))))
        cycles += [dict(method=tag["method"], policy=label, training_seed=index["training_seed"], **c)
                   for c in cycle_rows(a, index, label)]
    for policy in p["policies"]:
        row = read_json(root / "budget_ledger.json")[policy["label"]]
        n = sum(x["policy"] == policy["label"] for x in games) * p["horizon"]
        assert row["charged"] == row["observed"] == n
    all_labels = {x["label"] for x in p["policies"]}
    balanced = [s for s in p["scenarios"] if {r["policy"] for r in games if r["scenario"] == s} == all_labels]
    methods = list(dict.fromkeys(x["label"].split("_seed")[0] for x in p["policies"]))
    per_scenario, summary, paired = [], [], []
    metric_keys = ["utility", "throughput_mbps", "mean_qos_ue", "mean_positive_service_ue", "mean_slot_jain", "jain_mean_rate", "zero_reward_slots", "longest_zero_streak", "return_stranded_uav_slots", "return_wait_uav_slots", "relay_active_uav_slots", "relay_empty_slots"]
    for method in methods:
        for s in balanced:
            own = [r for r in games if r["method"] == method and r["scenario"] == s]
            per_scenario.append(dict(method=method, scenario=s, **{k: float(np.mean([r[k] for r in own])) for k in metric_keys}))
        own = [r for r in per_scenario if r["method"] == method]
        if not own:
            continue
        cc = [c for c in cycles if c["method"] == method and c["scenario"] in balanced]
        lat = [c["service_latency_slots"] for c in cc if c["service_latency_slots"] is not None]
        low, high = ci([r["utility"] for r in own])
        total_relay = sum(r["relay_active_uav_slots"] for r in own)
        summary.append(dict(method=method, common_test_scenarios=len(own), training_seeds=2 if method in ["C-R", "C-N"] else None,
                            **{k: float(np.mean([r[k] for r in own])) for k in metric_keys},
                            utility_ci95_low=low, utility_ci95_high=high,
                            zero_slot_fraction=float(np.mean([r["zero_reward_slots"] / p["horizon"] for r in own])),
                            empty_relay_fraction=sum(r["relay_empty_slots"] for r in own)/total_relay if total_relay else None,
                            cycles_started=len(cc), cycles_rejoined=sum(c["rejoin_active_slot"] is not None for c in cc),
                            cycles_served_again=len(lat), latency_median_observed=float(np.median(lat)) if lat else None,
                            censor_horizon=sum(c["censor_reason"] == "horizon" for c in cc),
                            next_return_without_service=sum(c["censor_reason"] == "next_return" for c in cc)))
    for s in balanced:
        cr = next(r for r in per_scenario if r["method"] == "C-R" and r["scenario"] == s)
        cn = next(r for r in per_scenario if r["method"] == "C-N" and r["scenario"] == s)
        paired.append(dict(scenario=s, **{k + "_CR_minus_CN": cr[k]-cn[k] for k in metric_keys}))
    for name, rows in [("games", games), ("windows1000", windows), ("cycles", cycles),
                       ("per_scenario", per_scenario), ("summary", summary), ("paired_CR_CN", paired)]:
        table(analysis / (name + ".csv"), rows)
        atomic_json(analysis / (name + ".json"), rows)
    if balanced:
        plt = figure_module()
        fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
        for method in methods:
            own = [r for r in windows if r["method"] == method and r["scenario"] in balanced]
            ends = sorted({r["last"] for r in own})
            ax.plot(ends, [np.mean([r["utility"] for r in own if r["last"] == end]) for end in ends], marker="o", markersize=3, label=method)
        ax.set(xlabel="Executed test slots", ylabel="Window utility", title="Common completed scenarios; 1000-slot windows")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8, ncol=2)
        fig.savefig(analysis / "utility_10k.png", dpi=180)
        fig.savefig(analysis / "utility_10k.pdf")
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        for ax, key in zip(axes, ["utility", "mean_qos_ue"]):
            ax.axhline(0, color="gray", linewidth=1)
            ax.plot(range(1, len(paired)+1), [r[key + "_CR_minus_CN"] for r in paired], "o")
            ax.set(xlabel="Common test scenario index", ylabel="C-R minus C-N", title=key)
            ax.grid(alpha=.2)
        fig.savefig(analysis / "paired_CR_CN.png", dpi=180)
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for i, method in enumerate(methods):
            cc = [c for c in cycles if c["method"] == method and c["scenario"] in balanced]
            lat = sorted(c["service_latency_slots"] for c in cc if c["service_latency_slots"] is not None)
            if lat:
                axes[0].step(lat, np.arange(1, len(lat)+1)/len(lat), where="post", label=method)
            axes[1].bar(i, sum(c["censor_reason"] == "horizon" for c in cc), color="#c8c8c8")
            axes[1].bar(i, sum(c["censor_reason"] == "next_return" for c in cc),
                        bottom=sum(c["censor_reason"] == "horizon" for c in cc), color="#d77722")
        axes[0].set(xlabel="Rejoin to service (slots)", ylabel="Conditional empirical CDF", title="Observed successful cycles only")
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            axes[0].legend(fontsize=7)
        axes[1].set(xticks=range(len(methods)), xticklabels=methods, ylabel="Cycle count", title="Unobserved service: horizon (gray), next Return (orange)")
        axes[1].tick_params(axis="x", labelrotation=60)
        fig.savefig(analysis / "postcharge_service.png", dpi=180)
        plt.close(fig)
    status = read_json(root / "status.json")
    atomic_json(analysis / "audit_receipt.json", dict(status=status, balanced_scenarios=balanced,
                audited_committed_slots=len(games)*p["horizon"], deep_planner_recomputed=deep,
                scenario_aggregation="average learned seeds within each scenario first; bootstrap common scenarios; not slots/cycles as independent samples"))
    lines = ["# Separate Return-v4 benchmark", "", f"Status: **{status['status']}**. Diagnostic: {p['diagnostic']}. Complete common scenarios: {len(balanced)}/{len(p['scenarios'])}.", "",
             "Summary uses only scenario blocks completed by all nine configurations (seven methods, two seeds for each learned arm). Raw games.csv retains every committed game. Ten scenarios remain ten independent scenario units, not twenty because RL has two seeds.", "",
             "| Method | Scenarios | Utility | QoS UE | Slot Jain | Zero-slot fraction |", "|---|---:|---:|---:|---:|---:|"]
    for r in summary:
        lines.append(f"| {r['method']} | {r['common_test_scenarios']} | {r['utility']:.6f} | {r['mean_qos_ue']:.2f} | {r['mean_slot_jain']:.4f} | {r['zero_slot_fraction']:.3%} |")
    lines += ["", "Utility confidence intervals bootstrap scenarios after averaging the two learned seeds within each scenario; they do not estimate broad uncertainty over training seeds. Two seeds are a limited replication.", "",
              "Greedies have privileged current exact UE/network access and isolated one-slot previews. They are not information- or compute-matched to QMIX. preview_calls and decision timings are reported in games.csv. No global optimum/oracle claim.", "",
              "postcharge_service.png shows a CDF conditional on observed service plus separate counts of horizon censoring/next-Return-without-service. It is not a censoring-adjusted survival estimate. Full cycle records remain in cycles.json.", "",
              "No training or checkpoint reselection occurs in benchmark. Models were selected by fixed validation before test. Throughput is a rescaling of utility, not independent confirmation."]
    (analysis / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return finish_export(root, out, "analysis/REPORT.md", "benchmark_resume.zip")


def preserve_failed_export(root, exc, archive_name):
    """Export partial evidence even when analysis cannot certify a complete run."""
    root = Path(root).resolve()
    out = fresh_export(root)
    message = ("# Export audit could not complete\n\n" + repr(exc) +
               "\n\nRaw output, partial game files, checkpoints and budget ledgers are retained. "
               "This archive is evidence, not a certified completed/resumable run. "
               "Preserve it for inspection; never reset counters to repeat uncommitted work.\n")
    (out / "analysis/REPORT.md").write_text(message, encoding="utf-8")
    atomic_json(out / "analysis/export_error.json", dict(error=repr(exc), status="audit_failed_raw_preserved"))
    full_archive(root, out / archive_name)
    pack(out / "review_small.zip", [(p, "analysis/"+p.name) for p in sorted((out/"analysis").iterdir())])
    atomic_json(out / "EXPORT_INDEX.json", dict(status=read_json(root / "status.json"), export_audit="failed",
                resume_archive=archive_name, files=[dict(path=x.name, sha256=file_hash(x), bytes=x.stat().st_size)
                for x in out.iterdir() if x.is_file()]))
    print("EXPORT AUDIT FAILED; raw evidence preserved:", repr(exc), flush=True)
    return out


def export_training(root, deep=False):
    try:
        return _export_training(root, deep)
    except Exception as exc:
        from scripts.training.run_bs5_v4_fresh_train import training_archive_names
        names = training_archive_names(read_json(Path(root) / "protocol.json"))
        return preserve_failed_export(root, exc, names["training_resume"])


def export_benchmark(root, deep=False):
    try:
        return _export_benchmark(root, deep)
    except Exception as exc:
        return preserve_failed_export(root, exc, "benchmark_resume.zip")
