"""Merge completed disjoint training partitions for inference/benchmark ONLY.

Retain each original protocol/runtime, validate shared recipe, paired initial
weights, actual UE tape hashes and validation-selected checkpoints. Never
average weights, select models on test, resume training, or silently dedupe jobs.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from scripts.analysis.bs5_bundle import atomic_json, canonical_hash, file_hash, read_json
from scripts.analysis.bs5_v4_fresh_report import figure_module, table
from scripts.training.run_bs5_v4_fresh_train import all_complete, read_progress, read_selection
from scripts.training.run_bs5_launch_ab import source_identity


def recipe(p):
    variable = {"arms", "training_seeds", "schedules", "planned_training_steps", "planned_validation_steps", "runtime"}
    return {k: v for k, v in p.items() if k not in variable}


def open_input(path, cache):
    path = Path(path).resolve()
    if path.is_dir():
        assert (path / "protocol.json").exists(), f"Select the run/handoff directory: {path}"
        return path
    assert path.suffix == ".zip"
    dest = cache / file_hash(path)
    assert not dest.exists(), "Duplicate archive input."
    dest.mkdir(parents=True)
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None, "Input ZIP CRC failed"
        for m in z.infolist():
            target = (dest / m.filename).resolve()
            assert target.is_relative_to(dest) and (m.external_attr >> 16) & 0o170000 != 0o120000, "Unsafe ZIP"
        z.extractall(dest)
    candidates = [x for x in dest.glob("*/protocol.json")]
    assert len(candidates) == 1, "Handoff must have one top-level protocol"
    return candidates[0].parent


def verify_input(root):
    manifest = read_json(root / "HANDOFF_SHA256SUMS.json")
    for rel, digest in manifest.items():
        target = (root / rel).resolve()
        assert target.is_relative_to(root) and file_hash(target) == digest, f"Handoff checksum failed: {rel}"
    p = read_json(root / "protocol.json")
    assert p["source_sha256"] == source_identity()["sha256"], "Use the SAME new parallel Dataset for every notebook."
    assert read_json(root / "status.json")["status"] == "complete" and all_complete(root, p), "Partition incomplete"
    plan = read_json(root / "benchmark_plan.json")
    assert canonical_hash(plan) == p["benchmark_plan_sha256"]
    tapes = read_json(root / "analysis/MATCHED_TRACE_RECEIPT.json")
    for key, value in tapes.items():
        assert key == canonical_hash(value["spec"])
    return p, plan, tapes


def plot_combined(out, p, inputs):
    analysis = out / "analysis"
    analysis.mkdir()
    bins, curve, episodes, run_rows = [], [], [], []
    for root in inputs:
        for name, target in [("training_bins100", bins), ("validation_curve", curve),
                             ("training_episodes", episodes), ("runs", run_rows)]:
            with (root / "analysis" / (name + ".csv")).open(newline="") as f:
                target.extend(csv.DictReader(f))
    for name, rows in [("training_bins100", bins), ("validation_curve", curve), ("training_episodes", episodes), ("runs", run_rows)]:
        table(analysis / (name + ".csv"), rows)
    plt = figure_module()
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    for seed in p["training_seeds"]:
        for arm in p["arms"]:
            own = sorted([r for r in bins if r["arm"] == arm and int(r["training_seed"]) == seed], key=lambda r: int(r["training_step"]))
            color, style = ("#1766a3" if arm == "C-R" else "#d77722"), ("-" if seed == p["training_seeds"][0] else "--")
            label = f"{arm} / {seed}"
            axes[0].plot([int(r["training_step"]) for r in own], [float(r["utility"]) for r in own], color=color, alpha=.17, linewidth=.7)
            for stage in dict.fromkeys(r["stage"] for r in own):
                rows = [r for r in own if r["stage"] == stage]
                y = [np.average([float(x["utility"]) for x in rows[max(0, j-4):j+1]],
                                weights=[int(x["slots"]) for x in rows[max(0, j-4):j+1]]) for j in range(len(rows))]
                axes[0].plot([int(r["training_step"]) for r in rows], y, color=color, linestyle=style,
                             label=label if stage == own[0]["stage"] else None)
            val = sorted([r for r in curve if r["arm"] == arm and int(r["training_seed"]) == seed], key=lambda r: int(r["step"]))
            axes[1].plot([int(r["step"]) for r in val], [float(r["utility"]) for r in val], color=color, linestyle=style, marker="o", markersize=3, label=label)
    for ax in axes:
        ax.set_ylabel("Utility")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
        for step in p["stage_ends"][:-1]:
            ax.axvline(step, color="gray", alpha=.3, linewidth=.7)
    axes[0].set_title("Separate jobs: raw bins + trailing 5-bin mean within each substage")
    axes[1].set_title("Common fixed-task validation; actual measured checkpoints")
    axes[1].set_xlabel("Training environment steps per run")
    for ext in ["png", "pdf"]:
        fig.savefig(analysis / ("learning_curves." + ext), dpi=180)
    plt.close(fig)


def merge(output, paths):
    out = Path(output).resolve()
    assert not out.exists(), "Never overwrite a merged bundle."
    out.mkdir(parents=True)
    cache = out / "_transport"
    inputs = [open_input(p, cache) for p in paths]
    protocols, jobs, tapes, plans = [], {}, {}, []
    provenance = {}
    for root in inputs:
        p, plan, own_tapes = verify_input(root)
        if protocols:
            assert recipe(p) == recipe(protocols[0]), "Scientific recipe/source/test plan differs between jobs"
            for field in ["python", "numpy", "torch", "threads", "device", "cuda", "deterministic"]:
                assert p["runtime"][field] == protocols[0]["runtime"][field], f"Training runtime differs: {field}"
        protocols.append(p)
        plans.append(plan)
        for key, value in own_tapes.items():
            if key in tapes:
                assert tapes[key] == value, "Actual UE tape or initial state mismatch"
            tapes[key] = value
        for seed in p["training_seeds"]:
            for arm in p["arms"]:
                key = (arm, seed)
                assert key not in jobs, f"Duplicate job {key}: select exactly one completed output per job"
                run = root / "runs" / f"{arm}_seed{seed}"
                progress, selection = read_progress(run), read_selection(run)
                assert [r["step"] for r in selection["curve"]] == p["evaluation_steps"]
                best = max(selection["curve"], key=lambda r: (r["utility"], -r["step"]))
                assert best["step"] == selection["best_step"] == progress["best_step"]
                cp = run / "checkpoints" / f"{best['step']:06d}.pt"
                assert file_hash(cp) == best["checkpoint_sha256"]
                payload = torch.load(cp, map_location="cpu", weights_only=True)
                assert payload["protocol_sha256"] == canonical_hash(p) and payload["arm"] == arm and payload["training_seed"] == seed
                assert payload["weights_sha256"] == best["weights_sha256"]
                jobs[key] = (root, run, progress, selection)
                provenance[f"{arm}_seed{seed}"] = dict(protocol=p, protocol_sha256=canonical_hash(p), best_step=best["step"], checkpoint_sha256=best["checkpoint_sha256"])
    seeds = sorted({seed for arm, seed in jobs})
    expected_seeds = [955551, 955552] if protocols[0]["diagnostic"] else [551, 552]
    assert seeds == expected_seeds and set(jobs) == {(arm, seed) for arm in ["C-R", "C-N"] for seed in seeds}, "Need all four jobs C-R/C-N x 551/552"
    for seed in seeds:
        cr, cn = jobs[("C-R", seed)], jobs[("C-N", seed)]
        pa, pb = read_json(cr[0] / "protocol.json"), read_json(cn[0] / "protocol.json")
        assert pa["schedules"][str(seed)] == pb["schedules"][str(seed)], "Paired game schedules differ"
        assert cr[2]["initial_weights_sha256"] == cn[2]["initial_weights_sha256"], "Paired initial weights differ"
        for spec in pa["schedules"][str(seed)]:
            tape_spec = {k: spec[k] for k in ["condition", "horizon", "scenario_seed", "assignment_seed", "residual_battery", "diagnostic"]}
            key = canonical_hash(tape_spec)
            assert key in read_json(cr[0] / "analysis/MATCHED_TRACE_RECEIPT.json")
            assert key in read_json(cn[0] / "analysis/MATCHED_TRACE_RECEIPT.json")
    combined = deepcopy(protocols[0])
    combined.update(arms=["C-R", "C-N"], training_seeds=seeds,
                    schedules={str(seed): next(p["schedules"][str(seed)] for p in protocols if str(seed) in p["schedules"]) for seed in seeds},
                    planned_training_steps=4*combined["total_steps"],
                    planned_validation_steps=4*len(combined["evaluation_steps"])*len(combined["validation_pool"])*combined["full_horizon"],
                    runtime=dict(scope="Per-job original runtimes in TRAINING_PROVENANCE.json; merged bundle is inference-only"))
    atomic_json(out / "protocol.json", combined)
    atomic_json(out / "benchmark_plan.json", plans[0])
    atomic_json(out / "TRAINING_PROVENANCE.json", provenance)
    atomic_json(out / "MATCHED_TRACE_RECEIPT.json", tapes)
    for (arm, seed), (root, run, progress, selection) in jobs.items():
        target = out / "runs" / f"{arm}_seed{seed}"
        target.mkdir(parents=True)
        atomic_json(target / "progress.json", progress)
        atomic_json(target / "checkpoint_selection.json", selection)
        for step in {selection["best_step"], combined["total_steps"]}:
            path = Path("checkpoints") / f"{step:06d}.pt"
            (target / path).parent.mkdir(exist_ok=True)
            shutil.copy2(run / path, target / path)
    atomic_json(out / "status.json", dict(status="complete", bundle_kind="merged_inference_only", jobs=sorted(provenance), training_steps=sum(x[2]["steps"] for x in jobs.values()), merge_training_steps=0, merge_test_steps=0))
    plot_combined(out, combined, inputs)
    (out / "analysis/REPORT.md").write_text("# Four completed training jobs merged\n\nShared science/runtime stack, schedules, initial weights and actual UE tape hashes verified. Original runtimes and protocol hashes retained per job. Best checkpoints fixed by validation. No training, test rollout or checkpoint reselection during merge. This bundle cannot resume training.\n")
    # Copies have already been verified; staging is disposable extracted transport.
    if cache.exists():
        shutil.rmtree(cache)
    atomic_json(out / "HANDOFF_SHA256SUMS.json", {f.relative_to(out).as_posix(): file_hash(f) for f in sorted(out.rglob("*")) if f.is_file()})
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--inputs", nargs="+", type=Path, required=True)
    args = ap.parse_args()
    print(merge(args.output, args.inputs))
