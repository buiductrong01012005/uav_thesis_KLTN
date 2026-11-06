"""Separate final benchmark; consumes frozen validation-selected training outputs."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from scripts.experiments.bs5_v4_fresh_common import (
    Budget, BS5CurriculumQMIX, act, atomic_json, canonical_hash, env_for, experiment_lock,
    file_hash, full_condition, read_json, rollout, setup, source_identity, weights_hash,
)
from scripts.training.run_bs5_v4_fresh_train import all_complete, read_selection
from src.baselines.return_v4_safe_greedy import make_greedy


def read_commit(root):
    versions = sorted((Path(root) / "commits").glob("*.json"))
    return read_json(versions[-1] if versions else Path(root) / "commit.json")


def save_commit(root, commit):
    path = root / "commits" / f"{len(commit['completed']):05d}.json"
    assert not path.exists(), "Duplicate benchmark commit."
    atomic_json(path, commit)
    atomic_json(root / "commit.json", commit)


def prepare(root, training, device="cpu", scenarios=10, timing_profile=None):
    root, training = Path(root).resolve(), Path(training).resolve()
    assert not root.exists(), "Existing benchmark output; resume it rather than prepare again."
    if (training / "HANDOFF_SHA256SUMS.json").exists():
        for rel, digest in read_json(training / "HANDOFF_SHA256SUMS.json").items():
            assert file_hash(training / rel) == digest, f"Handoff checksum failed: {rel}"
    p = read_json(training / "protocol.json")
    expected_seeds = {955551, 955552} if p["diagnostic"] else {551, 552}
    assert set(p["arms"]) == {"C-R", "C-N"} and set(p["training_seeds"]) == expected_seeds, "Merge all four completed arm/seed jobs before benchmark."
    plan = read_json(training / "benchmark_plan.json")
    assert canonical_hash(plan) == p["benchmark_plan_sha256"]
    assert p["source_sha256"] == source_identity()["sha256"]
    assert all_complete(training, p), "Finish all four training runs and validation before the final benchmark."
    assert scenarios in [10, 20]
    seeds = plan["test_scenarios"] + (plan["reserved_scenarios"] if scenarios == 20 else [])
    h = plan["horizon"]
    if p["diagnostic"]:
        seeds, h = [280901], 12
    root.mkdir(parents=True)
    (root / "models").mkdir()
    policies = []
    for seed in p["training_seeds"]:
        for arm in p["arms"]:
            label = f"{arm}_seed{seed}"
            run = training / "runs" / label
            selection = read_selection(run)
            best = next(x for x in selection["curve"] if x["step"] == selection["best_step"])
            original = run / "checkpoints" / f"{best['step']:06d}.pt"
            assert file_hash(original) == best["checkpoint_sha256"]
            dest = root / "models" / f"{label}.pt"
            shutil.copy2(original, dest)
            policies.append(dict(label=label, arm=arm, training_seed=seed, path=dest.relative_to(root).as_posix(),
                                 step=best["step"], sha256=file_hash(dest), weights_sha256=best["weights_sha256"], privileged=False))
    policies += [dict(label=x, arm=x, training_seed=None, privileged=x != "Random-Relay") for x in plan["baselines"]]
    jobs = [dict(scenario=s, policy=policy["label"]) for s in seeds for policy in policies]
    timings = {}
    if timing_profile is not None:
        receipt = read_json(timing_profile)
        assert receipt["status"] == "passed" and receipt["identity"]["source_sha256"] == p["source_sha256"]
        assert receipt["identity"]["runtime"] == setup("cpu")
        timings = receipt["benchmark_startup_timing"]
    protocol = dict(version=plan["version"], diagnostic=p["diagnostic"], training_protocol=p,
                    training_protocol_sha256=canonical_hash(p), source_sha256=p["source_sha256"],
                    plan=plan, scenarios=seeds, horizon=h, policies=policies, jobs=jobs,
                    runtime=setup(device), planned_steps=len(jobs) * h, diagnostic_startup_timing=timings,
                    previews="isolated one-slot simulator calls; accounted separately; no modification of executed env")
    if (training / "TRAINING_PROVENANCE.json").exists():
        protocol["training_job_provenance"] = read_json(training / "TRAINING_PROVENANCE.json")
    for name in ["training_bins100.csv", "training_episodes.csv", "validation_curve.csv", "runs.csv", "learning_curves.png", "learning_curves.pdf"]:
        original = training / "analysis" / name
        if original.exists():
            (root / "training_analysis").mkdir(exist_ok=True)
            shutil.copy2(original, root / "training_analysis" / name)
    atomic_json(root / "protocol.json", protocol)
    save_commit(root, dict(completed=[]))
    Budget(root / "budget_ledger.json", {x["label"]: len(seeds) * h for x in policies})
    atomic_json(root / "status.json", dict(status="prepared", completed=0, total=len(jobs)))
    return protocol


def run(root, max_seconds=10 * 3600, max_games=None):
    root = Path(root).resolve()
    protocol = read_json(root / "protocol.json")
    assert setup(protocol["runtime"]["device"]) == protocol["runtime"], "Benchmark runtime changed."
    assert source_identity()["sha256"] == protocol["source_sha256"]
    p, plan = protocol["training_protocol"], protocol["plan"]
    policies = {x["label"]: x for x in protocol["policies"]}
    with experiment_lock(root):
        commit = read_commit(root)
        budget = Budget(root / "budget_ledger.json")
        assert [{k: x[k] for k in ["scenario", "policy"]} for x in commit["completed"]] == protocol["jobs"][:len(commit["completed"])]
        for label, row in budget.data.items():
            n = sum(x["policy"] == label for x in commit["completed"]) * protocol["horizon"]
            assert row["charged"] == row["observed"] == n, "Uncommitted game: preserve evidence; no silent replay."
        started, count, models, status = time.monotonic(), 0, {}, "paused_at_game_boundary"
        try:
            for job in protocol["jobs"][len(commit["completed"]):]:
                estimates = [read_json(root / x["index"])["runtime_s"] for x in commit["completed"] if x["policy"] == job["policy"]]
                initial_estimate = protocol.get("diagnostic_startup_timing", {}).get(job["policy"].split("_seed")[0], {}).get("extrapolated_10k_s", 300.)
                predicted = max(estimates, default=initial_estimate)
                if max_seconds - (time.monotonic() - started) < max(600., predicted * 1.5) or (max_games is not None and count >= max_games):
                    return
                policy = policies[job["policy"]]
                scenario = job["scenario"]
                spec = dict(condition=full_condition(p), horizon=protocol["horizon"], scenario_seed=scenario,
                            assignment_seed=scenario, residual_battery=30., diagnostic=True)
                rng = np.random.default_rng(np.random.SeedSequence([scenario, plan["random_seed_namespace"]]))
                heuristic = None
                before = None
                if policy.get("path"):
                    cp = root / policy["path"]
                    assert file_hash(cp) == policy["sha256"]
                    if policy["label"] not in models:
                        learner = BS5CurriculumQMIX(env_for(p, spec, root / "traces", policy["arm"]),
                                                   lr=p["lr"], gamma=p["gamma"], device=protocol["runtime"]["device"])
                        saved = torch.load(cp, map_location=learner.device, weights_only=True)
                        learner.agent.load_state_dict(saved["agent"])
                        learner.mixer.load_state_dict(saved["mixer"])
                        learner.agent.eval()
                        models[policy["label"]] = learner
                    learner = models[policy["label"]]
                    before = weights_hash(learner)
                    assert before == policy["weights_sha256"]
                def choose(env, out):
                    nonlocal heuristic
                    if policy.get("path"):
                        return act(learner, out, 0., rng), 0., 0
                    if policy["label"] == "Random-Relay":
                        return [int(rng.choice(np.flatnonzero(mask))) for mask in out["action_mask"]], 0., 0
                    if heuristic is None:
                        heuristic = make_greedy(policy["label"], env, plan["zero_service_patience"])
                    elif env.t > 0:
                        # Observe the PREVIOUS executed slot, never a hypothetical score.
                        heuristic.observe_service(float(env.last_service_snapshot["network"]["utility"]))
                    n = heuristic.preview_calls
                    actions = heuristic.act(out["action_mask"])
                    return actions, 0., heuristic.preview_calls - n
                directory = root / "games" / policy["label"] / str(scenario)
                index = rollout(directory, p, spec, policy["arm"], root / "traces", budget, policy["label"], choose)
                if before:
                    assert weights_hash(learner) == before
                index.update(policy=policy["label"], training_seed=policy["training_seed"],
                             privileged=policy["privileged"], weights_sha256=before,
                             checkpoint_sha256=policy.get("sha256"), algorithm_seed_namespace=plan["random_seed_namespace"] if policy["label"] == "Random-Relay" else None)
                atomic_json(directory / "index.json", index)
                commit["completed"].append(dict(**job, index=(directory / "index.json").relative_to(root).as_posix()))
                save_commit(root, commit)
                count += 1
                print(dict(phase="benchmark_committed", **job, completed=len(commit["completed"]), total=len(protocol["jobs"])), flush=True)
            status = "complete"
        except BaseException as exc:
            status = "partial_error"
            atomic_json(root / "error.json", dict(error=repr(exc), instruction="Preserve output/ledger; no hidden repeats."))
            raise
        finally:
            atomic_json(root / "status.json", dict(status=status, completed=len(commit["completed"]), total=len(protocol["jobs"])))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["prepare", "run", "export"])
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--training", type=Path)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--scenarios", type=int, choices=[10, 20], default=10)
    ap.add_argument("--max-seconds", type=float, default=10 * 3600)
    ap.add_argument("--max-games", type=int)
    ap.add_argument("--deep-planner", action="store_true")
    ap.add_argument("--timing-profile", type=Path)
    a = ap.parse_args()
    if a.command == "prepare":
        assert a.training is not None
        prepare(a.output, a.training, a.device, a.scenarios, a.timing_profile)
    elif a.command == "run":
        run(a.output, a.max_seconds, a.max_games)
    else:
        from scripts.analysis.bs5_v4_fresh_report import export_benchmark
        print(export_benchmark(a.output, deep=a.deep_planner))


if __name__ == "__main__":
    main()
