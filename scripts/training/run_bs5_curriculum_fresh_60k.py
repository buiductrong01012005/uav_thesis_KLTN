"""New 0->60k experiment authorized when historical full states are unavailable.

Only C-R/C-N, seeds551/552. Reuses the original curriculum runner through24k,
then restores its NEW full states into the optimized 24->60k tail runner. All
240k environment steps are fresh compute; no historical evidence is imported.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from scripts.analysis.bs5_bundle import atomic_json, canonical_hash, file_hash, read_json
from scripts.training.run_bs5_curriculum_pilot import protocol_new, registry_new, run_one
from scripts.training.run_bs5_curriculum_extend import (
    DEFAULT,
    FRESH_ORIGIN,
    check_runtime,
    finalize,
    names,
    prepare,
    run as run_tail,
    runtime,
)
from scripts.training.run_bs5_launch_ab import experiment_lock, freeze_source, source_identity
from scripts.training.run_bs5_direct_replay_reset import Timers

SPEC = ROOT / "configs/bs5_curriculum_fresh_60k_v1.json"
SCIENCE = ROOT / "configs/bs5_scientific_source_lock_v1.json"


def check_scientific_source():
    lock = read_json(SCIENCE)
    assert lock["historical_source_sha256"] == read_json(DEFAULT)["parent_source_sha256"]
    for rel, digest in lock["files"].items():
        assert rel.startswith("src/") and file_hash(ROOT / rel) == digest, (
            f"scientific source drift: {rel}"
        )


def prepare_fresh(output, *, device="auto", diagnostic_config=None, diagnostic_policy=None):
    output = Path(output).resolve()
    assert not output.exists(), "new output required; use run --resume on prepared outputs"
    check_scientific_source()
    spec = read_json(SPEC)
    config = read_json(ROOT / "configs/bs5_curriculum_fixed_v1.json")
    if diagnostic_config is not None:
        assert diagnostic_policy and diagnostic_policy.get("diagnostic")
        config = deepcopy(diagnostic_config)
        assert all(s >= 900000 for s in config["training_seeds"])
        assert config["requested_steps_per_run"] < diagnostic_policy["total_steps"] <= 1000
    config["device"] = (
        ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    )
    assert not config["device"].startswith("cuda") or torch.cuda.is_available(), (
        "Enable GPU in Kaggle settings"
    )
    torch.set_num_threads(config["torch_threads"])
    identity = source_identity()
    p = protocol_new(config, identity)
    p["arms"] = spec["arms"]
    p["training_conditions"] = {arm: p["training_conditions"][arm] for arm in p["arms"]}
    p["scope"] = "diagnostic_fresh_prefix_not_main" if diagnostic_config else spec["scope"]
    p["origin"] = FRESH_ORIGIN
    p["execution_environment"] = dict(runtime(), device=config["device"])
    policy = dict(
        read_json(DEFAULT),
        origin=FRESH_ORIGIN,
        parent_source_sha256=identity["sha256"],
        scope=spec["scope"],
    )
    if diagnostic_policy:
        policy.update(diagnostic_policy)
        policy.update(origin=FRESH_ORIGIN, parent_source_sha256=identity["sha256"])
    else:
        assert (
            p["training_seeds"] == spec["training_seeds"]
            and p["steps_per_run"] == spec["prefix_steps"]
        )
    output.mkdir(parents=True)
    prefix = output / "prefix24k"
    prefix.mkdir()
    freeze_source(prefix, identity)
    atomic_json(prefix / "protocol.json", p)
    registry = registry_new(p)
    registry["runs"] = {name: dict(status="not_started", steps=0) for name in names(policy)}
    atomic_json(prefix / "run_registry.json", registry)
    atomic_json(
        output / "fresh_experiment.json",
        dict(
            version=spec["version"],
            origin=FRESH_ORIGIN,
            source_sha256=identity["sha256"],
            prefix_protocol_sha256=canonical_hash(p),
            policy=policy,
            diagnostic=bool(diagnostic_policy),
            historical_imported_steps=0,
            planned_new_training_steps=policy["total_steps"] * len(registry["runs"]),
            runtime=p["execution_environment"],
            note="Independent new 0->60k experiment; no archival24k metrics/weights are spliced into this lineage.",
        ),
    )
    return p


def budget(output):
    output = Path(output)
    experiment = read_json(output / "fresh_experiment.json")
    policy = experiment["policy"]
    prefix = read_json(output / "prefix24k/run_registry.json")
    tail_file = output / "tail36k/budget_ledger.json"
    tail = read_json(tail_file) if tail_file.exists() else {}
    executed_prefix = sum(a["executed_train_steps"] for a in prefix["attempts"])
    durable_prefix = sum(e["steps"] for e in prefix["runs"].values())
    result = dict(
        origin=FRESH_ORIGIN,
        historical_imported_steps=0,
        planned_new_training_steps=experiment["planned_new_training_steps"],
        prefix_executed_steps=executed_prefix,
        prefix_durable_steps=durable_prefix,
        tail_executed_steps=tail.get("new_executed_steps", 0),
        tail_durable_steps=tail.get("durable_new_steps", 0),
        whole_fresh_executed_steps=executed_prefix + tail.get("new_executed_steps", 0),
        whole_fresh_durable_steps=durable_prefix + tail.get("durable_new_steps", 0),
        runs={
            n: dict(
                prefix_status=e["status"],
                prefix_steps=e["steps"],
                total_steps=read_json(output / "tail36k/run_registry.json")["runs"][n]["steps"]
                if tail_file.exists()
                else e["steps"],
            )
            for n, e in prefix["runs"].items()
        },
        prefix_attempts=prefix["attempts"],
        tail_attempts=tail.get("attempts", []),
        uncertain_unflushed_steps_upper_bound=sum(
            a.get("uncertain_unflushed_steps_upper_bound", 0) for a in prefix["attempts"]
        )
        + tail.get("uncertain_unflushed_steps_upper_bound", 0),
        count_semantics="Fresh compute includes prefix attempts and tail attempts, including rollback; never imported historical compute.",
    )
    assert set(result["runs"]) == set(names(policy))
    atomic_json(output / "fresh_budget_ledger.json", result)
    return result


def recover_abrupt_prefix(prefix, registry, name, horizon):
    """Account completed uncommitted episodes before the old runner archives them."""
    directory = prefix / "runs" / name
    for attempt in registry["attempts"]:
        if attempt["run"] == name and attempt["status"] == "running":
            ends = [read_json(f)["end_step"] for f in (directory / "episodes").glob("*.json")]
            last = max(ends, default=attempt["from_step"])
            attempt["executed_train_steps"] = max(
                attempt["executed_train_steps"], last - attempt["from_step"]
            )
            attempt.update(
                status="abrupt_restart",
                uncertain_unflushed_steps_upper_bound=horizon,
                reason="Completed episode lower bound recovered before rollback; up to one in-flight episode unflushed.",
            )
    for path in directory.rglob("*.tmp"):
        if "aborted_attempts" in path.relative_to(directory).parts:
            continue
        assert path.resolve().is_relative_to(directory.resolve()) and not path.is_symlink()
        target = (
            directory
            / "aborted_attempts"
            / f"atomic_{time.time_ns()}"
            / path.relative_to(directory)
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        path.rename(target)
    atomic_json(prefix / "run_registry.json", registry)


def run_fresh(output, *, resume=False, prefix_stop_after=None, export=True):
    output = Path(output).resolve()
    experiment = read_json(output / "fresh_experiment.json")
    assert experiment["origin"] == FRESH_ORIGIN
    assert source_identity()["sha256"] == experiment["source_sha256"], (
        "fresh source changed; use frozen source"
    )
    prefix = output / "prefix24k"
    p, registry = read_json(prefix / "protocol.json"), read_json(prefix / "run_registry.json")
    assert canonical_hash(p) == experiment["prefix_protocol_sha256"]
    policy = experiment["policy"]
    torch.set_num_threads(p["torch_threads"])
    check_runtime(
        policy, dict(p, parent_execution_environment=p["execution_environment"]), runtime()
    )
    with experiment_lock(output):
        try:
            for name in names(policy):
                if registry["runs"][name]["status"] == "complete":
                    continue
                assert resume or registry["runs"][name]["status"] == "not_started", "use --resume"
                arm, seed = name.split("_seed")
                recover_abrupt_prefix(prefix, registry, name, p["horizon"])
                timer = Timers(prefix, f"fresh_prefix_{name}", p["device"])
                try:
                    with timer.phase("prefix_train_eval_persistence_combined"):
                        run_one(
                            prefix,
                            p,
                            registry,
                            arm,
                            int(seed),
                            resume=resume,
                            export=False,
                            stop_after=prefix_stop_after,
                            final_stage_diagnostics=False,
                        )
                finally:
                    timer.flush(
                        run=name,
                        scope="original prefix runner; train/eval/update children in commit.json",
                    )
                    budget(output)
            registry["status"] = "complete"
            atomic_json(prefix / "run_registry.json", registry)
            tail_dir = output / "tail36k"
            if not tail_dir.exists():
                prepare(prefix, tail_dir, policy)
            if export:
                run_tail(tail_dir, resume=True)
            else:
                # Only for bounded tests. Production always uses per-run finalization.
                assert experiment["diagnostic"]
                from scripts.training.run_bs5_curriculum_extend import tail

                ep = read_json(tail_dir / "protocol.json")
                reg = read_json(tail_dir / "run_registry.json")
                for name in names(policy):
                    if reg["runs"][name]["status"] != "complete":
                        arm, seed = name.split("_seed")
                        tail(tail_dir, ep, reg, arm, int(seed), Timers(tail_dir, "diagnostic_tail"))
                reg["status"] = "complete"
                atomic_json(tail_dir / "run_registry.json", reg)
        finally:
            budget(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare-fresh", "run", "analyze", "export"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--analysis-output", type=Path)
    args = parser.parse_args()
    if args.command == "prepare-fresh":
        prepare_fresh(args.output, device=args.device)
        budget(args.output)
    elif args.command == "run":
        run_fresh(args.output, resume=args.resume)
    elif args.command == "export":
        finalize(args.output / "tail36k")
        budget(args.output)
    else:
        from scripts.analysis.analyze_bs5_curriculum_extend import analyze

        assert args.analysis_output, "--analysis-output outside evidence required"
        analyze(args.output / "tail36k", args.analysis_output)


if __name__ == "__main__":
    main()
