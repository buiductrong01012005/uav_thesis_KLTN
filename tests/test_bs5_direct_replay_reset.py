"""Paired protocol, reset durability, export invariance and offline evidence integrity."""

import os
from copy import deepcopy
import random
import subprocess
import sys
import zipfile

import numpy as np
import pytest
import torch

from scripts.analysis.analyze_bs5_direct_replay_reset_bundle import analyze, validate
from scripts.analysis.bs5_bundle import (
    atomic_json,
    canonical_hash,
    file_hash,
    load_trace,
    read_json,
)
from scripts.training import run_bs5_direct_replay_reset as runner
from scripts.training.run_bs5_launch_ab import freeze_source, source_identity, write_trace


def inputs():
    config = read_json(runner.ROOT / "configs/bs5_direct_replay_reset_v1.json")
    return config, read_json(runner.ROOT / config["baseline"])


def create(root):
    config, baseline = inputs()
    identity = source_identity()
    p = runner.resolved_protocol(config, identity, baseline, smoke=True, device="cpu")
    p.update(
        training_seeds=[955551],
        steps_per_run=48,
        horizon=3,
        reset_steps=[6, 9, 12, 15, 18, 21],
        common_evaluation_steps=[0, 12, 24, 36, 48],
        evaluation_steps=[0, 12, 24, 36, 48],
        warmup=2,
        batch_size=2,
        update_every=2,
        target_every=6,
        epsilon_decay_steps=48,
        validation_pool=[180001],
        max_rollback_steps=12,
    )
    p["expected_updates"] = len(runner.update_schedule(p))
    p["expected_updates_by_lineage"] = [
        sum(lineage == k for _, lineage, _ in runner.update_schedule(p)) for k in range(7)
    ]
    root.mkdir()
    freeze_source(root, identity)
    atomic_json(root / "protocol.json", p)
    atomic_json(root / "baseline_refs.json", baseline)
    atomic_json(root / "source_diff.json", {})
    atomic_json(root / "schema.json", {})
    registry = dict(
        status="prepared",
        protocol_sha256=canonical_hash(p),
        attempts=[],
        runs={f"{runner.ARM}_seed955551": dict(status="not_started", steps=0)},
    )
    atomic_json(root / "run_registry.json", registry)
    return p, registry


def execute(root, p, registry, **kwargs):
    torch.set_num_threads(1)
    timers = runner.Timers(root, "bounded_test")
    try:
        runner.run_one(root, p, registry, 955551, timers, **kwargs)
    finally:
        timers.flush()


def signature(root):
    directory = root / "runs/D-R-reset_seed955551"
    state = torch.load(directory / "resume.pt", weights_only=False)
    return dict(
        final=state["commit"]["final_weights_sha256"],
        episodes=[
            read_json(f)["trajectory_sha256"]
            for f in sorted((directory / "episodes").glob("*.json"))
        ],
        updates=[read_json(f) for f in sorted((directory / "updates").glob("*.json"))],
        replay_rng=state["replay"].rng.getstate(),
        exploration=state["exploration_rng"],
        resets=state["commit"]["resets"],
    )


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    root = tmp_path_factory.mktemp("reset") / "complete"
    p, registry = create(root)
    execute(root, p, registry)
    validate(root, require_complete=True, check_manifest=False)
    return root


def test_exact_historical_schedule_scientific_source_and_update_expectation():
    config, baseline = inputs()
    p = runner.resolved_protocol(config, source_identity(), baseline, device="cpu")
    assert p["scenario_schedule"] == baseline["protocol"]["scenario_schedule"]
    assert p["reset_steps"] == [3000, 4500, 6000, 7500, 9000, 10500]
    assert p["expected_updates_by_lineage"] == [501, 126, 126, 126, 126, 126, 3126]
    assert p["expected_updates"] == 4257 and 12000 not in p["reset_steps"]
    assert p["max_rollback_steps"] == 3000
    assert (
        len(p["training_seeds"])
        * len(p["evaluation_steps"])
        * len(p["validation_pool"])
        * p["horizon"]
        == 27000
    )
    for seed in [551, 552]:
        assert (
            baseline["metas"][f"D-R_seed{seed}"]["initial_weights_sha256"]
            == baseline["metas"][f"C-R_seed{seed}"]["initial_weights_sha256"]
        )
        assert len(baseline["training_references"][str(seed)]) == 160


def test_main_initializer_rejects_unpaired_weights_before_rollout(monkeypatch):
    config, baseline = inputs()
    p = runner.resolved_protocol(config, source_identity(), baseline, device="cpu")
    monkeypatch.setattr(runner, "weights_hash", lambda _: "unpaired-runtime")
    with pytest.raises(AssertionError, match="Initial weights mismatch"):
        runner.initialize(p, 551, baseline)


def test_display_name_never_disables_relay_or_changes_full_task():
    config, baseline = inputs()
    p = runner.resolved_protocol(config, source_identity(), baseline, smoke=True, device="cpu")
    for step in [0] + p["reset_steps"]:
        env = runner.make_env(p["full_condition"], p, p["allow_relay"])
        out = env.reset(seed=130004)
        assert env.cfg.allow_relay and len(env.active_ids) == 5 and env.condition["stage"] == "D"
        assert (
            env.cfg.failed_bs_idx == 5
            and env.cfg.enable_ue_mobility
            and env.cfg.enable_battery_cycle
        )
        assert out["action_mask"][:, 1::2].any()
        assert runner.ARM.endswith("R") is False and step != 12000


def test_clear_preserves_adam_targets_exploration_and_collection_masks(completed):
    directory = completed / "runs/D-R-reset_seed955551"
    state = torch.load(directory / "resume.pt", weights_only=False)
    assert len(state["commit"]["resets"]) == 6
    for entry in state["commit"]["resets"]:
        assert entry["preserved_before"] == entry["preserved_after"]
    replay = state["replay"]
    assert replay.pos == 27 and len(replay) == 27 and replay.composition() == {"D": 27}
    assert all(t.lineage == 6 and t.collection_step >= 21 for t in replay.data)
    assert all(t.done == float((t.collection_step + 1) % 3 == 0) for t in replay.data)
    for t in replay.data:
        assert all(t.current_action_mask[i, a] for i, a in enumerate(t.actions))
        assert t.next_action_mask.any(axis=1).all()


@pytest.mark.parametrize(
    "boundary,phase",
    [(6, "before_clear"), (6, "after_clear"), (9, "before_clear"), (9, "after_clear")],
)
def test_resume_exactly_once_on_both_sides_of_first_two_clears(
    completed, tmp_path, boundary, phase
):
    root = tmp_path / "resumed"
    p, registry = create(root)
    with pytest.raises(KeyboardInterrupt):
        execute(root, p, registry, stop_at=(boundary, phase))
    directory = root / "runs/D-R-reset_seed955551"
    orphan = directory / "episodes/999999.json"
    atomic_json(orphan, {"uncommitted": True})
    execute(root, p, registry, resume=True)
    assert not orphan.exists() and list((directory / "aborted_attempts").rglob("999999.json"))
    assert signature(root) == signature(completed)
    validate(root, require_complete=True, check_manifest=False)


def test_real_frequent_exports_leave_tensor_actions_rewards_updates_unchanged(completed, tmp_path):
    root = tmp_path / "frequent"
    p, registry = create(root)
    python_rng, numpy_rng, torch_rng = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
    )
    assert python_rng and numpy_rng and torch_rng.numel()
    execute(root, p, registry, cadence_hook=lambda: runner.export_bundles(root))
    assert signature(root) == signature(completed)
    latest = runner.export_bundles(root)
    names_before = sorted((root / "exports").glob("*.zip"))
    assert runner.export_bundles(root)["snapshot_sha256"] == latest["snapshot_sha256"]
    assert sorted((root / "exports").glob("*.zip")) == names_before


def test_main_default_exports_per_completed_run_only(tmp_path, monkeypatch):
    root = tmp_path / "cadence"
    p, _ = create(root)
    assert p["export_policy"] == "per-run"
    called = []
    monkeypatch.setattr(
        runner,
        "export_bundles",
        lambda root, timers: called.append(read_json(root / "run_registry.json")),
    )
    runner.run_matrix(root)
    assert len(called) == 1 and called[0]["status"] == "complete"


def test_missing_tamper_and_cached_metric_mutation_rejected(completed, tmp_path):
    import shutil

    root = tmp_path / "mutated"
    shutil.copytree(completed, root)
    directory = root / "runs/D-R-reset_seed955551"
    path = directory / "evaluations/000000/180001.json.gz"
    original = path.read_bytes()
    path.unlink()
    with pytest.raises(AssertionError, match="file set"):
        validate(root, check_manifest=False)
    path.write_bytes(original + b"tampered")
    with pytest.raises(AssertionError, match="hash mismatch"):
        validate(root, check_manifest=False)
    path.write_bytes(original)
    packet = load_trace(path)
    packet["metrics"]["utility"] += 0.01
    write_trace(path, packet)
    index = read_json(directory / "packet_index.json")
    index[path.relative_to(directory).as_posix()] = dict(
        sha256=file_hash(path), bytes=path.stat().st_size
    )
    atomic_json(directory / "packet_index.json", index)
    commit = read_json(directory / "commit.json")
    commit["packet_index"] = index
    atomic_json(directory / "commit.json", commit)
    with pytest.raises(AssertionError, match="tampered cached metrics"):
        validate(root, check_manifest=False)


def test_offline_bundle_stdlib_only_no_checkpoints_and_byte_manifest(completed, tmp_path):
    receipt = runner.export_bundles(completed)
    name = next(n for n in receipt["bundles"] if n.endswith("_audit_light.zip"))
    evidence = tmp_path / "evidence"
    with zipfile.ZipFile(completed / "exports" / name) as archive:
        assert not any(n.endswith(".pt") for n in archive.namelist())
        archive.extractall(evidence)
    block = """
import builtins, sys
original = builtins.__import__
def blocked(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'numpy', 'src'}:
        raise RuntimeError('forbidden import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = blocked
from analyze_bs5_direct_replay_reset_bundle import analyze
analyze('.', sys.argv[1], require_complete=True)
print('stdlib-only checkpoint-free audit passed')
"""
    process = subprocess.run(
        [sys.executable, "-B", "-c", block, str(tmp_path / "rebuilt")],
        cwd=evidence,
        text=True,
        capture_output=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert process.returncode == 0, process.stdout + process.stderr
    path = evidence / "protocol.json"
    path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n") + b"\n")
    with pytest.raises(AssertionError, match="hash mismatch"):
        analyze(evidence, tmp_path / "bad")


def test_baseline_common_cadence_and_primary_numbers(tmp_path):
    from scripts.analysis.analyze_bs5_direct_replay_reset_bundle import baseline_rows, endpoint

    _, baseline = inputs()
    rows = baseline_rows(baseline, list(range(0, 24001, 3000)))
    assert len(rows) == 360 and {r["step"] for r in rows} == set(range(0, 24001, 3000))
    expected = {
        ("C-R", 551): 0.3425027042127932,
        ("C-R", 552): 0.3021356660042024,
        ("D-R", 551): 0.20861200358216264,
        ("D-R", 552): 0.299077179311678,
    }
    for (arm, seed), value in expected.items():
        selected = [r for r in rows if r["arm"] == arm and r["seed"] == seed]
        result = endpoint(f"{arm}_{seed}", arm, seed, selected, 24000)
        assert result["last_utility"] == pytest.approx(value, abs=1e-12)


@pytest.mark.parametrize("committed_end", [24000, 4500])
def test_comparison_tables_use_common_curve_and_pooled_relay_denominator(
    tmp_path, monkeypatch, committed_end
):
    from scripts.analysis import analyze_bs5_direct_replay_reset_bundle as auditor

    config, baseline = inputs()
    p = runner.resolved_protocol(config, source_identity(), baseline, device="cpu")
    atomic_json(tmp_path / "protocol.json", p)
    atomic_json(tmp_path / "baseline_refs.json", baseline)
    old = auditor.baseline_rows(baseline, p["evaluation_steps"])
    # Synthetic scorer input only: exercise comparisons using historical D-R values
    # relabelled as the new arm, without representing them as a real new run.
    rows = [
        dict(r, arm=runner.ARM, run=f"{runner.ARM}_seed{r['seed']}")
        for r in old
        if r["arm"] == "D-R" and r["step"] <= committed_end
    ]
    runs = [
        auditor.endpoint(
            f"{runner.ARM}_seed{seed}",
            runner.ARM,
            seed,
            [r for r in rows if r["seed"] == seed],
            committed_end,
            len(runner.update_schedule(p, committed_end)),
        )
        for seed in [551, 552]
    ]
    synthetic = dict(
        status="complete",
        scope="unit_test_synthetic",
        runs=runs,
        validation=rows,
        resets=[],
        training_steps=2 * committed_end,
        learner_updates=2 * len(runner.update_schedule(p, committed_end)),
        validation_slots=len(rows) * p["horizon"],
        verified_packet_bytes=0,
        verified_packet_count=0,
    )
    monkeypatch.setattr(auditor, "validate", lambda *args, **kwargs: deepcopy(synthetic))
    auditor.analyze(tmp_path, tmp_path / "analysis", check_manifest=False)
    import csv

    with (tmp_path / "analysis/comparisons.csv").open() as stream:
        comparisons = list(csv.DictReader(stream))
    primary = next(
        r
        for r in comparisons
        if r["seed"] == "mean"
        and r["selection"] == "last"
        and r["baseline"] == "D-R"
        and r["metric"] == "utility"
    )
    assert float(primary["ratio_of_means"]) == 1 and float(primary["difference"]) == 0
    common_end = max(r["step"] for r in rows)
    episodes = [r for r in rows if r["seed"] == 551 and r["step"] == common_end]
    expected = sum(r["relay_empty_slots"] for r in episodes) / sum(
        r["relay_active_uav_slots"] for r in episodes
    )
    measured = next(
        r
        for r in comparisons
        if r["seed"] == "551"
        and r["selection"] == "last"
        and r["baseline"] == "D-R"
        and r["metric"] == "empty_relay_fraction"
    )
    assert float(measured["reset"]) == pytest.approx(expected)


def test_seed_shards_keep_exact_original_members(completed, tmp_path):
    receipt = runner.export_bundles(completed)
    name = next(n for n in receipt["bundles"] if n.endswith("_audit_light.zip"))
    original = completed / "exports" / name
    parts = runner.seed_shards(original, tmp_path, "test", limit=1)
    assert len(parts) == 1 and parts[0]["partial"]
    with zipfile.ZipFile(original) as a, zipfile.ZipFile(tmp_path / parts[0]["name"]) as b:
        assert set(a.namelist()) == set(b.namelist())
        assert all(a.read(n) == b.read(n) for n in a.namelist())


def test_final_resume_repairs_json_registry_crash_window(completed, tmp_path):
    import shutil

    root = tmp_path / "crash_window"
    shutil.copytree(completed, root)
    p = read_json(root / "protocol.json")
    registry = read_json(root / "run_registry.json")
    registry["status"] = "incomplete"
    registry["runs"]["D-R-reset_seed955551"] = dict(status="incomplete", steps=36)
    atomic_json(root / "run_registry.json", registry)
    directory = root / "runs/D-R-reset_seed955551"
    stale = read_json(directory / "commit.json")
    stale["steps"] = 36
    atomic_json(directory / "commit.json", stale)
    execute(root, p, registry, resume=True)
    assert signature(root) == signature(completed)
    validate(root, require_complete=True, check_manifest=False)


def test_baseline_censor_values_are_booleans_and_restricted_time_is_explicit():
    from scripts.analysis.analyze_bs5_direct_replay_reset_bundle import (
        baseline_rows,
        selected_metric,
    )

    _, baseline = inputs()
    rows = baseline_rows(baseline, [0])
    assert all(isinstance(r["first_service_censored"], bool) for r in rows)
    assert (
        selected_metric([dict(first_service_slot=None, slots=150)], "first_service_slot_restricted")
        == 151
    )


@pytest.mark.parametrize(
    "writer_name,base_name",
    [("atomic_json", "bundle_atomic_json"), ("write_trace", "bundle_write_trace")],
)
def test_atomic_io_retries_transient_windows_lock_but_never_hides_denial(
    tmp_path, monkeypatch, writer_name, base_name
):
    original = getattr(runner, base_name)
    attempts, sleeps = [], []

    def transient(*args):
        attempts.append(args)
        if len(attempts) < 3:
            raise PermissionError("temporary cloud file lock")
        return original(*args)

    monkeypatch.setattr(runner, "WINDOWS_IO_ATTEMPTS", 3)
    monkeypatch.setattr(runner, base_name, transient)
    monkeypatch.setattr(runner.time, "sleep", sleeps.append)
    target = tmp_path / ("packet.json.gz" if writer_name == "write_trace" else "packet.json")
    getattr(runner, writer_name)(target, {"value": 42})
    assert len(attempts) == 3 and sleeps == [0.05, 0.1]
    assert (load_trace(target) if writer_name == "write_trace" else read_json(target)) == {
        "value": 42
    }

    def denied(*args):
        raise PermissionError("permanent denial")

    monkeypatch.setattr(runner, base_name, denied)
    with pytest.raises(PermissionError, match="permanent denial"):
        getattr(runner, writer_name)(target, {"value": 0})
    assert (load_trace(target) if writer_name == "write_trace" else read_json(target)) == {
        "value": 42
    }
