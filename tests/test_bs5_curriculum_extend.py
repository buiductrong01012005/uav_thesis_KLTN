"""Bounded diagnostic parents only; no 24k main rollout in this suite."""

from copy import deepcopy
import os
import pickle
import shutil
import subprocess
import sys
from types import ModuleType
import zipfile

import numpy as np
import pytest
import torch

from scripts.analysis.bs5_bundle import atomic_json, file_hash, read_json
from scripts.analysis.analyze_bs5_curriculum_extend import analyze, validate
from scripts.training.run_bs5_curriculum_extend import (
    DEFAULT,
    ROOT,
    check_runtime,
    extend_protocol,
    finalize,
    names,
    preflight,
    prepare,
    tail,
    run,
)
from scripts.training.run_bs5_curriculum_pilot import (
    act,
    condition_for,
    evaluate,
    make_env,
    protocol_new,
    registry_new,
    run_one,
)
from scripts.training.run_bs5_launch_ab import (
    freeze_source,
    load_state,
    save_state,
    source_identity,
)
from scripts.training.run_bs5_direct_replay_reset import Timers
from src.rl.bs5_curriculum_qmix import BS5CurriculumQMIX


def exact(a, b):
    if torch.is_tensor(a):
        assert torch.equal(a.cpu(), b.cpu())
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            exact(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            exact(x, y)
    else:
        assert a == b


def exact_replay(a, b):
    assert a.capacity == b.capacity and a.pos == b.pos and a.composition() == b.composition()
    assert a.rng.getstate() == b.rng.getstate() and len(a) == len(b)
    for x, y in zip(a.data, b.data):
        exact(vars(x), vars(y))


def test_main_schedule_frozen_prefix_and_only_s4_extension():
    p = protocol_new(read_json(ROOT / "configs/bs5_curriculum_fixed_v1.json"), {"sha256": "test"})
    before = deepcopy(p)
    policy = read_json(DEFAULT)
    new = extend_protocol(p, policy, {"sha256": "new"})
    assert p == before and new["stage_ends"] == [3000, 6000, 12000, 60000]
    assert len(names(policy)) == 4 and len(new["common_evaluation_steps"]) == 21
    for seed in (551, 552):
        assert new["scenario_schedule"][str(seed)][:160] == p["scenario_schedule"][str(seed)]
        assert len(new["scenario_schedule"][str(seed)]) == 400
    assert (
        new["scenario_schedule"]
        == extend_protocol(p, policy, {"sha256": "new"})["scenario_schedule"]
    )
    for step in (23999, 24000, 24001, 59999):
        assert condition_for(step, "C-R", new) == condition_for(23999, "C-R", p)
    assert len([s for s in new["common_evaluation_steps"] if s > 24000]) == 12
    assert 4 * 12 * 10 * 150 == 72000
    assert sum([3000, 3000, 6000, 48000]) == 60000


def test_missing_parents_and_runtime_fail_closed(tmp_path):
    policy = read_json(DEFAULT)
    report = preflight(tmp_path / "missing", policy)
    assert (
        not report["ready"] and len([p for p in report["missing"] if p.endswith("resume.pt")]) == 4
    )
    assert report["main_steps_executed"] == 0
    with pytest.raises(AssertionError, match="Torch2.11"):
        check_runtime(policy, {"device": "cpu"}, {"torch": "2.13.0+cpu"})
    with pytest.raises(RuntimeError, match="Missing/incompatible"):
        prepare(tmp_path / "missing", tmp_path / "blocked", policy)
    assert not (tmp_path / "blocked/runs").exists()


@pytest.fixture(scope="module")
def diagnostic(tmp_path_factory):
    root = tmp_path_factory.mktemp("extend60k_diagnostic")
    parent = root / "parent"
    parent.mkdir()
    torch.set_num_threads(1)
    identity = source_identity()
    freeze_source(parent, identity)
    config = read_json(ROOT / "configs/bs5_curriculum_fixed_v1.json")
    config.update(
        training_seeds=[955551, 955552],
        horizon=2,
        stage_ends=[8, 16, 32, 48],
        requested_steps_per_run=48,
        eval_every=8,
        warmup=2,
        batch_size=2,
        replay_capacity=128,
        target_every=8,
        epsilon_decay_steps=32,
        validation_scenario_seeds=[180001, 180001],
        stage_validation_scenario_seeds=[181001, 181001],
        train_scenario_seeds=[182001, 182004],
        fixture_scenario_seeds=[183001, 183004],
    )
    p = protocol_new(config, identity)
    p["scope"] = "diagnostic_parent_not_main"
    atomic_json(parent / "protocol.json", p)
    registry = registry_new(p)
    for seed in p["training_seeds"]:
        for arm in ("C-R", "C-N"):
            run_one(parent, p, registry, arm, seed, export=False)
    c = read_json(parent / "runs/C-R_seed955551/commit.json")
    policy = dict(
        read_json(DEFAULT),
        diagnostic=True,
        scope="diagnostic_smoke_not_main",
        training_seeds=p["training_seeds"],
        parent_steps=48,
        total_steps=80,
        parent_source_sha256=identity["sha256"],
        parent_updates=c["updates"],
        final_updates=c["updates"] + 8,
    )
    return root, parent, p, policy


def test_full_state_roundtrip_all_rngs_and_eval_isolation(diagnostic, tmp_path):
    _, parent, p, _ = diagnostic
    original = torch.load(parent / "runs/C-R_seed955551/resume.pt", weights_only=False)
    learner = BS5CurriculumQMIX(make_env(condition_for(48, "C-R", p), p, True))
    rng = np.random.default_rng(99)
    commit, replay = load_state(parent / "runs/C-R_seed955551/resume.pt", learner, rng)
    save_state(tmp_path, learner, replay, rng, commit)
    actual = torch.load(tmp_path / "resume.pt", weights_only=False)
    for key in original.keys() - {"replay"}:
        exact(original[key], actual[key])
    assert pickle.dumps(original["replay"]) == pickle.dumps(actual["replay"])
    evaluate(learner, p, tmp_path / "eval", "C-R", 955551, 48)
    save_state(tmp_path / "after_eval", learner, replay, rng, commit)
    after = torch.load(tmp_path / "after_eval/resume.pt", weights_only=False)
    for key in actual.keys() - {"replay"}:
        exact(actual[key], after[key])
    assert pickle.dumps(after["replay"]) == pickle.dumps(actual["replay"])


def test_tail_resume_matches_uninterrupted_and_masks(diagnostic, tmp_path):
    _, parent, p, policy = diagnostic
    report = preflight(parent, policy)
    assert report["ready"], report
    before = {
        f.relative_to(parent).as_posix(): file_hash(f) for f in parent.rglob("*") if f.is_file()
    }
    extension = tmp_path / "extension"
    new = prepare(parent, extension, policy)
    registry = read_json(extension / "run_registry.json")
    assert len(registry["runs"]) == 4 and all(
        e["status"] == "imported_pending_tail" for e in registry["runs"].values()
    )
    # Copy the imported durable snapshot, then interrupt/resume one lineage.
    interrupted = tmp_path / "interrupted"
    shutil.copytree(extension, interrupted)
    reg2 = read_json(interrupted / "run_registry.json")
    with pytest.raises(KeyboardInterrupt):
        tail(interrupted, new, reg2, "C-R", 955551, Timers(interrupted, "probe"), stop_after=56)
    # Uncommitted chunks must be archived without modifying committed evidence.
    atomic_json(interrupted / "runs/C-R_seed955551/episodes/000028.json", {"orphan": True})
    (interrupted / "runs/C-R_seed955551/events/000028.json.tmp").write_text("partial atomic write")
    tail(interrupted, new, reg2, "C-R", 955551, Timers(interrupted, "resume"))
    for seed in policy["training_seeds"]:
        for arm in policy["arms"]:
            tail(extension, new, registry, arm, seed, Timers(extension, f"smoke_{arm}_{seed}"))
    actual = torch.load(interrupted / "runs/C-R_seed955551/resume.pt", weights_only=False)
    expected = torch.load(extension / "runs/C-R_seed955551/resume.pt", weights_only=False)
    for key in (
        "agent",
        "mixer",
        "target_agent",
        "target_mixer",
        "optimizer",
        "exploration_rng",
        "python_rng",
        "numpy_rng",
        "torch_rng",
        "cuda_rng",
    ):
        exact(actual[key], expected[key])
    exact_replay(actual["replay"], expected["replay"])
    assert actual["commit"]["updates"] == expected["commit"]["updates"] == policy["final_updates"]
    assert actual["commit"]["lineage"] == report["runs"]["C-R_seed955551"]["lineage"]
    for arm in ("C-R", "C-N"):
        env = make_env(condition_for(48, arm, new), new, arm == "C-R")
        out = env.reset(seed=180001)
        if arm == "C-N":
            assert not out["action_mask"][:, 1::2].any()
            learner = BS5CurriculumQMIX(env)
            actions, _ = act(learner, out, 1, np.random.default_rng(8))
            assert all(a % 2 == 0 for a in actions)
        else:
            assert out["action_mask"][:, 1::2].any()
    registry["status"] = "complete"
    atomic_json(extension / "run_registry.json", registry)
    result = analyze(extension, tmp_path / "analysis", require_complete=True)
    assert len(result["runs"]) == 4
    assert all(r["steps"] == 80 for r in result["runs"])
    assert all(v["parent_raw_validated_episodes"] == 7 for v in result["coverage"].values())
    receipt = finalize(extension)
    hashes = {a["path"]: file_hash(extension / a["path"]) for a in receipt["artifacts"]}
    assert finalize(extension) == receipt
    run(extension, resume=True)
    assert hashes == {a["path"]: file_hash(extension / a["path"]) for a in receipt["artifacts"]}
    assert abs(receipt["closing_timer"]["residual_wall_s"]) < 1e-5
    assert receipt["closing_timer"]["outer"]["export"] > 0
    audit = next(extension / a["path"] for a in receipt["artifacts"] if "audit_light" in a["path"])
    offline = tmp_path / "offline"
    with zipfile.ZipFile(audit) as z:
        assert not any(n.endswith(".pt") for n in z.namelist())
        z.extractall(offline)
    # Block Torch/NumPy/env imports in a clean stdlib-only subprocess.
    code = """
import importlib.abc,sys,runpy
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch','numpy','src','matplotlib'):
   raise RuntimeError('blocked scientific import '+fullname)
sys.meta_path.insert(0,Deny())
sys.path.insert(0,sys.argv[1])
sys.argv=[sys.argv[1]+'/analyze_bs5_curriculum_extend.py',sys.argv[1],'--output',sys.argv[2],'--require-complete']
runpy.run_path(sys.argv[0],run_name='__main__')
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-c", code, str(offline), str(tmp_path / "rebuilt")],
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert read_json(tmp_path / "rebuilt/verification.json")["accepted"]
    # Missing, duplicate, scalar and rate corruption fail even without manifest.
    (offline / "SHA256SUMS.json").unlink()
    packet = next((offline / "runs").glob("*/evaluations/*/*.gz"))
    original = packet.read_bytes()
    packet.unlink()
    with pytest.raises((AssertionError, FileNotFoundError)):
        validate(offline)
    packet.write_bytes(original)
    extra = packet.with_name("duplicate.json.gz")
    shutil.copyfile(packet, extra)
    with pytest.raises(AssertionError):
        validate(offline)
    extra.unlink()
    from scripts.analysis.bs5_bundle import load_trace
    from scripts.training.run_bs5_launch_ab import write_trace
    from scripts.analysis.bs5_bundle import canonical_hash

    for field in ("scalar", "rate"):
        bad = load_trace(packet)
        if field == "scalar":
            bad["metrics"]["net_role_gain"] += 0.1
        else:
            bad["trace"][0]["service"]["network"]["ue_rate_mbps"][0] += 0.1
        write_trace(packet, bad)
        # Bypass only the checksum layer to exercise independent raw validators.
        directory = packet.parents[2]
        commit = read_json(directory / "commit.json")
        rel = packet.relative_to(directory).as_posix()
        saved = deepcopy(commit)
        commit["extension_packet_index"][rel] = dict(
            sha256=file_hash(packet), bytes=packet.stat().st_size
        )
        atomic_json(directory / "commit.json", commit)
        with pytest.raises(AssertionError):
            validate(offline)
        packet.write_bytes(original)
        atomic_json(directory / "commit.json", saved)
    broken = read_json(offline / "protocol.json")
    broken["scenario_schedule"][str(policy["training_seeds"][0])][0]["fixture_seed"] += 1
    registry_off = read_json(offline / "run_registry.json")
    registry_off["protocol_sha256"] = canonical_hash(broken)
    atomic_json(offline / "protocol.json", broken)
    atomic_json(offline / "run_registry.json", registry_off)
    with pytest.raises(AssertionError, match="frozen prefix"):
        validate(offline)
    assert before == {
        f.relative_to(parent).as_posix(): file_hash(f) for f in parent.rglob("*") if f.is_file()
    }
    assert list((interrupted / "runs/C-R_seed955551/aborted_attempts").rglob("000028.json"))
    assert list((interrupted / "runs/C-R_seed955551/aborted_attempts").rglob("000028.json.tmp"))
    ledger = read_json(extension / "budget_ledger.json")
    assert ledger["imported_steps"] == 192 and ledger["new_executed_steps"] == 128


def test_preflight_rejects_weights_only_or_tampered_parent(diagnostic, tmp_path):
    _, parent, _, policy = diagnostic
    altered = tmp_path / "badparent"
    shutil.copytree(parent, altered)
    path = altered / "runs/C-R_seed955551/resume.pt"
    torch.save({"agent": {}}, path)
    report = preflight(altered, policy)
    assert not report["ready"] and "not full resume" in report["errors"][0]


def test_no_relay_bootstrap_ignores_illegal_high_q():
    p = protocol_new(read_json(ROOT / "configs/bs5_curriculum_fixed_v1.json"), {"sha256": "test"})
    env = make_env(condition_for(24000, "C-N", p), p, False)
    learner = BS5CurriculumQMIX(env)
    other = deepcopy(learner)
    with torch.no_grad():
        for agent in (learner.agent, other.agent, learner.target_agent, other.target_agent):
            for param in agent.parameters():
                param.zero_()
        learner.agent.head[-1].bias[1::2] = 1e6
        learner.target_agent.head[-1].bias[1::2] = 100
        other.target_agent.head[-1].bias[1::2] = 100
    batch = dict(
        obs=np.zeros((2, 5, 116), np.float32),
        state=np.ones((2, 111), np.float32),
        actions=np.zeros((2, 5), np.int64),
        reward=np.zeros(2, np.float32),
        next_obs=np.zeros((2, 5, 116), np.float32),
        next_state=np.ones((2, 111), np.float32),
        next_action_mask=np.tile(np.array([True, False] * 5), (2, 5, 1)),
        done=np.zeros(2, np.float32),
    )
    a, b = learner.train_batch(batch), other.train_batch(batch)
    assert a["target_q_mean"] == b["target_q_mean"]
    assert a["loss"] == b["loss"]


def test_kaggle_cells_find_extracted_parents_and_resume(tmp_path, monkeypatch):
    notebook = read_json(ROOT / "scripts/notebooks/bs5_curriculum_extend_60k_kaggle.ipynb")
    cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    assert len(cells) == 4
    input_root, work = tmp_path / "input", tmp_path / "working"
    dataset = input_root / "source/uav_thesis"
    runner = dataset / "scripts/training/run_bs5_curriculum_extend.py"
    runner.parent.mkdir(parents=True)
    runner.write_text("source sentinel")
    parent = input_root / "parents"
    parent.mkdir()
    atomic_json(parent / "protocol.json", {"test": "not a real parent"})
    (parent / "source_snapshot.zip").write_text("source snapshot sentinel")
    for name in ("C-R_seed551", "C-R_seed552", "C-N_seed551", "C-N_seed552"):
        d = parent / "runs" / name
        d.mkdir(parents=True)
        (d / "resume.pt").write_text("full-state discovery sentinel; never loaded")
    work.mkdir()
    calls = []

    def fake_run(command, **kwargs):
        assert kwargs["cwd"] == dataset and kwargs["check"]
        calls.append(command)
        action = command[3]
        out = work / "bs5_curriculum_extend_60k"
        if action == "prepare-extension":
            atomic_json(out / "protocol.json", {"test": "prepared"})
        elif action in ("run", "export"):
            if action == "run":
                assert "--resume" in command
            artifacts = []
            for kind in ("review_small", "audit_light", "repro_full"):
                f = out / "exports" / f"{kind}_test.zip"
                f.parent.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(f, "w") as z:
                    z.writestr("scope.txt", "cell wiring test only")
                artifacts.append(dict(path=f.relative_to(out).as_posix(), sha256=file_hash(f)))
            atomic_json(out / "exports/receipt_test.json", dict(artifacts=artifacts))
        else:
            assert action == "preflight"
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(torch, "__version__", "2.11.0+cu128")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(np, "__version__", "2.1.3")
    display_module = ModuleType("IPython.display")
    display_module.FileLink = lambda name: name
    display_module.display = lambda value: None
    monkeypatch.setitem(sys.modules, "IPython.display", display_module)
    monkeypatch.chdir(tmp_path)
    namespace = {}
    for cell in cells:
        code = cell.replace("/kaggle/input", input_root.as_posix()).replace(
            "/kaggle/working", work.as_posix()
        )
        exec(compile(code, "kaggle_cell", "exec"), namespace)
    for cell in cells[1:3]:
        exec(compile(cell, "kaggle_resume", "exec"), namespace)
    assert sum(c[3] == "prepare-extension" for c in calls) == 1
    assert sum(c[3] == "run" for c in calls) == 2
    assert len(list(work.glob("*.zip"))) == 3 and (work / "receipt_test.json").exists()
