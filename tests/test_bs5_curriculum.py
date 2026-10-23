from copy import deepcopy
import pickle
import random
import subprocess
import sys
from types import ModuleType
import zipfile

import numpy as np
import pytest
import torch

from scripts.analysis.analyze_bs5_curriculum_bundle import (
    analyze,
    cached_analysis_matches,
    episode_metrics,
    validate,
)
from scripts.analysis.bs5_bundle import atomic_json, file_hash, load_trace, read_json
from scripts.training.run_bs5_curriculum_pilot import (
    ROOT,
    capture_decision,
    evaluate,
    freeze_source,
    make_env,
    offline_acceptance,
    protocol_new,
    record,
    registry_new,
    run_matrix,
    run_one,
    source_identity,
    write_trace,
)
from src.envs.bs5_curriculum_env import INACTIVE, condition_for
from src.envs.bs5_launch_ab_env import Geometry, make_env as target_env, score_network
from src.rl.bs5_curriculum_qmix import BS5CurriculumQMIX
from src.rl.bs5_curriculum_replay import CurriculumReplay, CurriculumTransition


def config():
    return read_json(ROOT / "configs/bs5_curriculum_fixed_v1.json")


def test_cached_analysis_accepts_linux_windows_newlines_but_rejects_changed_values(tmp_path):
    cached = tmp_path / "cached.json"
    generated = tmp_path / "rebuilt.json"
    cached.write_bytes(b'{\n  "utility": 0.25\n}\n')
    generated.write_bytes(b'{\r\n  "utility": 0.25\r\n}\r\n')
    assert cached_analysis_matches(cached, generated)
    generated.write_bytes(b'{\r\n  "utility": 0.26\r\n}\r\n')
    assert not cached_analysis_matches(cached, generated)
    cached, generated = tmp_path / "cached.csv", tmp_path / "rebuilt.csv"
    cached.write_bytes(b"run,utility\nC-R,0.25\n")
    generated.write_bytes(b"run,utility\r\nC-R,0.25\r\n")
    assert cached_analysis_matches(cached, generated)
    generated.write_bytes(b"run,utility\r\nC-R,0.26\r\n")
    assert not cached_analysis_matches(cached, generated)


def test_fixed_schedule_has_twelve_thousand_full_steps_and_no_gate_dependency():
    p = protocol_new(config(), {"sha256": "test"})
    assert p["stage_ends"] == [3000, 6000, 12000, 24000]
    spans = p["training_conditions"]["C-R"]
    assert spans[-1]["start"] == 12000 and spans[-1]["end"] == 24000
    assert condition_for(2999, "C-R", p)["stage"] == "S1"
    assert condition_for(3000, "C-N", p)["stage"] == "S2"
    assert signature_without_labels(condition_for(10500, "C-R", p)) == signature_without_labels(
        condition_for(12000, "C-R", p)
    )
    assert 4500 not in p["common_evaluation_steps"] and 4500 in p["evaluation_steps"]


def signature_without_labels(c):
    return {k: v for k, v in c.items() if k not in {"stage", "substage"}}


@pytest.mark.parametrize("step", [0, 3000, 4500, 6000])
def test_inactive_agents_never_serve_route_collide_or_reactivate(step):
    p = protocol_new(config(), {"sha256": "test"})
    c = condition_for(step, "C-R", p)
    active_sets = set()
    for assignment in range(6):
        env = make_env(c, p, True)
        env.reset(seed=140001, assignment_seed=assignment)
        active_sets.add(tuple(env.active_ids))
        inactive = np.flatnonzero(env.status == INACTIVE)
        positions = env.uav_positions[inactive].copy()
        battery = env.battery.copy()
        before = score_network(env)
        env.uav_positions[inactive] = env.uav_positions[env.active_ids[0]]
        env.roles[inactive] = 1
        assert score_network(env)["utility"] == before["utility"]
        for _ in range(8):
            out = env.step([8] * 5)
            assert len(np.flatnonzero(env.status == 0)) == 2
            assert np.all(env.status[inactive] == INACTIVE)
            network = score_network(env)
            assert set(network["association"]).isdisjoint(set(inactive))
            assert set(network["routes"]).isdisjoint(set(inactive))
            assert set(network["routes"].values()).isdisjoint(set(inactive))
            np.testing.assert_array_equal(env.battery, battery)
            assert out["info"]["status_names"].count("CurriculumInactive") == 3
        # An inactive object occupying the active start cannot block its movement.
        if not c["stay_only"]:
            agent = env.active_ids[0]
            valid = np.flatnonzero(env.get_action_mask()[agent])
            moving = [a for a in valid if a // 2 != 4]
            if moving:
                from src.envs.v0_env import MOVE_DELTAS

                expected = env.uav_positions[agent] + MOVE_DELTAS[moving[0] // 2]
                env.uav_positions[inactive] = expected
                actions = [8] * 5
                actions[agent] = int(moving[0])
                env.step(actions)
                np.testing.assert_array_equal(env.uav_positions[agent], expected)
        assert positions.shape == (3, 2)
    assert len(active_sets) > 1


@pytest.mark.parametrize("relay", [True, False])
def test_full_task_wrapper_matches_audited_target_through_lifecycle(relay):
    p = protocol_new(config(), {"sha256": "test"})
    env = make_env(condition_for(12000, "C-R", p), p, relay)
    target = target_env("B-R" if relay else "B-N")
    a, b = env.reset(seed=130004), target.reset(seed=130004)
    rng = np.random.default_rng(11)
    for _ in range(150):
        np.testing.assert_array_equal(a["obs"], b["obs"])
        np.testing.assert_array_equal(a["action_mask"], b["action_mask"])
        actions = [int(rng.choice(np.flatnonzero(mask))) for mask in a["action_mask"]]
        a, b = env.step(actions), target.step(actions)
        assert a["reward"] == b["reward"]
        assert env.rng.bit_generator.state == target.rng.bit_generator.state
        for key in ("uav_positions", "roles", "status", "battery", "ue_positions"):
            np.testing.assert_array_equal(getattr(env, key), getattr(target, key))


def test_replay_uses_collection_masks_and_keeps_terminal_boundary():
    replay = CurriculumReplay(2, 2)
    old = np.zeros((5, 10), dtype=np.int8)
    old[:, 8] = 1
    opened = np.ones((5, 10), dtype=np.int8)
    for eid, mask in enumerate((old, opened)):
        replay.add(
            CurriculumTransition(
                np.zeros((5, 116)),
                np.zeros(111),
                np.array([8] * 5),
                0.0,
                np.zeros((5, 116)),
                np.zeros(111),
                mask.copy(),
                float(eid == 0),
                mask.copy(),
                "S1" if eid == 0 else "S2",
                "S1" if eid == 0 else "S2a",
                eid,
                0,
            )
        )
    batch = replay.sample(2)
    assert sum(row[:, 1::2].any() for row in batch["next_action_mask"]) == 1
    assert sorted(batch["done"].tolist()) == [0.0, 1.0]
    assert replay.composition() == {"S1": 1, "S2a": 1}


def test_hand_calculated_shadow_witness():
    p = protocol_new(config(), {"sha256": "test"}, smoke=True)
    p["horizon"] = 1
    c = condition_for(0, "C-R", p)
    env = make_env(c, p, True)
    out = env.reset(seed=140001)
    decision = capture_decision(env)
    nxt = env.step([8] * 5)
    row = record(
        env,
        Geometry(env),
        decision,
        out["action_mask"],
        [8] * 5,
        nxt,
        p,
        "C-R",
        551,
        140001,
        1,
        0,
        c,
        {"agent_forward_calls": 1},
        evaluation=True,
    )
    for net, utility in (
        (row["service"]["network"], 0.4),
        (row["shadow"]["actual"], 0.4),
        (row["shadow"]["same_access_direct"], 0.2),
        (row["shadow"]["all_access"], 0.3),
    ):
        net.update(ue_rate_mbps=[utility * 4] * 90, utility=utility)
    row["shadow"]["role_enumeration"] = [dict(utility=0.5)]
    m = episode_metrics([row])
    for key, value in (
        ("route_gain", 0.2),
        ("role_opportunity_cost", 0.1),
        ("net_role_gain", 0.1),
        ("oracle_headroom", 0.1),
    ):
        assert m[key] == pytest.approx(value, abs=1e-12)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    root = tmp_path_factory.mktemp("curriculum")
    identity = source_identity()
    freeze_source(root, identity)
    p = protocol_new(config(), identity, smoke=True)
    p["training_seeds"] = [955551]
    atomic_json(root / "protocol.json", p)
    atomic_json(root / "schema.json", {"test": True})
    atomic_json(root / "run_registry.json", registry_new(p))
    run_matrix(root, export=False)
    return root


def test_all_stages_offline_without_checkpoints_and_exact_budgets(tiny, tmp_path):
    result = analyze(tiny, tmp_path, require_complete=True, check_manifest=False)
    assert result["validated_runs"] == 4 and result["training_steps"] == 1536
    cr = next(r for r in result["runs"] if r["arm"] == "C-R")
    assert cr["stage_budgets"] == {"S1": 48, "S2": 48, "S3": 96, "S4": 192}
    assert cr["updates"] < next(r for r in result["runs"] if r["arm"] == "D-R")["updates"]
    assert offline_acceptance(tiny)["torch_env_imports_blocked"]


def test_offline_validator_rejects_missing_chunk_and_rate_or_scalar_corruption(tiny):
    path = next((tiny / "runs").glob("*/evaluations/000000/*.gz"))
    original = path.read_bytes()
    packet = load_trace(path)
    try:
        path.unlink()
        with pytest.raises(FileNotFoundError):
            validate(tiny, check_manifest=False)
    finally:
        path.write_bytes(original)
    for field in ("rate", "scalar"):
        bad = deepcopy(packet)
        if field == "rate":
            bad["trace"][0]["service"]["network"]["ue_rate_mbps"][0] += 0.1
        else:
            bad["metrics"]["net_role_gain"] += 0.1
        write_trace(path, bad)
        try:
            with pytest.raises(AssertionError):
                validate(tiny, check_manifest=False)
        finally:
            path.write_bytes(original)


def test_eval_preserves_policy_global_rng_weights_optimizer_and_replay(tiny):
    p = read_json(tiny / "protocol.json")
    env = make_env(condition_for(0, "C-R", p), p, True)
    learner = BS5CurriculumQMIX(env)
    replay = CurriculumReplay(10, 1)
    state = (
        pickle.dumps(random.getstate()),
        pickle.dumps(np.random.get_state()),
        torch.get_rng_state().clone(),
        pickle.dumps(replay),
        deepcopy(learner.agent.state_dict()),
        deepcopy(learner.optimizer.state_dict()),
    )
    # Existing diagnostics are rewritten only with identical deterministic evidence.
    evaluate(learner, p, tiny / "eval_isolation_probe", "C-R", 955551, 0)
    assert state[0] == pickle.dumps(random.getstate()) and state[1] == pickle.dumps(
        np.random.get_state()
    )
    assert torch.equal(state[2], torch.get_rng_state()) and state[3] == pickle.dumps(replay)
    for key, value in state[4].items():
        torch.testing.assert_close(value, learner.agent.state_dict()[key], rtol=0, atol=0)
    assert state[5] == learner.optimizer.state_dict()


def test_cached_table_corruption_rejected_even_with_recomputed_manifest(tiny, tmp_path):
    evidence = tmp_path / "evidence"
    with zipfile.ZipFile(tiny / "exports/audit_light.zip") as bundle:
        bundle.extractall(evidence)
    table = evidence / "analysis/effects.csv"
    table.write_text(table.read_text() + "corrupted\n")
    manifest = read_json(evidence / "SHA256SUMS.json")
    manifest["analysis/effects.csv"] = file_hash(table)
    atomic_json(evidence / "SHA256SUMS.json", manifest)
    with pytest.raises(AssertionError, match="cached analysis mismatch"):
        analyze(evidence, tmp_path / "rebuilt", require_complete=True)


def test_kaggle_four_cells_discover_extracted_source_prepare_resume_export(tmp_path, monkeypatch):
    notebook = read_json(ROOT / "scripts/notebooks/bs5_curriculum_kaggle.ipynb")
    cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    assert len(cells) == 4
    dataset = tmp_path / "input/dataset/uav_thesis"
    for name in (
        "scripts/training/run_bs5_curriculum_pilot.py",
        "src/envs/bs5_curriculum_env.py",
        "AGENTS.md",
    ):
        path = dataset / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("sentinel")
    atomic_json(dataset / "configs/bs5_curriculum_fixed_v1.json", config())
    work = tmp_path / "working"
    calls = []

    def fake_run(command, **kwargs):
        assert kwargs["cwd"] == dataset and kwargs["check"]
        out = work / "bs5_curriculum_fixed_20261023"
        calls.append(command)
        if "--prepare" in command:
            p = read_json(work / "bs5_curriculum_fixed_20261023_config.json")
            p["steps_per_run"] = p["requested_steps_per_run"]
            atomic_json(out / "protocol.json", p)
            atomic_json(out / "preflight/smoke_acceptance.json", {"accepted": True})
        elif "--run" in command:
            assert "--resume" in command
            atomic_json(out / "run_registry.json", {"status": "complete"})
        elif "--export" in command:
            (out / "exports").mkdir(exist_ok=True)
            for kind in ("audit_light", "repro_full"):
                with zipfile.ZipFile(out / f"exports/{kind}.zip", "w") as bundle:
                    bundle.writestr("test.txt", "cell wiring only; no main training")
        else:
            raise AssertionError(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    display_module = ModuleType("IPython.display")
    display_module.FileLink = lambda name: name
    display_module.display = lambda value: None
    monkeypatch.setitem(sys.modules, "IPython.display", display_module)
    monkeypatch.chdir(tmp_path)
    namespace = {}
    for cell in cells:
        code = cell.replace("/kaggle/input", (tmp_path / "input").as_posix()).replace(
            "/kaggle/working", work.as_posix()
        )
        exec(compile(code, "kaggle_cell", "exec"), namespace)
    for cell in cells[1:3]:
        exec(
            compile(cell.replace("/kaggle/working", work.as_posix()), "kaggle_resume", "exec"),
            namespace,
        )
    assert sum("--prepare" in c for c in calls) == 1
    assert sum("--run" in c for c in calls) == 2
    assert len(list(work.glob("*_*.zip"))) == 2


def test_resume_at_stage_boundary_matches_uninterrupted_weights_optimizer_replay(tiny, tmp_path):
    p = read_json(tiny / "protocol.json")
    atomic_json(tmp_path / "protocol.json", p)
    registry = registry_new(p)
    atomic_json(tmp_path / "run_registry.json", registry)
    with pytest.raises(KeyboardInterrupt):
        run_one(tmp_path, p, registry, "C-R", 955551, export=False, stop_after=48)
    # Exercise archival of an uncommitted event packet.
    atomic_json(tmp_path / "runs/C-R_seed955551/events/000008.json", [{"aborted": True}])
    run_one(tmp_path, p, registry, "C-R", 955551, resume=True, export=False)
    actual = torch.load(tmp_path / "runs/C-R_seed955551/resume.pt", weights_only=False)
    expected = torch.load(tiny / "runs/C-R_seed955551/resume.pt", weights_only=False)
    for group in ("agent", "mixer", "target_agent", "target_mixer"):
        for key, value in expected[group].items():
            torch.testing.assert_close(value, actual[group][key], rtol=0, atol=0)
    assert actual["commit"]["updates"] == expected["commit"]["updates"]
    assert actual["replay"].rng.getstate() == expected["replay"].rng.getstate()
    assert actual["replay"].composition() == expected["replay"].composition()
    for a, b in zip(actual["replay"].data, expected["replay"].data):
        assert a.collection_step == b.collection_step and a.lineage == b.lineage
        np.testing.assert_array_equal(a.next_action_mask, b.next_action_mask)
    for key, value in expected["optimizer"]["state"].items():
        for field, tensor in value.items():
            torch.testing.assert_close(
                tensor, actual["optimizer"]["state"][key][field], rtol=0, atol=0
            )
    assert list((tmp_path / "runs/C-R_seed955551/aborted_attempts").rglob("000008.json"))
