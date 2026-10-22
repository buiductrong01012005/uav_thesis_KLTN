from copy import deepcopy
import pickle

import numpy as np
import pytest
import torch

from scripts.analysis.bs5_bundle import (
    analyze,
    atomic_json,
    load_trace,
    read_json,
    validate,
)
from scripts.training.run_bs5_launch_ab import (
    ROOT,
    freeze_source,
    load_state,
    make_protocol,
    offline_acceptance,
    registry_new,
    run_matrix,
    run_one,
    save_state,
    schema,
    source_identity,
    source_files,
    weights_hash,
    write_trace,
)
from src.envs.bs5_launch_ab_env import (
    ARMS,
    BS5LaunchConfig,
    BS5LaunchEnv,
    Geometry,
    make_env,
    score_network,
    shadow_scores,
)
from src.envs.episode_random_failure_env import (
    EpisodeFailureCartesianConfig,
    EpisodeFailureCartesianEnv,
)
from src.rl.bs5_launch_qmix import BS5LaunchQMIX
from src.rl.qmix import ReplayBuffer, Transition


def test_source_inventory_works_without_windows_junction_api(monkeypatch):
    import os

    monkeypatch.delattr(os.path, "isjunction", raising=False)
    files = source_files()
    assert ROOT / "AGENTS.md" in files
    assert ROOT / "src/envs/bs5_launch_ab_env.py" in files
    assert not any(p.suffix in {".pyc", ".pyo"} for p in files)


def fixture_router():
    env = BS5LaunchEnv(BS5LaunchConfig(num_ues=2, num_hotspots=1, ue_demand_mbps=20))
    env.reset(seed=7)
    env.status[:] = 1
    env.status[:3] = 0
    env.roles[:] = 0
    env.roles[2] = 1
    env.uav_positions[:3] = [[14, 8], [14, 10], [13, 9]]
    env.ue_positions[:] = [[14, 8], [14, 10]]
    relay_xy = env._cell_to_xy(env.uav_positions[2])
    env._bh_a2g_rate = lambda xy, *args: 10.0 if np.array_equal(xy, relay_xy) else 9.0
    env._bh_a2a_rate = lambda *args: 20.0
    env._shannon_rate_mbps = lambda *args: 20.0
    return env


def test_shared_relay_fixture_never_loses_direct_rates_or_double_counts():
    env = fixture_router()
    before = score_network(env, False)
    after = score_network(env)
    np.testing.assert_array_equal(before["ue_rate_mbps"], [9, 9])
    assert np.all(after["ue_rate_mbps"] >= before["ue_rate_mbps"])
    assert sorted(after["ue_rate_mbps"]) == [9, 10]
    assert sum(after["relay_allocations_mbps"][2].values()) <= 10
    assert sum(after["ue_rate_mbps"]) == 19
    assert len([u for group in after["association"].values() for u in group]) == 2


def test_relay_with_no_direct_fallback_really_serves_traffic():
    env = make_env("B-R")
    env.reset(seed=140001)
    env.uav_positions[:2] = [[19, 16], [16, 11]]
    env.roles[:2] = [0, 1]
    env.status[2:] = 2
    env.ue_positions[:] = [19, 19]
    direct, required = Geometry(env).ue_masks(env.ue_positions)
    assert not direct.any() and required.all()
    assert score_network(env, False)["utility"] == 0
    assert score_network(env)["utility"] > 0


@pytest.mark.parametrize("arm", ARMS)
def test_bs5_reset_layout_gateway_residual_pairing(arm):
    env = make_env(arm)
    out = env.reset(seed=130001)
    assert (env.state_dim, env.obs_dim, env.n_actions) == (111, 116, 10)
    assert out["state"][105:111].argmax() == 5
    np.testing.assert_array_equal(env.failed_bs_xy, [925, 925])
    assert not any(np.array_equal(p, env.bs_xy[5]) for p in env.surviving_bs_xy)
    assert all(env._ue_position_is_valid(p) for p in env.ue_positions)
    assert env.residual_uav_idx == 4
    assert env.battery[4] == 30
    assert np.all(env.roles == 0)
    with pytest.raises(ValueError):
        env.reset(seed=1, options={"failed_bs_idx": 3})


@pytest.mark.parametrize("arm", ["A-N", "B-N"])
def test_no_relay_mask_and_execution_including_lifecycle(arm):
    env = make_env(arm)
    out = env.reset(seed=130001)
    assert not out["action_mask"][:, 1::2].any()
    with pytest.raises(ValueError):
        env.step([9] * 5)
    rng = np.random.default_rng(77)
    states = set()
    for _ in range(150):
        actions = [int(rng.choice(np.flatnonzero(m))) for m in out["action_mask"]]
        out = env.step(actions)
        states.update(env.status.tolist())
        assert not out["action_mask"][:, 1::2].any()
        assert all(r == 0 for r, s in zip(env.roles, env.status) if s == 0)
        assert all(r is None for r in out["info"]["routes"].values())
    assert states == {0, 1, 2}


def test_same_movement_preserves_legacy_physics_and_direct_scalar_rates():
    current = make_env("A-N")
    old = EpisodeFailureCartesianEnv(EpisodeFailureCartesianConfig(failed_bs_idx=5))
    a = current.reset(seed=140001)
    b = old.reset(seed=140001)
    rng = np.random.default_rng(5)
    for _ in range(150):
        actions = [int(rng.choice(np.flatnonzero(m))) for m in a["action_mask"]]
        a, b = current.step(actions), old.step(actions)
        assert a["reward"] == b["reward"]
        for key in ("uav_positions", "roles", "battery", "status", "ue_positions"):
            np.testing.assert_array_equal(getattr(current, key), getattr(old, key))
        assert current.rng.bit_generator.state == old.rng.bit_generator.state


def test_logger_and_oracle_are_observational_at_service_phase():
    env = make_env("B-R")
    env.reset(seed=140002)
    ue, battery = env.ue_positions.copy(), env.battery.copy()
    out = env.step([8] * 5)
    snap = env.last_service_snapshot
    np.testing.assert_array_equal(snap["ue_positions"], ue)
    np.testing.assert_array_equal(snap["battery"], battery)
    assert not np.array_equal(env.ue_positions, ue)
    before = pickle.dumps(env.__dict__)
    shadow = shadow_scores(env, snap, oracle=True)
    assert pickle.dumps(env.__dict__) == before
    assert shadow["actual"]["utility"] == out["reward"]
    assert len(shadow["role_enumeration"]) == 32
    assert shadow["same_access_direct"]["association"] == shadow["actual"]["association"]


def test_random_geometries_never_degrade_same_access_direct():
    env = make_env("B-R")
    env.reset(seed=140003)
    rng = np.random.default_rng(91)
    for _ in range(50):
        ids = rng.choice(400, 5, replace=False)
        env.uav_positions[:] = np.array(np.unravel_index(ids, (20, 20))).T
        env.roles[:] = rng.integers(0, 2, 5)
        env.status[:] = rng.integers(0, 3, 5)
        actual, direct = score_network(env), score_network(env, False)
        assert np.all(actual["ue_rate_mbps"] + 1e-8 >= direct["ue_rate_mbps"])
        for rid, allocation in actual["relay_allocations_mbps"].items():
            assert sum(allocation.values()) <= actual["relay_gateway_caps_mbps"][rid] + 1e-8


def test_checkpoint_restores_optimizer_targets_replay_rng_and_next_update(tmp_path):
    torch.set_num_threads(1)
    env = make_env("B-N")
    out = env.reset(seed=140001)
    learner = BS5LaunchQMIX(env)
    replay = ReplayBuffer(32, 7)
    rng = np.random.default_rng(8)
    for _ in range(8):
        actions = learner.act(out["obs"], out["action_mask"], 1, rng)
        nxt = env.step(actions)
        replay.add(
            Transition(
                out["obs"],
                out["state"],
                np.array(actions),
                nxt["reward"],
                nxt["obs"],
                nxt["state"],
                nxt["action_mask"],
                0.0,
            )
        )
        out = nxt
    learner.train_batch(replay.sample(4))
    save_state(tmp_path, learner, replay, rng, dict(steps=8, episodes=0))
    expected_actions = learner.act(out["obs"], out["action_mask"], 1, rng)
    expected_metrics = learner.train_batch(replay.sample(4))
    expected_weights = weights_hash(learner)
    fresh = BS5LaunchQMIX(env)
    fresh_rng = np.random.default_rng(0)
    _, restored = load_state(tmp_path / "resume.pt", fresh, fresh_rng)
    assert fresh.act(out["obs"], out["action_mask"], 1, fresh_rng) == expected_actions
    assert fresh.train_batch(restored.sample(4)) == expected_metrics
    assert weights_hash(fresh) == expected_weights


@pytest.fixture(scope="module")
def tiny_bundle(tmp_path_factory):
    root = tmp_path_factory.mktemp("bs5_tiny")
    identity = source_identity()
    freeze_source(root, identity)
    config = read_json(ROOT / "configs/bs5_launch_ab_v1.json")
    config.update(
        training_seeds=[949999],
        horizon=6,
        requested_steps_per_run=12,
        warmup=4,
        batch_size=4,
        train_scenario_seeds=[150001, 150004],
        validation_scenario_seeds=[160001, 160002],
    )
    protocol = make_protocol(config, identity)
    protocol["evaluation_steps"] = [0, 6, 12]
    atomic_json(root / "protocol.json", protocol)
    atomic_json(root / "schema.json", schema())
    atomic_json(root / "run_registry.json", registry_new(protocol))
    run_matrix(root, export=False)
    return root


def test_offline_bundle_rates_completeness_and_blocked_imports(tiny_bundle, tmp_path):
    result = analyze(tiny_bundle, tmp_path, require_complete=True)
    assert result["training_steps"] == 48
    assert result["validated_runs"] == 4
    assert result["learner_updates"] == 12
    assert (tmp_path / "learning_curves.svg").exists()
    assert offline_acceptance(tiny_bundle)["torch_env_imports_blocked"]


def test_validator_detects_missing_run_bad_rates_and_duplicate_resume_updates(tiny_bundle):
    root = tiny_bundle
    path = root / "run_registry.json"
    original = read_json(path)
    changed = deepcopy(original)
    changed["runs"].pop(next(iter(changed["runs"])))
    atomic_json(path, changed)
    try:
        with pytest.raises(AssertionError, match="missing/extra run"):
            validate(root)
    finally:
        atomic_json(path, original)
    packet_path = next((root / "runs").glob("*/evaluations/000000/*.json.gz"))
    packet = load_trace(packet_path)
    bad = deepcopy(packet)
    bad["trace"][0]["reward"] += 0.1
    write_trace(packet_path, bad)
    try:
        with pytest.raises(AssertionError, match="rate-reward"):
            validate(root)
    finally:
        write_trace(packet_path, packet)
    updates_path = next((root / "runs").glob("*/updates/000000.json"))
    updates = read_json(updates_path)
    atomic_json(updates_path, updates + [updates[0]])
    try:
        with pytest.raises(AssertionError, match="duplicate update"):
            validate(root)
    finally:
        atomic_json(updates_path, updates)


def test_interrupted_run_resumes_committed_boundary_without_duplicate_records(
    tiny_bundle, tmp_path, monkeypatch
):
    import scripts.training.run_bs5_launch_ab as runner

    protocol = read_json(tiny_bundle / "protocol.json")
    root = tmp_path
    atomic_json(root / "protocol.json", protocol)
    registry = registry_new(protocol)
    atomic_json(root / "run_registry.json", registry)
    real_evaluate = runner.evaluate

    def interrupt_at_end(learner, protocol, output, arm, seed, step):
        if step == 12:
            raise KeyboardInterrupt()
        return real_evaluate(learner, protocol, output, arm, seed, step)

    monkeypatch.setattr(runner, "evaluate", interrupt_at_end)
    with pytest.raises(KeyboardInterrupt):
        run_one(root, protocol, registry, "B-N", 949999, export=False)
    assert read_json(root / "runs/B-N_seed949999/commit.json")["steps"] == 6
    monkeypatch.setattr(runner, "evaluate", real_evaluate)
    run_one(root, protocol, registry, "B-N", 949999, resume=True, export=False)
    resumed = torch.load(root / "runs/B-N_seed949999/resume.pt", weights_only=False)
    expected = torch.load(tiny_bundle / "runs/B-N_seed949999/resume.pt", weights_only=False)
    for key in ("agent", "mixer", "target_agent", "target_mixer"):
        for name, tensor in expected[key].items():
            torch.testing.assert_close(resumed[key][name], tensor, rtol=0, atol=0)
    assert list((root / "runs/B-N_seed949999/aborted_attempts").rglob("000001.json"))
    assert registry["attempts"][0]["executed_train_steps"] == 12
    assert registry["attempts"][1]["executed_train_steps"] == 6
