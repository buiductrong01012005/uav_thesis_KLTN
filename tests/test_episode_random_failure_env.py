from copy import deepcopy
import pickle

import numpy as np
import pytest

from src.envs.episode_random_failure_env import (
    EpisodeFailureCartesianConfig,
    EpisodeFailureCartesianEnv,
    EpisodeFailureV9Config,
    EpisodeFailureV9Env,
    EpisodeFailureMovementEnv,
)
from src.envs.thesis_cartesian_r6_curriculum_env import (
    CartesianR6BatteryConfig,
    CartesianR6BatteryEnv,
)
from src.envs.v9_r6_macro_relay_battery_focus_env import (
    V9R6MacroRelayBatteryFocusConfig,
    V9R6MacroRelayBatteryFocusEnv,
)
from src.envs.final_movement_only_env import FinalMovementOnlyBatteryFocusEnv


KINDS = [
    (
        EpisodeFailureCartesianEnv,
        EpisodeFailureCartesianConfig,
        CartesianR6BatteryEnv,
        CartesianR6BatteryConfig,
    ),
    (
        EpisodeFailureV9Env,
        EpisodeFailureV9Config,
        V9R6MacroRelayBatteryFocusEnv,
        V9R6MacroRelayBatteryFocusConfig,
    ),
    (
        EpisodeFailureMovementEnv,
        EpisodeFailureV9Config,
        FinalMovementOnlyBatteryFocusEnv,
        V9R6MacroRelayBatteryFocusConfig,
    ),
]


def assert_same_physics(a, b):
    for name in (
        "failed_bs_xy",
        "failed_bs_cell",
        "surviving_bs_xy",
        "surviving_bs_cells",
        "affected_mask",
        "affected_cells",
        "ue_positions",
        "hotspot_centers",
        "hotspot_vel",
        "ue_offsets",
        "ue_local_velocity_cells_per_slot",
        "uav_positions",
        "roles",
        "status",
        "battery",
        "relay_target",
        "return_start_time",
        "station_arrival_time",
        "rejoin_time",
        "request_open",
        "request_load_mbps",
        "request_deficit",
    ):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name), err_msg=name)
    assert a.rng.bit_generator.state == b.rng.bit_generator.state
    an, bn = a.compute_network(), b.compute_network()
    np.testing.assert_array_equal(an["ue_rate_mbps"], bn["ue_rate_mbps"])
    assert an["routes"] == bn["routes"]
    assert an["association"] == bn["association"]
    assert an["backhaul_caps_mbps"] == bn["backhaul_caps_mbps"]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("bs", range(6))
def test_forced_random_matches_legacy_fixed_physics(kind, bs):
    ec, cc, oldec, oldcc = kind
    env = ec(cc(failure_mode="random", record_service_snapshot=True))
    old = oldec(oldcc(failed_bs_idx=bs))
    a, b = env.reset(seed=7021, options={"failed_bs_idx": bs}), old.reset(seed=7021)
    assert a["state"].shape == (111,)
    np.testing.assert_array_equal(a["state"][:105], b["state"])
    np.testing.assert_array_equal(a["obs"][:, 111:], b["obs"][:, 105:])
    assert_same_physics(env, old)
    rng = np.random.default_rng(17)
    for _ in range(12):
        mask = a["action_mask"]
        actions = [int(rng.choice(np.flatnonzero(m))) for m in mask]
        a, b = env.step(actions), old.step(actions)
        assert a["reward"] == b["reward"]
        np.testing.assert_array_equal(a["action_mask"], b["action_mask"])
        assert env.last_service_snapshot["network"]["utility"] == a["reward"]
        assert_same_physics(env, old)


def test_legacy_config_only_mutation_reproduces_stale_topology_bug():
    old = CartesianR6BatteryEnv(CartesianR6BatteryConfig(failed_bs_idx=3))
    old.cfg.failed_bs_idx = 5
    out = old.reset(seed=71)
    assert out["episode_meta"]["failed_bs_idx"] == 5
    np.testing.assert_array_equal(old.failed_bs_xy, old.bs_xy[3])
    assert any(np.array_equal(x, old.bs_xy[5]) for x in old.surviving_bs_xy)
    assert not any(np.array_equal(x, old.bs_xy[3]) for x in old.surviving_bs_xy)


def test_repeated_reset_rebuilds_every_cache_and_preserves_input_config():
    cfg = EpisodeFailureCartesianConfig(failure_mode="random", failed_bs_idx=3)
    original = deepcopy(cfg)
    env = EpisodeFailureCartesianEnv(cfg)
    for bs in [3, 5, 0, 2, 4, 1, 5, 3]:
        out = env.reset(seed=71, options={"failed_bs_idx": bs})
        fresh = CartesianR6BatteryEnv(CartesianR6BatteryConfig(failed_bs_idx=bs))
        fresh.reset(seed=71)
        assert_same_physics(env, fresh)
        assert out["episode_meta"]["failed_bs_idx"] == bs
        assert bs not in out["episode_meta"]["surviving_bs_ids"]
        assert out["state"][105:111].argmax() == bs
    assert cfg == original


def test_random_seeded_and_unseeded_failure_sequences_are_reproducible_and_independent():
    a = EpisodeFailureCartesianEnv(EpisodeFailureCartesianConfig(failure_mode="random"))
    b = EpisodeFailureCartesianEnv(EpisodeFailureCartesianConfig(failure_mode="random"))
    seq = []
    for k in range(60):
        a.reset(seed=73 if k == 0 else None)
        b.reset(seed=73 if k == 0 else None)
        seq.append(a.cfg.failed_bs_idx)
        assert a.cfg.failed_bs_idx == b.cfg.failed_bs_idx
        for _ in range(k % 3):
            a.step([8] * 5)
    assert set(seq) == set(range(6))
    a.reset(seed=73)
    assert a.cfg.failed_bs_idx == seq[0]


def test_random_reset_does_not_consume_ue_or_battery_rng():
    env = EpisodeFailureCartesianEnv(EpisodeFailureCartesianConfig(failure_mode="random"))
    for seed in range(30):
        env.reset(seed=seed)
        fixed = CartesianR6BatteryEnv(CartesianR6BatteryConfig(failed_bs_idx=env.cfg.failed_bs_idx))
        fixed.reset(seed=seed)
        assert_same_physics(env, fixed)


@pytest.mark.parametrize("bad", [-1, 6, 2.0, True, "3", None])
def test_invalid_explicit_failure_rejected_before_mutation(bad):
    env = EpisodeFailureCartesianEnv()
    env.reset(seed=7)
    before = pickle.dumps(env.__dict__)
    with pytest.raises(ValueError):
        env.reset(seed=9, options={"failed_bs_idx": bad})
    assert before == pickle.dumps(env.__dict__)


@pytest.mark.parametrize(
    "kw",
    [
        {"failed_bs_idx": -1},
        {"failed_bs_idx": 6},
        {"failure_mode": "oops"},
        {"failure_candidates": ()},
        {"failure_candidates": (3, 3)},
        {"failure_candidates": (0, 6)},
        {"failure_candidates": (True,)},
    ],
)
def test_invalid_configuration_rejected(kw):
    with pytest.raises(ValueError):
        EpisodeFailureCartesianEnv(EpisodeFailureCartesianConfig(**kw))


def test_subset_selection_and_explicit_override_restore_fixed_configuration():
    env = EpisodeFailureCartesianEnv(
        EpisodeFailureCartesianConfig(
            failure_mode="random",
            failure_candidates=(2, 5),
            failed_bs_idx=3,
        )
    )
    seen = set()
    for s in range(30):
        env.reset(seed=s)
        seen.add(env.cfg.failed_bs_idx)
    assert seen == {2, 5}
    env.reset(seed=1, options={"failed_bs_idx": 3})
    assert env.cfg.failed_bs_idx == 3
    fixed = EpisodeFailureCartesianEnv(EpisodeFailureCartesianConfig(failed_bs_idx=5))
    fixed.reset(seed=1, options={"failed_bs_idx": 0})
    fixed.reset(seed=1)
    assert fixed.cfg.failed_bs_idx == 5
    with pytest.raises(ValueError):
        fixed.reset(options={"failure_bs": 4})


@pytest.mark.parametrize("bs", range(6))
def test_failure_fixed_within_episode_and_ues_confined_under_mobility(bs):
    env = EpisodeFailureCartesianEnv(EpisodeFailureCartesianConfig(failed_bs_idx=bs))
    env.reset(seed=91)
    for _ in range(30):
        assert env.cfg.failed_bs_idx == bs
        assert all(env._ue_position_is_valid(p) for p in env.ue_positions)
        env.step([8] * 5)


def test_recording_is_observational_and_snapshot_precedes_mobility_and_energy():
    a = EpisodeFailureCartesianEnv(EpisodeFailureCartesianConfig(record_service_snapshot=True))
    b = EpisodeFailureCartesianEnv()
    a.reset(seed=5)
    b.reset(seed=5)
    initial_ue, initial_battery = a.ue_positions.copy(), a.battery.copy()
    x, y = a.step([8] * 5), b.step([8] * 5)
    assert x["reward"] == y["reward"]
    assert_same_physics(a, b)
    snap = a.last_service_snapshot
    np.testing.assert_array_equal(snap["ue_positions"], initial_ue)
    np.testing.assert_array_equal(snap["battery"], initial_battery)
    assert not np.array_equal(a.ue_positions, initial_ue)
    assert not np.array_equal(a.battery, initial_battery)
    snap["ue_positions"][:] = -999
    assert a.ue_positions.min() >= 0


def test_failure_context_changes_only_context_when_physical_features_identical():
    env = EpisodeFailureCartesianEnv()
    env.reset(seed=9)
    before = env.get_state().copy()
    env.cfg.failed_bs_idx = 5  # encoding-only unit test, not a valid episode update
    after = env.get_state()
    np.testing.assert_array_equal(before[:105], after[:105])
    assert before[105:111].argmax() == 3
    assert after[105:111].argmax() == 5


def test_valid_coarse_observation_alias_has_explicit_failure_resolution():
    # Synthetic valid states, not samples from the default hotspot generator.
    # Both UE cells belong to coarse cell (1,1), with no Access within radius.
    starts = ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4))
    states = []
    for bs, ue_cell in [(0, (4, 5)), (3, (6, 4))]:
        env = EpisodeFailureCartesianEnv(
            EpisodeFailureCartesianConfig(
                failed_bs_idx=bs,
                uav_start_cells=starts,
            )
        )
        env.reset(seed=73)
        assert env.affected_mask[ue_cell]
        env.ue_positions[:] = ue_cell
        env._refresh_request_state()
        states.append(env.get_state())
    np.testing.assert_array_equal(states[0][:105], states[1][:105])
    assert not np.array_equal(states[0][105:], states[1][105:])


def test_failure_aware_network_optimizer_and_checkpoint_roundtrip(tmp_path):
    import torch
    from src.rl.episode_failure_spatial_qmix import EpisodeFailureSpatialQMIX
    from src.rl.v5_relay_request_spatial_qmix import V5RelayRequestSpatialQMIXLearner
    from src.rl.qmix import ReplayBuffer, Transition

    torch.set_num_threads(1)
    env = EpisodeFailureCartesianEnv(EpisodeFailureCartesianConfig(failure_mode="random"))
    out = env.reset(seed=1)
    with pytest.raises(ValueError):
        V5RelayRequestSpatialQMIXLearner(5, env.obs_dim, env.state_dim, 10)
    learner = EpisodeFailureSpatialQMIX(env)
    buffer = ReplayBuffer(64)
    rng = np.random.default_rng(0)
    for k in range(12):
        if k % 4 == 0:
            out = env.reset(seed=k)
        actions = learner.act(out["obs"], out["action_mask"], 0.5, rng)
        nxt = env.step(actions)
        buffer.add(
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
    assert np.isfinite(learner.train_batch(buffer.sample(8)))
    obs = torch.as_tensor(out["obs"])
    with torch.no_grad():
        old = learner.agent(obs).clone()
    path = tmp_path / "new.pt"
    learner.save(str(path))
    fresh = EpisodeFailureSpatialQMIX(env)
    fresh.load(str(path))
    torch.testing.assert_close(fresh.agent(obs), old, rtol=0, atol=0)
    # There must be a trainable path from failure context to agent Q values.
    obs.requires_grad_(True)
    fresh.agent(obs).sum().backward()
    assert obs.grad[:, 105:111].abs().sum() > 0


def test_movement_and_cartesian_access_actions_have_same_physics():
    a = EpisodeFailureCartesianEnv(EpisodeFailureCartesianConfig(failure_mode="random"))
    b = EpisodeFailureMovementEnv(EpisodeFailureV9Config(failure_mode="random"))
    rng = np.random.default_rng(27)
    for seed in range(6):
        a.reset(seed=seed)
        out = b.reset(seed=seed)
        for _ in range(10):
            moves = [int(rng.choice(np.flatnonzero(m))) for m in out["action_mask"]]
            x, out = a.step([2 * m for m in moves]), b.step(moves)
            assert x["reward"] == out["reward"]
            assert_same_physics(a, b)


def test_legacy_bs5_launch_has_no_initial_five_step_service_signal():
    env = EpisodeFailureCartesianEnv(EpisodeFailureCartesianConfig(failed_bs_idx=5))
    rng = np.random.default_rng(990)
    for seed in range(12):
        out = env.reset(seed=seed)
        for _ in range(5):
            actions = [int(rng.choice(np.flatnonzero(m))) for m in out["action_mask"]]
            out = env.step(actions)
            assert out["reward"] == 0.0
