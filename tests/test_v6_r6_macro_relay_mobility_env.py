import numpy as np

from src.envs.v0_env import ACCESS, RELAY
from src.envs.v6_r6_macro_relay_mobility_env import (
    MOVE_EAST,
    RELAY_BASE,
    STAY,
    V6R6MacroRelayMobilityConfig,
    V6R6MacroRelayMobilityEnv,
)


def make_env(horizon=30):
    return V6R6MacroRelayMobilityEnv(
        V6R6MacroRelayMobilityConfig(seed=1101, horizon=horizon)
    )


def test_v6_shapes_and_action_count():
    env = make_env()
    out = env.reset(seed=1101)
    assert env.n_actions == 9
    assert out["state"].shape == (105,)
    assert out["obs"].shape == (5, 110)
    assert out["action_mask"].shape == (5, 9)
    assert np.all(env.roles == ACCESS)


def test_relay_target_mapping_is_invertible():
    env = make_env()
    env.reset(seed=1101)
    for i in range(5):
        others = [j for j in range(5) if j != i]
        for slot, target in enumerate(others):
            a = RELAY_BASE + slot
            assert env.target_for_relay_action(i, a) == target
            assert env.relay_action_for_target(i, target) == a


def test_normal_movement_means_access_even_after_previous_relay():
    env = make_env()
    env.reset(seed=1101)
    env.roles[0] = RELAY
    env.relay_target[0] = 1
    env.step([STAY] * 5)
    assert env.roles[0] == ACCESS
    assert env.relay_target[0] == -1


def test_relay_macro_calls_assisted_move_and_is_one_slot(monkeypatch):
    env = make_env()
    env.reset(seed=1101)
    actor = 0
    targets = np.flatnonzero(env.request_open)
    assert len(targets) > 0
    target = int(next(t for t in targets if int(t) != actor))
    action = env.relay_action_for_target(actor, target)
    assert env.get_action_mask()[actor, action] == 1

    monkeypatch.setattr(env, "_best_relay_move", lambda rid, target_cell: MOVE_EAST)
    before = env.uav_positions[actor].copy()
    actions = [STAY] * 5
    actions[actor] = action
    out = env.step(actions)

    assert env.roles[actor] == RELAY
    assert env.relay_target[actor] == target
    assert out["info"]["accepted_macro_relays_this_slot"][0][:2] == [actor, target]
    assert out["info"]["accepted_macro_relays_this_slot"][0][2] == MOVE_EAST
    # Parent collision handling may block the candidate, but the macro move is
    # still recorded as the controller's chosen move.
    assert env.last_assisted_move[actor] == MOVE_EAST

    # Next normal movement immediately returns the UAV to Access: no Release action.
    env.step([STAY] * 5)
    assert env.roles[actor] == ACCESS
    assert env.relay_target[actor] == -1


def test_relay_move_controller_never_scores_worse_than_stay():
    env = make_env()
    env.reset(seed=1101)
    rid = 0
    target = 1
    target_cell = env.uav_positions[target].copy()
    move = env._best_relay_move(rid, target_cell)
    chosen = env.uav_positions[rid] + env_move_delta(move)
    stay = env.uav_positions[rid].copy()
    assert env._relay_candidate_score(target_cell, chosen) >= env._relay_candidate_score(target_cell, stay)


def env_move_delta(move):
    from src.envs.v0_env import MOVE_DELTAS
    return MOVE_DELTAS[int(move)]


def test_designated_relay_is_not_opportunistically_reused_by_other_accesses():
    env = make_env()
    env.reset(seed=1101)
    rid, target = 0, 1
    env.roles[:] = ACCESS
    env.roles[rid] = RELAY
    env.relay_target[:] = -1
    env.relay_target[rid] = target
    net = env.compute_network()
    routes = {int(k): v for k, v in net["routes"].items()}
    for aid, routed_relay in routes.items():
        if aid != target:
            assert routed_relay != rid


def test_duplicate_claim_has_single_winner():
    env = make_env()
    env.reset(seed=1101)
    env.request_open[:] = False
    env.relay_needed[:] = False
    target = 0
    env.request_open[target] = True
    env.relay_needed[target] = True

    actions = [STAY] * 5
    actions[1] = env.relay_action_for_target(1, target)
    actions[2] = env.relay_action_for_target(2, target)
    out = env.step(actions)
    accepted = out["info"]["accepted_macro_relays_this_slot"]
    rejected = out["info"]["rejected_macro_relay_claims_this_slot"]
    assert len(accepted) == 1
    assert accepted[0][0] == 1
    assert accepted[0][1] == target
    assert [2, target] in rejected
    assert env.roles[2] == ACCESS


def test_spatial_qmix_end_to_end_with_9_actions():
    from src.rl.v5_relay_request_spatial_qmix import V5RelayRequestSpatialQMIXLearner

    env = make_env()
    out = env.reset(seed=1101)
    learner = V5RelayRequestSpatialQMIXLearner(
        env.cfg.num_uavs,
        env.obs_dim,
        env.state_dim,
        env.n_actions,
        grid_size=env.cfg.grid_size,
        coarse_size=env.cfg.coarse_obs_size,
        device="cpu",
    )
    actions = learner.act(out["obs"], out["action_mask"], 0.0, np.random.default_rng(1))
    assert len(actions) == 5
    assert all(0 <= a < 9 for a in actions)
    nxt = env.step(actions)
    assert nxt["obs"].shape == (5, 110)
    assert nxt["action_mask"].shape == (5, 9)
