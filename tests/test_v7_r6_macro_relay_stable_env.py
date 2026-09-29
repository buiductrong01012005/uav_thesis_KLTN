import numpy as np

from src.envs.v0_env import ACCESS, RELAY
from src.envs.v7_r6_macro_relay_stable_env import (
    MOVE_EAST,
    RELAY_BASE,
    STAY,
    V7R6MacroRelayStableConfig,
    V7R6MacroRelayStableEnv,
)


def make_env(horizon=30):
    return V7R6MacroRelayStableEnv(
        V7R6MacroRelayStableConfig(seed=1101, horizon=horizon)
    )


def test_v7_shapes_and_action_count():
    env = make_env()
    out = env.reset(seed=1101)
    assert env.n_actions == 10
    assert out["state"].shape == (105,)
    assert out["obs"].shape == (5, 110)
    assert out["action_mask"].shape == (5, 10)
    assert np.all(env.roles == ACCESS)


def test_relay_action_semantics_are_global_target_stable():
    env = make_env()
    env.reset(seed=1101)
    for actor in range(5):
        for target in range(5):
            action = RELAY_BASE + target
            if target == actor:
                try:
                    env.target_for_relay_action(actor, action)
                except ValueError:
                    pass
                else:
                    raise AssertionError("RelayTo(self) must be invalid")
            else:
                assert env.target_for_relay_action(actor, action) == target
                assert env.relay_action_for_target(actor, target) == action


def test_relay_to_self_is_always_masked():
    env = make_env()
    env.reset(seed=1101)
    mask = env.get_action_mask()
    for actor in range(5):
        assert mask[actor, RELAY_BASE + actor] == 0


def test_normal_movement_means_access_even_after_previous_relay():
    env = make_env()
    env.reset(seed=1101)
    env.roles[0] = RELAY
    env.relay_target[0] = 1
    env.step([STAY] * 5)
    assert env.roles[0] == ACCESS
    assert env.relay_target[0] == -1


def test_relay_macro_uses_assisted_plan_and_is_one_slot(monkeypatch):
    env = make_env()
    env.reset(seed=1101)
    actor = 0
    targets = np.flatnonzero(env.request_open)
    assert len(targets) > 0
    target = int(next(t for t in targets if int(t) != actor))
    action = env.relay_action_for_target(actor, target)
    assert env.get_action_mask()[actor, action] == 1

    monkeypatch.setattr(env, "_best_relay_plan", lambda rid, target_cell: (MOVE_EAST, (1, 1.0, 0.0, 0.0)))
    actions = [STAY] * 5
    actions[actor] = action
    out = env.step(actions)

    assert env.roles[actor] == RELAY
    assert env.relay_target[actor] == target
    assert out["info"]["accepted_macro_relays_this_slot"][0][:2] == [actor, target]
    assert out["info"]["accepted_macro_relays_this_slot"][0][2] == MOVE_EAST
    assert env.last_assisted_move[actor] == MOVE_EAST

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


def test_duplicate_claim_winner_is_best_geometry_not_lowest_actor(monkeypatch):
    env = make_env()
    env.reset(seed=1101)
    env.request_open[:] = False
    env.relay_needed[:] = False
    target = 0
    env.request_open[target] = True
    env.relay_needed[target] = True

    # Actor 2 gets the better geometry score even though actor 1 has smaller id.
    def fake_plan(actor, target_cell):
        score = (1, float(actor), 0.0, 0.0)
        return STAY, score

    monkeypatch.setattr(env, "_best_relay_plan", fake_plan)
    actions = [STAY] * 5
    actions[1] = env.relay_action_for_target(1, target)
    actions[2] = env.relay_action_for_target(2, target)
    out = env.step(actions)
    accepted = out["info"]["accepted_macro_relays_this_slot"]
    rejected = out["info"]["rejected_macro_relay_claims_this_slot"]
    assert len(accepted) == 1
    assert accepted[0][0] == 2
    assert accepted[0][1] == target
    assert [1, target] in rejected
    assert env.roles[1] == ACCESS


def test_spatial_qmix_end_to_end_with_10_actions():
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
    assert all(0 <= a < 10 for a in actions)
    nxt = env.step(actions)
    assert nxt["obs"].shape == (5, 110)
    assert nxt["action_mask"].shape == (5, 10)
