import numpy as np

from scripts.evaluation.run_uav_operational_eval import (
    LocalAllAccess,
    counts_from_exact_snapshot,
    episode_start_snapshot,
    make_env,
)
from src.envs.v0_env import ACCESS


def test_executed_network_snapshot_matches_reward_after_mobility():
    e = make_env(3101, 3)
    o = e.reset(seed=3101)
    assert e.cfg.backhaul_radius_cells == 6.0
    assert e.cfg.access_radius_cells == 4.0
    e._capture_network = True
    actions = [int(np.flatnonzero(mask)[0]) for mask in o["action_mask"]]
    o = e.step(actions)
    snap = e.last_network_snapshot
    assert abs(o["reward"] - snap["utility"]) < 1e-10
    assert len(snap["ue_cells"]) == e.cfg.num_ues
    assert len(snap["ue_rate_mbps"]) == e.cfg.num_ues
    assert len(snap["uav_status"]) == e.cfg.num_uavs
    assert len(snap["battery_at_service"]) == e.cfg.num_uavs
    served, full, via_relay, paths = counts_from_exact_snapshot(snap, e.cfg.ue_demand_mbps)
    assert 0 <= via_relay <= served <= e.cfg.num_ues
    assert 0 <= full <= served
    assert 0 <= paths <= e.cfg.num_uavs


def test_local_5a_never_chooses_relay_and_respects_masks():
    e = make_env(3101, 4)
    o = e.reset(seed=3101)
    pol = LocalAllAccess(e)
    for _ in range(4):
        a = pol.act(o["action_mask"])
        for i, act in enumerate(a):
            assert o["action_mask"][i, act] == 1
            assert e.decode_action(act)[1] == ACCESS
        o = e.step(a)


def test_initial_snapshot_has_no_fabricated_links():
    e = make_env(3101, 3)
    e.reset(seed=3101)
    s = episode_start_snapshot(e)
    assert s["is_initial_state_without_action"]
    assert s["routes"] == {}
    assert s["association"] == {}
    assert sum(s["ue_rate_mbps"]) == 0.0
