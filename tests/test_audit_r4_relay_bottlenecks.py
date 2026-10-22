import numpy as np
import pytest

from scripts.audit_r4_relay_bottlenecks import (
    audit_oracle, radio_offer, radio_with_roles_exact, slot_metrics,
)
from scripts.eval_relay_stress_r4_qmix import make_env, direct_access_cells
from src.envs.v0_env import ACCESS, RELAY
from src.envs.v4_fixed_mobile_battery_env import ACTIVE


def network_snapshot(seed=3106):
    env = make_env(seed, 4)
    env.reset(seed=seed)
    env.roles[:]=[ACCESS, RELAY, ACCESS, RELAY, ACCESS]
    env.last_network = None
    net = env.compute_network()
    return env, env.last_network


def test_radio_offer_and_bh_reconstruction_exact():
    env, snap = network_snapshot()
    metrics, offers, hops = slot_metrics(env, snap, direct_access_cells(env))
    reconstructed = sum(min(offers[aid],snap['backhaul_caps_mbps'].get(aid,0))
                        for aid in snap['association'])
    assert np.isclose(reconstructed, np.sum(snap['ue_rate_mbps']),atol=1e-7)
    assert np.isclose(metrics['direct_bs_lost_mbps'] + metrics['access_to_relay_lost_mbps'] + metrics['relay_to_bs_lost_mbps'], metrics['backhaul_reduction_mbps'],atol=1e-6)
    assert metrics['required_served'] <= metrics['required_associated'] <= metrics['required_covered'] <= metrics['required_ues']
    assert np.isclose(metrics['utility'], np.sum(snap['ue_rate_mbps'])/(
        env.cfg.ue_demand_mbps*env.cfg.num_ues))


def test_role_oracle_restores_complete_simulation_state():
    env, snap = network_snapshot(3102)
    attrs = ('uav_positions','ue_positions','roles','status')
    saved = {k:getattr(env,k).copy() for k in attrs}
    initial = float(snap['utility'])
    result = audit_oracle(env,snap)
    assert np.isclose(result['oracle_actual_utility'],initial,atol=1e-10)
    assert result['oracle_best_role_utility'] + 1e-10 >= initial
    assert result['oracle_best_role_utility'] + 1e-10 >= result['oracle_all_access_utility']
    for k in attrs:
        assert np.array_equal(saved[k],getattr(env,k)),k


def test_replay_snapshot_is_before_ue_mobility():
    env = make_env(3117,3)
    out=env.reset(seed=3117)
    before=env.ue_positions.copy()
    out=env.step([env.encode_action(4,ACCESS)]*env.cfg.num_uavs)
    snap=env.last_network
    assert snap is not None
    # The environment's current user positions are next slot; scorer must use captured positions.
    assert np.array_equal(snap['ue_positions'],before)
    assert np.max(np.abs(snap['ue_positions']-env.ue_positions)) > 0
    assert np.isclose(float(out['reward']),float(snap['utility']))
    cf=radio_with_roles_exact(env,snap,snap['roles'])
    assert np.isclose(cf['utility'],snap['utility'],atol=1e-10)
