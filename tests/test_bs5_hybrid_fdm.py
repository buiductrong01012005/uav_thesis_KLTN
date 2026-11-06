import numpy as np
import pytest

from src.diagnostics.bs5_hybrid_fdm import (
    HYBRID, HybridFDMConfig, access_fdm, associate_fdm, bh_capacity, edge_budget,
    overlap_rate, routes_fdm, schedules_fdm, score_hybrid_fdm,
)
from src.diagnostics.bs5_spectrum_airtime import (
    Edge, access_rates, build_routes, schedule_matrix, valid_half_duplex_state,
)
from src.envs.bs5_launch_ab_env import score_network
from src.envs.bs5_tcf_env import BS5TCFEnv


def fixture(distance_cells=2, local_users=10):
    env = BS5TCFEnv()
    env.surviving_bs_xy = np.asarray([[125., 125.]])
    env.uav_positions[:] = [[2, 2+distance_cells], [2, 8+distance_cells], [19, 0], [18, 0], [17, 0]]
    env.status[:] = 3
    env.status[:2] = 0
    env.roles[:] = [1, 0, 0, 0, 0]
    env.ue_positions[:] = [19., 0.]
    env.ue_positions[:local_users] = env.uav_positions[0]
    env.ue_positions[local_users:local_users+25] = env.uav_positions[1]
    return env


def test_partial_overlap_integrated_without_mean_interference():
    # One half has SINR1, the other SINR10; do not average interferer first.
    value = overlap_rate(10e6, 10., [(5e6, 9.)], 1.)
    expected = 5*np.log2(2)+5*np.log2(11)
    assert value == pytest.approx(expected)
    assert value > 10*np.log2(1+10/5.5)


@pytest.mark.parametrize("alpha", [-1, 0, 1, float("nan")])
def test_invalid_split_rejected(alpha):
    with pytest.raises(ValueError):
        HybridFDMConfig(alpha)


def test_actual_endpoint_bandwidth_and_power_budget():
    env = fixture()
    roles = [2, 0, 0, 0, 0]
    config = HybridFDMConfig()
    assert edge_budget(env, "b0", "u0", roles, config) == pytest.approx((45e6, 100))
    assert edge_budget(env, "u0", "u1", roles, config) == pytest.approx((45e6, 50))
    assert edge_budget(env, "b0", "u1", roles, config) == pytest.approx((90e6, 100))


def test_no_hybrid_scoring_matches_historical_and_does_not_mutate():
    env = fixture()
    before = {k:getattr(env,k).copy() for k in ['roles','status','battery','uav_positions','ue_positions']}
    old = env.compute_network()
    new = score_hybrid_fdm(env, env.roles)
    np.testing.assert_allclose(new['ue_rate_mbps'], old['ue_rate_mbps'], atol=1e-8, rtol=0)
    assert new['association'] == old['association']
    for k,v in before.items():
        np.testing.assert_array_equal(getattr(env,k),v)


def test_general_fdm_equations_reduce_to_old_access_and_bh_when_no_hybrid():
    env = fixture()
    roles = env.roles.copy()
    association = score_network(env, allow_relay=False)['association']
    assert associate_fdm(env,roles,HybridFDMConfig()) == association
    old,_,_ = access_rates(env,association,18,'colored_reuse',1)
    new,_ = access_fdm(env,association,roles,HybridFDMConfig())
    np.testing.assert_allclose(new,old,atol=1e-12,rtol=0)
    e1,p1 = build_routes(env,association,90,1)
    e2,p2 = routes_fdm(env,association,roles,HybridFDMConfig())
    assert p1 == p2
    assert [(e.source,e.target) for e in e1] == [(e.source,e.target) for e in e2]
    m1,s1 = schedule_matrix(env,e1,90,1,'shared_half_duplex')
    m2,s2 = schedules_fdm(env,e2,roles,HybridFDMConfig())
    assert s1 == s2
    np.testing.assert_allclose(m1,m2,atol=1e-10,rtol=0)


def test_hybrid_can_serve_locally_and_forward_without_extra_received_data():
    env = fixture()
    roles = np.asarray([HYBRID,0,0,0,0])
    pure = score_hybrid_fdm(env, env.roles)
    h = score_hybrid_fdm(env,roles)
    m = h['metrics']
    assert m['own_hybrid_mbps'] > 0 and m['hybrid_forwarded_mbps'] > 0
    assert m['throughput_mbps'] > pure['metrics']['throughput_mbps']
    assert h['paths'][0] and len(h['paths'][0]) == 1
    incoming = sum(h['edge_load_mbps'][i] for i,e in enumerate(h['edges']) if e.target == 'u0')
    outgoing = sum(h['edge_load_mbps'][i] for i,e in enumerate(h['edges']) if e.source == 'u0')
    assert incoming == pytest.approx(h['flows'][0]+outgoing)
    assert np.all(h['edge_load_mbps'] <= h['edge_capacity_mbps']+1e-6)
    assert sum(h['durations']) <= 1+1e-6
    assert all(valid_half_duplex_state(h['edges'],s) for s in h['states'])
    p = h['power_bounds'][0]
    assert p['access_tx_mw'] == pytest.approx(10**2.7/2)
    assert p['bh_tx_limit_mw'] == pytest.approx(50)
    assert p['combined_tx_bound_mw'] < 10**2.7


def test_backhaul_bottleneck_hybrid_is_not_guaranteed_to_forward_or_improve():
    env = fixture(distance_cells=6,local_users=25)
    pure = score_hybrid_fdm(env,env.roles)
    h = score_hybrid_fdm(env,[2,0,0,0,0])
    assert h['metrics']['own_hybrid_mbps'] > 0
    assert h['metrics']['hybrid_forwarded_mbps'] < 1e-6
    assert h['metrics']['throughput_mbps'] < pure['metrics']['throughput_mbps']


def test_incoming_band_negotiated_and_interference_applies_to_narrower_band():
    env = fixture()
    env.uav_positions[2] = [3,8]
    env.uav_positions[3] = [3,12]
    roles = [2,0,1,0,0]
    edges = [Edge('b0','u0',0), Edge('u2','u3',0)]
    solo = bh_capacity(env,edges,(0,),0,roles,HybridFDMConfig())
    together = bh_capacity(env,edges,(0,1),0,roles,HybridFDMConfig())
    assert together < solo


def test_unreachable_hybrid_cannot_bootstrap_an_extra_hop():
    env = fixture(distance_cells=6,local_users=10)
    h = score_hybrid_fdm(env,[1,2,0,0,0])
    assert h['paths'].get(1) == []
    assert h['metrics']['throughput_mbps'] == 0


def test_same_color_partial_access_interference_and_legacy_reduction():
    env = fixture()
    env.uav_positions[:2] = [[0,0],[19,19]]
    env.roles[:2] = 0
    env.ue_positions[:] = [10,10]
    env.ue_positions[0] = [0,0]
    env.ue_positions[1] = [19,19]
    association = {0:[0],1:[1]}
    config = HybridFDMConfig()
    old,_,_ = access_rates(env,association,18,'colored_reuse',1)
    new,_ = access_fdm(env,association,np.zeros(5,dtype=int),config)
    np.testing.assert_allclose(new,old,atol=1e-12,rtol=0)
    changed,_ = access_fdm(env,association,np.asarray([0,2,0,0,0]),config)
    assert np.isfinite(changed).all() and np.all(changed <= 4)


def test_hybrid_association_accounts_for_reduced_power():
    env = fixture()
    env.uav_positions[:2] = [[5,5],[5,7]]
    env.roles[:2] = 0
    env.ue_positions[:] = [19,0]
    env.ue_positions[0] = [5,6]
    association = associate_fdm(env,[2,0,0,0,0],HybridFDMConfig())
    assert association[1] == [0] and association[0] == []
