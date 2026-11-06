import numpy as np
import pytest
from src.envs.bs5_energy_collision import BS5EnergyCollisionEnv, CollisionConfig
from src.envs.bs5_energy_swap import BS5EnergySwapEnv, EnergyConfig
from src.envs.v0_env import MOVE_DELTAS


def make(weight=.1,arm='C-R'):
    e=BS5EnergyCollisionEnv(arm=arm,collision_config=CollisionConfig(weight),energy_config=EnergyConfig(energy_weight=.05))
    e.reset(seed=620001,assignment_seed=620001);e.set_battery_soc(100)
    return e


def actions(e,moves=None,roles=None):
    moves=moves or [[0,0]]*5;roles=roles or [0]*5
    m=e.get_action_mask();out=[]
    for i in range(5):
        move=next(j for j,d in enumerate(MOVE_DELTAS) if np.array_equal(d,moves[i]))
        role=2 if e.multihop_arm=='H-only' else roles[i]
        a=e.encode_action(move,role)
        out.append(a if m[i,a] else int(np.flatnonzero(m[i])[0]))
    return out


def test_two_policy_moves_to_same_empty_cell_penalty_once():
    e=make();e.uav_positions[:]=[[8,7],[8,9],[10,3],[10,5],[10,7]]
    x=e.step(actions(e,[[0,1],[0,-1],[0,0],[0,0],[0,0]]))
    assert x['info']['active_blocked_uavs']==1
    assert x['info']['policy_move_requests']==2
    assert x['info']['reward_collision']==pytest.approx(-.02)
    assert x['reward']==pytest.approx(x['info']['tcf_score']+x['info']['reward_energy']-.02)
    assert len(set(map(tuple,e.uav_positions.tolist())))==5
    assert len(e.collision_events)==1


def test_autonomous_return_rejection_logged_without_actor_penalty():
    e=make();e.charging_stations[:]=[[0,8],[0,0],[0,19]]
    e.uav_positions[:]=[[8,7],[8,9],[7,7],[9,7],[7,9]]
    e.status[:2]=1;e.target_station_idx[:2]=0
    x=e.step(actions(e))
    assert x['info']['returning_blocked_uavs']==1
    assert x['info']['active_blocked_uavs']==0
    assert x['info']['reward_collision']==0


def test_shared_hub_and_allowed_swap_are_not_penalized():
    e=make();hub=e.charging_stations[0].copy()
    e.uav_positions[:]=[hub+[0,1],hub+[1,0],[8,8],[8,9],[10,10]]
    x=e.step(actions(e,[[0,-1],[-1,0],[0,1],[0,-1],[0,0]]))
    assert x['info']['active_blocked_uavs']==0
    assert x['info']['reward_collision']==0
    np.testing.assert_array_equal(e.uav_positions[0],hub)
    np.testing.assert_array_equal(e.uav_positions[1],hub)
    assert e.uav_positions[2].tolist()==[8,9] and e.uav_positions[3].tolist()==[8,8]


def test_energy_override_is_not_a_policy_collision():
    e=make();e.uav_positions[:]=[[5,5],[8,8],[8,9],[10,10],[11,11]]
    e.set_battery_soc([20,100,100,100,100])
    x=e.step(actions(e))
    assert e.last_collision_audit['effective_status'][0]==1
    assert x['info']['active_blocked_uavs']==0


@pytest.mark.parametrize('arm',['C-N','C-R','C-H','H-only'])
def test_zero_collision_weight_preserves_all_v1_dynamics_and_reward(arm):
    a=make(0.,arm)
    b=BS5EnergySwapEnv(arm=arm,energy_config=EnergyConfig(energy_weight=.05))
    b.reset(seed=620001,assignment_seed=620001);b.set_battery_soc(100)
    rng=np.random.default_rng(15)
    for _ in range(30):
        m=a.get_action_mask();acts=[int(rng.choice(np.flatnonzero(row))) for row in m]
        x=a.step(acts);y=b.step(acts)
        assert x['reward']==y['reward']
        for key in ['obs','state','action_mask']:np.testing.assert_array_equal(x[key],y[key])
        for key in ['energy_j','status','uav_positions','ue_positions']:np.testing.assert_array_equal(getattr(a,key),getattr(b,key))
        np.testing.assert_array_equal(a.last_v2_network['ue_rate_mbps'],b.last_v2_network['ue_rate_mbps'])


@pytest.mark.parametrize('weight',[float('nan'),float('inf'),-.1])
def test_invalid_weight(weight):
    with pytest.raises(ValueError):CollisionConfig(weight)
