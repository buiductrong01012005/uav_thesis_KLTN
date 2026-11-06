from copy import deepcopy
import numpy as np
import pytest
from src.envs.bs5_energy_swap import BS5EnergySwapEnv, EnergyConfig, linear_learning_rate
from src.envs.bs5_multihop_train import BS5MultihopTrainEnv
from src.envs.bs5_service_v2 import FULL_CONDITION


def env(arm='C-R', **kwargs):
    e = BS5EnergySwapEnv(FULL_CONDITION, arm=arm, **kwargs)
    e.reset(seed=620001, assignment_seed=620001)
    e.set_battery_soc(100)
    return e


def stay(e, role=0):
    a = []
    mask = e.get_action_mask()
    for i in range(5):
        requested = e.encode_action(4,2 if e.multihop_arm=='H-only' else role)
        a.append(requested if mask[i,requested] else int(np.flatnonzero(mask[i])[0]))
    return a


@pytest.mark.parametrize('arm,n',[('C-N',10),('C-R',10),('C-H',15),('H-only',5)])
def test_shapes_reward_conservation_and_frozen(arm,n):
    e=env(arm,energy_config=EnergyConfig(energy_weight=.05))
    x=e.step(stay(e))
    assert x['obs'].shape==(5,148) and x['state'].shape==(66,)
    assert x['action_mask'].shape==(5,n)
    assert x['reward']==pytest.approx(x['info']['tcf_score']-.05*x['info']['energy_cost_norm'])
    d=e.last_energy_audit
    np.testing.assert_allclose(np.array(d['energy_before_j'])+d['inserted_j']-np.array(d['removed_j'])-d['consumed_j'],d['energy_after_j'])
    assert np.max(np.abs(d['balance_residual_j']))<1e-7
    e.condition['battery']=False
    before=e.energy_j.copy()
    x=e.step(stay(e))
    np.testing.assert_array_equal(before,e.energy_j)
    assert x['info']['reward_energy']==0


def test_swap_is_twelve_full_slots_after_arrival_and_full_pack():
    e=env('H-only')
    e.status[0]=1
    e.uav_positions[0]=e.charging_stations[0]
    e.target_station_idx[0]=0
    e.set_battery_soc([25,100,100,100,100])
    e.step(stay(e))
    assert e.status[0]==2 and e.swap_remaining[0]==12
    assert e.last_energy_audit['flight_j'][0]>0 # arrival still airborne
    for n in range(1,13):
        e.step(stay(e))
        assert e.last_service_snapshot['status'][0]==2
        assert e.last_energy_audit['consumed_j'][0]==0
        assert e.swap_remaining[0]==12-n
        assert e.status[0]==(0 if n==12 else 2)
    assert e.battery[0]==pytest.approx(100) and e.logical_roles()[0]==2
    assert e.total_consumed_j[0]>0 # removal never refunds consumed energy
    e.step(stay(e))
    assert e.last_service_snapshot['status'][0]==0


def test_blocked_move_pays_hover_not_cruise():
    e=env()
    e.uav_positions[:]=[[8,8],[8,9],[9,8],[10,10],[11,11]]
    a=stay(e)
    from src.envs.v0_env import MOVE_DELTAS
    move=next(i for i,d in enumerate(MOVE_DELTAS) if np.array_equal(d,[0,1]))
    a[0]=e.encode_action(move,0)
    e.step(a)
    assert e.last_energy_audit['distance_m'][0]==0
    assert e.last_energy_audit['flight_j'][0]==pytest.approx(e.energy_config.hover_w*5)


def test_five_hop_tx_receive_energy_and_service_unchanged():
    e=env('C-H')
    e.surviving_bs_xy=np.asarray([e._cell_to_xy([0,0])])
    e.uav_positions[:]=[[0,2],[0,8],[0,14],[6,14],[12,14]]
    e.ue_positions[:]=[12,14]
    # Keep mobility off so the fixture remains fixed; energy remains enabled.
    e.cfg.enable_ue_mobility=False
    old=BS5MultihopTrainEnv(FULL_CONDITION,arm='C-H')
    old.reset(seed=620001,assignment_seed=620001)
    old.surviving_bs_xy=e.surviving_bs_xy.copy()
    old.uav_positions=e.uav_positions.copy()
    old.ue_positions=e.ue_positions.copy()
    old.roles[:]=1; old._hybrid_ids=set(range(5))
    expected=old.compute_network()['ue_rate_mbps']
    e.step(stay(e,2))
    np.testing.assert_allclose(e.last_v2_network['ue_rate_mbps'],expected,atol=1e-9)
    a=e.last_energy_audit
    assert a['bh_tx_fraction'][4]==0 # last Access endpoint only receives BH
    assert a['bh_rx_fraction'][4]>0
    assert all(x>0 for x in a['bh_tx_fraction'][:4])
    assert a['backhaul_pa_j'][4]==0
    np.testing.assert_allclose(a['backhaul_pa_j'],np.array(a['bh_tx_fraction'])*.1/.25*5)
    assert np.max(np.array(a['bh_tx_fraction'])+a['bh_rx_fraction'])<=1+1e-6


def test_mobility_trace_preserved_across_arms_and_original():
    e=env('C-R'); h=env('H-only')
    old=BS5MultihopTrainEnv(FULL_CONDITION,arm='C-R')
    old.reset(seed=620001,assignment_seed=620001)
    for _ in range(20):
        e.step(stay(e)); h.step(stay(h)); old.step([8]*5)
        np.testing.assert_array_equal(e.ue_positions,h.ue_positions)
        np.testing.assert_array_equal(e.ue_positions,old.ue_positions)


def test_return_radio_off_and_reserve_reaches_depot():
    e=env()
    e.uav_positions[:]=[[5,5],[8,9],[9,8],[10,10],[11,11]]
    e.set_battery_soc([30,100,100,100,100])
    for _ in range(100):
        e.step(stay(e))
        if e.status[0]==2:
            assert e.battery[0]>=20-1e-9
            break
    else:
        pytest.fail('Return did not reach depot')
    assert all(e.last_energy_audit[k][0]==0 for k in ['circuit_j','access_pa_j','backhaul_pa_j'])
    assert e.energy_events[0]['event']=='return_start'


def test_authoritative_energy_validation():
    e=env(); e.battery[0]=4
    with pytest.raises(ValueError,match='set_battery_soc'):
        e.step(stay(e))


def test_lr_uses_global_progress_and_floor():
    assert linear_learning_rate(0,120000)==pytest.approx(3e-4)
    assert linear_learning_rate(60000,120000)==pytest.approx(1.65e-4)
    assert linear_learning_rate(120000,120000)==pytest.approx(3e-5)
    assert linear_learning_rate(300000,120000)==pytest.approx(3e-5)
    with pytest.raises(ValueError): linear_learning_rate(1,0)


@pytest.mark.parametrize('kwargs',[{'energy_weight':-1},{'capacity_wh':0},{'pa_efficiency':2}])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError): EnergyConfig(**kwargs)


def test_energy_weight_only_changes_reward_for_identical_actions():
    a=env(energy_config=EnergyConfig(energy_weight=0))
    b=env(energy_config=EnergyConfig(energy_weight=.05))
    for _ in range(10):
        x=a.step(stay(a));y=b.step(stay(b))
        np.testing.assert_array_equal(x['obs'],y['obs'])
        np.testing.assert_array_equal(a.energy_j,b.energy_j)
        np.testing.assert_array_equal(a.last_v2_network['ue_rate_mbps'],b.last_v2_network['ue_rate_mbps'])
        assert x['reward']-y['reward']==pytest.approx(.05*y['info']['energy_cost_norm'])


def test_no_instant_hidden_ue_census_in_actor_input():
    e=env('C-H')
    e.step(stay(e,2))
    obs=e.get_obs().copy();state=e.get_state().copy()
    e.ue_positions[:]=[0,0]
    np.testing.assert_array_equal(obs,e.get_obs())
    assert not np.array_equal(state,e.get_state())
