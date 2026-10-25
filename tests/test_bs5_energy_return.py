"""Physical Return regressions, including the audited validation snapshot.

These are fixed-action fixture controls, not learned-policy performance tests.
"""
from copy import deepcopy
from dataclasses import asdict
import numpy as np
import pytest
from src.envs.bs5_energy_return_env import BS5EnergyReturnEnv,shortest_return_path
from src.envs.bs5_collision_safe_env import BS5CollisionSafeCurriculumEnv,assert_endpoint_occupancy
from scripts.analysis.bs5_return_audit import route,validate_return

CONDITION=dict(stage='S4',substage='S4',active_count=5,mobility=True,battery=True,
    starts='B',radius=None,jitter=0,stay_only=False)
STATIONS=((1,1),(18,1),(10,18))
POSITIONS=[[15,14],[14,11],[15,13],[14,13],[15,16]]

def env(cls=BS5EnergyReturnEnv):
    e=cls(CONDITION,horizon=1200)
    e.reset(seed=140001,assignment_seed=140001,diagnostic=True)
    return e

def snapshot(cls=BS5EnergyReturnEnv):
    # Prior raw scenario130009/slot120; no historical UE/RNG replay implied.
    e=env(cls);e.uav_positions[:]=POSITIONS;e.status[:]=[0,0,1,0,0]
    e.battery[:]=[63.69965362548828,64.19966888427734,5.599963188171387,59.399742126464844,62.59980010986328]
    e.roles[:]=[0,1,1,0,0];e.target_station_idx[:]=[-1,-1,2,-1,-1]
    return e

def decision(e):
    return {k:getattr(e,attr).tolist() for k,attr in [('positions','uav_positions'),('status','status'),('roles','roles'),('battery','battery')]}

def check_raw(e,before,actions):
    d=e.last_return_control
    r=dict(decision=before,post=decision(e),raw_actions=actions,
        service={k:e.last_service_snapshot[k].tolist() for k in ['status','roles','uav_positions']},
        task_condition=e.condition,return_control=d,collision=e.last_collision_resolution,
        charging_station_cells=e.charging_stations.tolist(),
        applied_actions=[dict(autonomous=before['status'][a]!=0 or a in d['forced_return_ids']) for a in range(5)])
    validate_return(r,asdict(e.cfg))
    return r

def test_independent_reverse_route_matches_bfs_and_ties():
    rng=np.random.default_rng(30903)
    for _ in range(300):
        cells=[tuple(map(int,p)) for p in rng.integers(0,20,size=(5,2))]
        start=cells[0];blocked=tuple(sorted(set(cells[1:])))
        a=shortest_return_path(20,STATIONS,blocked,start)
        idx,steps,path,_=route(20,STATIONS,blocked,start)
        assert (None if a is None else (a[0],len(a[1])-1,[list(p) for p in a[1]]))==(None if steps is None else (idx,steps,path))
    assert shortest_return_path(5,((0,2),(2,0)),(),(2,2))==(0,((2,2),(1,2),(0,2)))

def test_real_snapshot_has_energy_feasible_detour():
    e=snapshot();plan=e._plan(e.uav_positions[2],2)
    assert plan['station']==2 and plan['steps']==12
    assert plan['energy_for_path']==pytest.approx(2.4)
    assert plan['path'][1]==[16,13] and e.battery[2]>plan['energy_for_path']

def test_frozen_blocker_control_arrives_while_legacy_waits_to_zero(tmp_path):
    a=snapshot();b=snapshot(BS5CollisionSafeCurriculumEnv)
    arrival=None;rejoin=None;history=[]
    for t in range(1,41):
        before=decision(a)
        a.step([8]*5);b.step([8]*5)
        row=check_raw(a,before,[8]*5)
        history.append(dict(slot=t,new_position=a.uav_positions[2].tolist(),old_position=b.uav_positions[2].tolist(),
            new_battery=float(a.battery[2]),old_battery=float(b.battery[2]),new_status=int(a.status[2]),old_status=int(b.status[2])))
        if t==1:
            bad=deepcopy(row);bad['decision']['battery'][2]=0
            with pytest.raises(AssertionError):validate_return(bad,asdict(a.cfg))
        if arrival is None and a.status[2]==2:arrival=t
        if arrival is not None and rejoin is None and a.status[2]==0:rejoin=t
        assert_endpoint_occupancy(a.uav_positions,a.status,a.charging_stations)
    assert arrival==12 and rejoin==28
    assert a.battery[2]>0 and a.status[2]==0
    assert b.status[2]==1 and b.battery[2]==0 and b.uav_positions[2].tolist()==[15,13]
    import json
    (tmp_path/'blocked_snapshot_control.json').write_text(json.dumps(dict(
        scope='frozen other UAV positions, Stay-Access actions; not learned-policy performance',
        arrival_new=arrival,rejoin_new=rejoin,steps_per_env=40,history=history),indent=2))

@pytest.mark.parametrize('battery',[0.,.1,.19])
def test_unfunded_return_never_moves_or_serves(battery):
    e=env();e.uav_positions[0]=[10,16];e.status[:]=3;e.status[0]=1
    e.battery[0]=battery;e.target_station_idx[0]=2
    before=decision(e);out=e.step([8]*5);check_raw(e,before,[8]*5)
    assert e.uav_positions[0].tolist()==[10,16] and e.battery[0]==0
    assert out['reward']==0 and e.last_return_control['stranded_ids_post']==[0]
    assert e.last_return_control['prevented_energy_move_ids']==[0]

def test_active_shortfall_switches_before_service_and_uses_return_cost():
    e=env();e.uav_positions[0]=[10,17];e.status[:]=3;e.status[0]=0;e.battery[0]=.2
    before=decision(e);out=e.step([8]*5);check_raw(e,before,[8]*5) # Stay-Access costs .3, Return costs .2.
    assert e.last_return_control['forced_return_ids']==[0]
    assert e.status[0]==2 and e.uav_positions[0].tolist()==[10,18]
    assert e.battery[0]==0 and out['reward']==0

def test_return_trigger_uses_longer_feasible_path_not_manhattan():
    e=snapshot();e.status[2]=0;e.target_station_idx[2]=-1;e.roles[:]=0;e.battery[2]=7.5
    e.step([8]*5)
    # 7.2 after Active drain: above legacy threshold7, below routed threshold7.4.
    assert e.last_return_control['trigger_plans']['2']['steps']==12
    assert 7<e.battery[2]<7.4 and e.status[2]==1

def test_boxed_high_battery_not_forced_return_and_low_battery_waits():
    e=env();e.battery[:]=100
    e.step([8]*5)
    assert e.status[0]==0 and e.last_return_control['trigger_plans']['0']['steps'] is None
    e.status[0]=1;e.battery[0]=1;e.target_station_idx[0]=2;p=e.uav_positions[0].copy()
    e.step([8]*5)
    np.testing.assert_array_equal(e.uav_positions[0],p)
    assert e.last_return_control['plans']['0']['steps'] is None
    assert e.last_return_control['wait_ids']==[0]

def test_all_five_share_charger_full_rate_and_depart_with_legal_endpoints():
    e=env();e.uav_positions[:]=[10,18];e.status[:]=2;e.battery[:]=0;e.target_station_idx[:]=2
    for t in range(1,17):
        e.step([8]*5)
        np.testing.assert_array_equal(e.battery,[5*t]*5)
        np.testing.assert_array_equal(e.status,[2 if t<16 else 0]*5)
    e.step([0]*5)
    assert sum(np.all(e.uav_positions==[9,18],axis=1))==1
    assert_endpoint_occupancy(e.uav_positions,e.status,e.charging_stations)

def test_battery_free_curriculum_preserves_legacy_physics():
    c=deepcopy(CONDITION);c.update(stage='S1',substage='S1',battery=False,mobility=False,active_count=2,starts='fixture',stay_only=True)
    a=BS5EnergyReturnEnv(c);b=BS5CollisionSafeCurriculumEnv(c)
    x=a.reset(seed=140002,assignment_seed=140002);y=b.reset(seed=140002,assignment_seed=140002)
    actions=[int(np.flatnonzero(m)[0]) for m in x['action_mask']]
    x=a.step(actions);y=b.step(actions)
    assert x['reward']==y['reward'];assert a.last_return_control['plans']=={}
    for k in ['obs','state','action_mask']:np.testing.assert_array_equal(x[k],y[k])
