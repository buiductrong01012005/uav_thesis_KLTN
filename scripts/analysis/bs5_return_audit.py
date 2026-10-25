"""Independent stdlib physics auditor. Reverse distance fields, not env planner."""
from collections import deque
from functools import lru_cache
import struct

VERSION='bs5_energy_return_v3'
DELTAS=((-1,0),(1,0),(0,1),(0,-1),(0,0))
def f32(x):return struct.unpack('f',struct.pack('f',float(x)))[0]

@lru_cache(maxsize=8192)
def route(grid,stations,blocked,start):
    forbidden=set(blocked)-set(stations)-{start};choices=[]
    for idx,goal in enumerate(stations):
        distances={goal:0};todo=deque([goal])
        while todo:
            p=todo.popleft()
            for dr,dc in DELTAS[:4]:
                q=(p[0]+dr,p[1]+dc)
                if 0<=q[0]<grid and 0<=q[1]<grid and q not in forbidden and q not in distances:
                    distances[q]=distances[p]+1;todo.append(q)
        if start in distances:choices.append((distances[start],idx,distances))
    if not choices:
        steps,idx=min((sum(abs(a-b) for a,b in zip(start,p)),i) for i,p in enumerate(stations))
        return idx,None,None,steps
    steps,idx,distances=min(choices,key=lambda x:x[:2]);p=start;path=[list(p)]
    while distances[p]:
        p=next((p[0]+dr,p[1]+dc) for dr,dc in DELTAS[:4]
            if distances.get((p[0]+dr,p[1]+dc))==distances[p]-1)
        path.append(list(p))
    return idx,steps,path,steps

def plan_for(positions,status,a,cfg,stations):
    blocked=tuple(sorted(tuple(p) for i,(p,s) in enumerate(zip(positions,status)) if i!=a and s!=3 and tuple(p) not in stations))
    idx,steps,path,trigger=route(cfg['grid_size'],stations,blocked,tuple(positions[a]))
    return dict(station=idx,steps=steps,path=path,energy_for_path=None if steps is None else steps*cfg['returning_step_drain'],trigger_steps=trigger)

def validate_return(r,cfg):
    d=r['return_control'];assert d['version']==VERSION
    stations=tuple(map(tuple,r['charging_station_cells']));hubs=set(stations)
    before=r['decision'];post=r['post'];service=r['service'];effective=before['status'][:];frozen=not r['task_condition']['battery']
    wanted=[p[:] for p in before['positions']];forced=[];plans={};prevented=[]
    for a,s in enumerate(before['status']):
        if s!=0:continue
        dr,dc=DELTAS[r['raw_actions'][a]//2];p=before['positions'][a]
        wanted[a]=[p[0]+dr,p[1]+dc]
        cost=cfg['hover_drain']+cfg['move_extra_drain']*bool(dr or dc)+(cfg['access_radio_drain'] if r['raw_actions'][a]%2==0 else cfg['relay_radio_drain'])
        if not frozen and before['battery'][a]<cost:forced.append(a);effective[a]=1
    assert d['effective_status']==effective and d['forced_return_ids']==forced
    for a,s in enumerate(effective):
        if s==1 and not frozen:
            plan=plan_for(before['positions'],effective,a,cfg,stations);plans[str(a)]=plan
            wanted[a]=before['positions'][a][:]
            if plan['steps'] is not None and plan['steps']>0:
                if before['battery'][a]>=cfg['returning_step_drain']:wanted[a]=plan['path'][1]
                else:prevented.append(a)
    assert d['plans']==plans and d['prevented_energy_move_ids']==prevented
    assert r['collision']['requested_positions']==wanted, 'Return route/request mismatch'
    statuses=effective[:];roles=before['roles'][:];targets=d['target_station_idx_before'][:]
    expected_b=[]
    for a,s in enumerate(effective):
        b=before['battery'][a];pos=post['positions'][a]
        if before['status'][a]==0:roles[a]=r['raw_actions'][a]%2
        if s==1:
            targets[a]=plans[str(a)]['station']
            if pos==list(stations[targets[a]]):statuses[a]=2
        assert service['status'][a]==statuses[a] and service['roles'][a]==roles[a]
        if frozen:nb=b
        elif s==0:
            moved=wanted[a]!=before['positions'][a]
            drain=cfg['hover_drain']+cfg['move_extra_drain']*moved+(cfg['access_radio_drain'] if roles[a]==0 else cfg['relay_radio_drain'])
            nb=max(0.,f32(b-f32(drain)))
        elif s==1:nb=max(0.,f32(b-f32(cfg['returning_step_drain'])))
        elif s==2:nb=min(cfg['max_battery'],f32(b+f32(cfg['charge_rate_per_slot'])))
        else:nb=b
        expected_b.append(f32(nb))
        assert post['battery'][a]==f32(nb),'Energy mismatch'
        if s==2 and not frozen and nb>=cfg['rejoin_threshold']:
            statuses[a]=0;roles[a]=0;targets[a]=-1
        if s==1 and pos!=before['positions'][a]:
            assert b>=cfg['returning_step_drain'], 'Unfunded Return movement'
        assert 0<=post['battery'][a]<=cfg['max_battery']
    eligible=[a for a in range(5) if effective[a]==0 or (before['status'][a]==2 and statuses[a]==0)]
    trigger={str(a):plan_for(post['positions'],statuses,a,cfg,stations) for a in eligible} if not frozen else {}
    assert d['trigger_plans']==trigger,'Trigger plan mismatch'
    for a in eligible:
        if frozen:continue
        plan=trigger[str(a)];threshold=plan['trigger_steps']*cfg['returning_step_drain']+cfg['return_safety_margin']
        if expected_b[a]<=f32(threshold):statuses[a]=1;targets[a]=plan['station']
    assert post['status']==statuses and post['roles']==roles,'Lifecycle mismatch'
    assert d['target_station_idx_post']==targets
    wait=[a for a,s in enumerate(effective) if s==1 and tuple(before['positions'][a]) not in hubs and before['positions'][a]==post['positions'][a]]
    moved=[a for a,s in enumerate(effective) if s==1 and before['positions'][a]!=post['positions'][a]]
    stranded=[a for a,s in enumerate(statuses) if s==1 and tuple(post['positions'][a]) not in hubs and post['battery'][a]<cfg['returning_step_drain']]
    assert d['wait_ids']==wait and d['moved_ids']==moved and d['stranded_ids_post']==stranded
    for a in forced:assert r['applied_actions'][a]['autonomous']
