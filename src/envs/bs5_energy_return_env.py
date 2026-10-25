"""Versioned Return controller; legacy files, rewards and actor inputs unchanged.

Autonomous Return has current fleet positions (already available to controller).
It does not use UE/future trajectories or choose Access/Relay for the policy.
"""
from collections import deque
from functools import lru_cache
import numpy as np

from src.envs.bs5_collision_safe_env import (
    BS5CollisionSafeCurriculumEnv, resolve_endpoints, assert_endpoint_occupancy,
)

RETURN_VERSION = "bs5_energy_return_v3"
RETURN_CONTRACT = {
    "version": RETURN_VERSION,
    "planner": "shortest 4-neighbor BFS, current occupied non-hub cells blocked; replan each slot",
    "ties": "shortest distance, station index, lexicographic N/S/E/W action path",
    "stations": "shared hubs as collision-v2; any status may share",
    "trigger": "after energy: battery <= feasible path steps*return drain + legacy margin; if currently boxed use geometric lower bound and log unavailable route",
    "energy": "Return move only if battery >= full return-slot drain, no numerical tolerance subsidy; waits still drain",
    "stranded": "Returning off-hub with insufficient return-slot energy: immobile, no radio, still occupies cell; no rescue/reset",
    "partial_energy": "remaining positive energy drains to zero while immobile; no negative energy",
    "active_shortfall": "requested Active slot cost unaffordable: convert to autonomous Return before movement/service",
    "priority": "collision-v2 joint endpoint resolver, effective status after energy override",
    "unreachable": "wait and replan; no guaranteed progress under arbitrary blockers",
    "unchanged": "legacy drains/margin/charge/hysteresis, reward/radio, 111 state/116 obs, continuing S4 bootstrap",
}
NEIGHBORS = ((-1,0),(1,0),(0,1),(0,-1))


@lru_cache(maxsize=8192)
def shortest_return_path(grid, stations, blocked, start):
    """Immutable pure route, or None. First BFS parent gives action-path tie rule."""
    forbidden=set(blocked)-set(stations)-{start}
    goals={p:i for i,p in enumerate(stations)}
    parents={start:None};q=deque([start]);found=[];goal_depth=None;depth={start:0}
    while q:
        p=q.popleft();d=depth[p]
        if goal_depth is not None and d>goal_depth:break
        if p in goals:found.append((goals[p],p));goal_depth=d;continue
        for dr,dc in NEIGHBORS:
            nxt=(p[0]+dr,p[1]+dc)
            if 0<=nxt[0]<grid and 0<=nxt[1]<grid and nxt not in forbidden and nxt not in parents:
                parents[nxt]=p;depth[nxt]=d+1;q.append(nxt)
    if not found:return None
    station,end=min(found);path=[]
    while end is not None:path.append(end);end=parents[end]
    return station,tuple(reversed(path))


class BS5EnergyReturnEnv(BS5CollisionSafeCurriculumEnv):
    return_version=RETURN_VERSION

    def __init__(self,*args,**kwargs):
        self.last_return_control=None
        super().__init__(*args,**kwargs)

    def reset(self,*args,**kwargs):
        out=super().reset(*args,**kwargs);self.last_return_control=None
        return out

    def _plan(self,cell,agent):
        stations=tuple(map(tuple,self.charging_stations.tolist()))
        blocked=tuple(sorted(tuple(p) for i,(p,s) in enumerate(zip(self.uav_positions,self.status))
            if i!=agent and s!=3 and tuple(p) not in stations))
        result=shortest_return_path(self.cfg.grid_size,stations,blocked,tuple(map(int,cell)))
        if result is None:
            idx=int(np.argmin(np.abs(self.charging_stations-cell).sum(axis=1)))
            fallback=int(np.abs(self.charging_stations[idx]-cell).sum())
            return dict(station=idx,steps=None,path=None,energy_for_path=None,trigger_steps=fallback)
        idx,path=result;steps=len(path)-1
        return dict(station=idx,steps=steps,trigger_steps=steps,path=[list(p) for p in path],
            energy_for_path=steps*self.cfg.returning_step_drain)

    def _nearest_station(self,cell):
        # Parent invokes this after service/energy for each Active UAV. It is
        # also used by audit distance telemetry. No hidden state is mutated.
        if not getattr(self,"ready",False) or not self.condition['battery']:
            return super()._nearest_station(cell)
        matches=np.flatnonzero(np.all(self.uav_positions==cell,axis=1))
        agent=int(matches[0]) if len(matches) else -1
        plan=self._plan(cell,agent)
        # Transient boxing (including the launch cross) must not force a full
        # battery UAV into Return. No route means only a lower bound is known;
        # log this explicitly, never claim that the reserve guarantees arrival.
        return plan['station'],plan['trigger_steps']

    def _resolve_collisions(self,proposed,old,status_before):
        frozen=not self.condition['battery'];forced=[];plans={};prevented=[]
        if not frozen:
            # status_before belongs to the parent slot: mutating this local copy
            # deliberately makes its energy loop use Return drain for overrides.
            for a in np.flatnonzero(status_before==0):
                moved=not np.array_equal(proposed[a],old[a])
                role=int(self._slot_actions[a])%2
                cost=self.cfg.hover_drain+self.cfg.move_extra_drain*moved+(
                    self.cfg.access_radio_drain if role==0 else self.cfg.relay_radio_drain)
                if float(self.battery[a])<cost:
                    status_before[a]=1;self.status[a]=1;forced.append(int(a))
                    if self.return_start_time[a]<0:self.return_start_time[a]=self.t+1
            for a in np.flatnonzero(status_before==1):
                plan=self._plan(old[a],int(a));plans[str(a)]=plan
                self.target_station_idx[a]=plan['station']
                proposed[a]=old[a]
                if plan['steps'] is not None and plan['steps']>0:
                    if float(self.battery[a])>=self.cfg.returning_step_drain:
                        proposed[a]=plan['path'][1]
                    else:prevented.append(int(a))
        self.last_collision_resolution=resolve_endpoints(proposed,old,status_before,self.charging_stations)
        self.last_return_control=dict(version=RETURN_VERSION,effective_status=status_before.tolist(),
            forced_return_ids=forced,plans=plans,prevented_energy_move_ids=prevented)

    def step(self,actions):
        # Preserve all legacy wrappers (UE RNG, request telemetry, snapshots,
        # curriculum absence/freeze). Only hooks above change Return mechanics.
        self._slot_actions=tuple(map(int,actions));before=self.status.copy();old=self.uav_positions.copy()
        targets_before=self.target_station_idx.tolist()
        out=super().step(actions)
        d=self.last_return_control;effective=d['effective_status'];hubs={tuple(p) for p in self.charging_stations}
        eligible=[a for a in range(5) if effective[a]==0 or (before[a]==2 and self.status[a]==0)]
        d['trigger_plans']={str(a):self._plan(self.uav_positions[a],a) for a in eligible} if self.condition['battery'] else {}
        d['target_station_idx_post']=self.target_station_idx.tolist()
        d['target_station_idx_before']=targets_before
        d['wait_ids']=[a for a,s in enumerate(effective) if s==1 and tuple(old[a]) not in hubs and np.array_equal(old[a],self.uav_positions[a])]
        d['moved_ids']=[a for a,s in enumerate(effective) if s==1 and not np.array_equal(old[a],self.uav_positions[a])]
        d['stranded_ids_post']=[a for a,s in enumerate(self.status) if s==1 and tuple(self.uav_positions[a]) not in hubs and float(self.battery[a])<self.cfg.returning_step_drain]
        out['info']['return_control']=d
        assert_endpoint_occupancy(self.uav_positions,self.status,self.charging_stations)
        return out

    def _episode_meta(self):
        meta=super()._episode_meta();meta['return_contract']=RETURN_CONTRACT
        return meta
