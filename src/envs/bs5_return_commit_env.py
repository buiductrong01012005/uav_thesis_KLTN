"""Return-v4: retain a legal next hop instead of chasing a shorter live route.

Only the autonomous Returning controller owns this history. Actors, trigger
thresholds, energy accounting and collision-v2 stay exactly as in Return-v3.
No finite-energy controller can guarantee arrival under arbitrary blockers.
"""
from copy import deepcopy
import numpy as np
from src.envs.bs5_energy_return_env import (
    BS5EnergyReturnEnv, RETURN_CONTRACT as V3_CONTRACT, shortest_return_path,
)

RETURN_VERSION = 'bs5_energy_return_v4_commit'
RETURN_CONTRACT = dict(V3_CONTRACT, version=RETURN_VERSION,
    planner='commit BFS route; retain suffix while immediate next cell is free; replan only when blocked or no route',
    replanning='BFS over current occupied non-hub cells plus previous cell; if no route, retry without previous-cell exclusion',
    suffix='only next hop must be currently free; future suffix occupancy is reevaluated on arrival',
    memory='per Returning UAV: remaining route and last departed cell, cleared on reset/exit Return; not actor input',
    trigger='unchanged Return-v3 shortest feasible path reserve trigger; committed movement memory is never used by trigger/telemetry',
    unreachable='wait/retry; necessary backtracking is allowed; no guaranteed progress under arbitrary blockers')


class BS5CommittedReturnEnv(BS5EnergyReturnEnv):
    return_version = RETURN_VERSION

    def __init__(self, *args, **kwargs):
        self._route_cache = [None] * 5
        self._previous_return_cell = [None] * 5
        self._return_planning = False
        self._route_decisions = {}
        super().__init__(*args, **kwargs)

    def reset(self, *args, **kwargs):
        self._route_cache = [None] * 5
        self._previous_return_cell = [None] * 5
        self._return_planning = False
        self._route_decisions = {}
        return super().reset(*args, **kwargs)

    def _plan(self, cell, agent):
        if not self._return_planning:
            return super()._plan(cell, agent)
        assert 0 <= agent < 5
        start = tuple(map(int, cell))
        stations = tuple(map(tuple, self.charging_stations.tolist()))
        blocked = {tuple(map(int, p)) for i, (p, s) in
                   enumerate(zip(self.uav_positions, self.status))
                   if i != agent and s != 3 and tuple(p) not in stations}
        cached = self._route_cache[agent]
        valid_cache = (cached is not None and tuple(cached['path'][0]) == start)
        reusable = valid_cache and (len(cached['path']) == 1 or
                                   tuple(cached['path'][1]) not in blocked)
        if reusable:
            plan = deepcopy(cached)
            reason = 'retain_next_hop'
            avoided = False
        else:
            previous = self._previous_return_cell[agent]
            avoid = blocked | ({tuple(previous)} if previous is not None else set())
            result = shortest_return_path(self.cfg.grid_size, stations,
                                          tuple(sorted(avoid)), start)
            avoided = previous is not None and result is not None
            fallback = result is None and previous is not None
            if fallback:
                result = shortest_return_path(self.cfg.grid_size, stations,
                                              tuple(sorted(blocked)), start)
            if result is None:
                # The v3 fallback logs unavailable route and the geometric lower bound.
                plan = super()._plan(cell, agent)
                reason = 'unreachable_wait'
            else:
                station, path = result
                steps = len(path) - 1
                plan = dict(station=station, steps=steps, trigger_steps=steps,
                    path=[list(p) for p in path],
                    energy_for_path=steps*self.cfg.returning_step_drain)
                reason = ('necessary_backtrack' if fallback else
                          'replan_blocked' if valid_cache else 'initial_route')
        self._route_cache[agent] = deepcopy(plan) if plan['path'] is not None else None
        self._route_decisions[str(agent)] = dict(reason=reason,
            previous_cell=deepcopy(self._previous_return_cell[agent]),
            previous_cell_excluded=avoided)
        return plan

    def _resolve_collisions(self, proposed, old, status_before):
        self._route_decisions = {}
        self._return_planning = True
        try:
            super()._resolve_collisions(proposed, old, status_before)
        finally:
            self._return_planning = False
        self.last_return_control['version'] = RETURN_VERSION
        self.last_return_control['route_decisions'] = deepcopy(self._route_decisions)

    def step(self, actions):
        memory_before = deepcopy(self._route_cache)
        previous_before = deepcopy(self._previous_return_cell)
        old = self.uav_positions.copy()
        out = super().step(actions)
        for a in range(5):
            if self.status[a] != 1:
                self._route_cache[a] = None
                self._previous_return_cell[a] = None
                continue
            if self.last_return_control['effective_status'][a] != 1:
                # Return was triggered at this slot's end, no Return move yet.
                self._route_cache[a] = None
                self._previous_return_cell[a] = None
                continue
            if not np.array_equal(old[a], self.uav_positions[a]):
                self._previous_return_cell[a] = old[a].tolist()
                plan = self._route_cache[a]
                assert plan is not None and plan['path'][1] == self.uav_positions[a].tolist()
                plan['path'] = plan['path'][1:]
                plan['steps'] = plan['trigger_steps'] = len(plan['path']) - 1
                plan['energy_for_path'] = plan['steps'] * self.cfg.returning_step_drain
        d = self.last_return_control
        d.update(route_memory_before=memory_before,
                 route_memory_after=deepcopy(self._route_cache),
                 previous_cells_before=previous_before,
                 previous_cells_after=deepcopy(self._previous_return_cell))
        return out

    def _episode_meta(self):
        meta = super()._episode_meta()
        meta['return_contract'] = RETURN_CONTRACT
        return meta
