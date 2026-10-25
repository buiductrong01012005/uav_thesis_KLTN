"""Independent stdlib Return-v4 audit: reverse BFS plus persistent route state.

Reuses the unchanged v3 energy/lifecycle checker with an explicit movement-plan
callback in an isolated globals dictionary. Never patches the v3 module/env.
"""
from copy import deepcopy
from types import FunctionType
try:
    import bs5_return_audit as v3
except ImportError:
    from scripts.analysis import bs5_return_audit as v3

VERSION = 'bs5_energy_return_v4_commit'


def committed_plan(before, effective, agent, cfg, stations, cache, previous):
    start = tuple(before['positions'][agent])
    hubs = set(stations)
    blocked = {tuple(p) for a, (p, s) in
               enumerate(zip(before['positions'], effective))
               if a != agent and s != 3 and tuple(p) not in hubs}
    valid = cache is not None and tuple(cache['path'][0]) == start
    reusable = valid and (len(cache['path']) == 1 or tuple(cache['path'][1]) not in blocked)
    if reusable:
        return deepcopy(cache), dict(reason='retain_next_hop', previous_cell=previous,
                                     previous_cell_excluded=False)
    avoid = blocked | ({tuple(previous)} if previous is not None else set())
    idx, steps, path, trigger = v3.route(cfg['grid_size'], stations,
                                       tuple(sorted(avoid)), start)
    avoided = previous is not None and steps is not None
    fallback = steps is None and previous is not None
    if fallback:
        idx, steps, path, trigger = v3.route(cfg['grid_size'], stations,
                                           tuple(sorted(blocked)), start)
    reason = ('unreachable_wait' if steps is None else
              'necessary_backtrack' if fallback else
              'replan_blocked' if valid else 'initial_route')
    plan = dict(station=idx, steps=steps, path=path, trigger_steps=trigger,
                energy_for_path=None if steps is None else steps*cfg['returning_step_drain'])
    return plan, dict(reason=reason, previous_cell=previous,
                      previous_cell_excluded=avoided)


class CommitAuditor:
    def __init__(self):
        self.cache = [None]*5
        self.previous = [None]*5
        self.rows = 0

    def validate(self, row, cfg):
        d = row['return_control']
        assert d['version'] == VERSION
        assert d['route_memory_before'] == self.cache
        assert d['previous_cells_before'] == self.previous
        stations = tuple(map(tuple, row['charging_station_cells']))
        expected = {}; decisions = {}
        if row['task_condition']['battery']:
            for a, s in enumerate(d['effective_status']):
                if s == 1:
                    expected[str(a)], decisions[str(a)] = committed_plan(
                        row['decision'], d['effective_status'], a, cfg, stations,
                        self.cache[a], self.previous[a])
        assert d['plans'] == expected and d['route_decisions'] == decisions
        def movement_or_trigger(positions, status, a, config, hubs):
            if positions is row['decision']['positions']:
                return expected[str(a)]
            return v3.plan_for(positions, status, a, config, hubs)
        context = dict(v3.validate_return.__globals__, VERSION=VERSION,
                       plan_for=movement_or_trigger)
        checker = FunctionType(v3.validate_return.__code__, context)
        checker(row, cfg)
        for a in range(5):
            if row['post']['status'][a] != 1 or d['effective_status'][a] != 1:
                self.cache[a] = None; self.previous[a] = None
                continue
            plan = deepcopy(expected[str(a)])
            self.cache[a] = plan if plan['path'] is not None else None
            if row['decision']['positions'][a] != row['post']['positions'][a]:
                self.previous[a] = row['decision']['positions'][a][:]
                assert plan['path'][1] == row['post']['positions'][a]
                plan['path'] = plan['path'][1:]
                plan['steps'] = plan['trigger_steps'] = len(plan['path'])-1
                plan['energy_for_path'] = plan['steps']*cfg['returning_step_drain']
        assert d['route_memory_after'] == self.cache
        assert d['previous_cells_after'] == self.previous
        self.rows += 1
