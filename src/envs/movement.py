"""UE mobility, simultaneous UAV endpoints and committed BFS return routes."""

from collections import deque
from copy import deepcopy
from functools import lru_cache
from numbers import Integral
import itertools
import math
import numpy as np
from src.envs.config import MOVE_DELTAS

NEIGHBORS = ((-1, 0), (1, 0), (0, 1), (0, -1))
COLLISION_VERSION = "bs5_endpoint_collision_v2"

COLLISION_CONTRACT = {
    "version": "bs5_endpoint_collision_v2",
    "occupancy": "unique cell outside charging stations for all participating UAVs",
    "stations": "unlimited shared hub for Returning/Swapping/Active including rejoin/departure",
    "inactive": "curriculum status3 absent from routing and collision constraints",
    "choices": "requested endpoint or original endpoint only; no forced relocation",
    "priority": "Returning before Active, ascending UAV ID, among globally feasible endpoints",
    "stationary": "occupancy cannot be overwritten; dependencies propagate blocking",
    "swaps_cycles": "allowed simultaneous endpoint swaps/cycles; no swept-edge collision model",
    "return_blocked": "wait when no feasible requested move; no guaranteed progress through a stationary blocker",
    "energy": "Actual resolved displacement selects cruise or hover electrical power; blocked requests incur the separate Active movement penalty",
}
RETURN_CONTRACT = {
    "version": "bs5_energy_return_v4_commit",
    "planner": "commit BFS route; retain suffix while immediate next cell is free; replan only when blocked or no route",
    "ties": "shortest distance, station index, lexicographic N/S/E/W action path",
    "stations": "shared hubs as collision-v2; any status may share",
    "trigger": "Conservative route flight reserve + configured arrival SOC floor + one worst-case Active slot; trigger uses shortest feasible route, not committed movement memory",
    "energy": "Return move only if remaining Joules cover cruise power times slot duration; waits consume hover energy; payload radio off",
    "stranded": "Returning off-hub with insufficient return-slot energy: immobile, no radio, still occupies cell; no rescue/reset",
    "partial_energy": "remaining positive energy drains to zero while immobile; no negative energy",
    "active_shortfall": "requested Active slot cost unaffordable: convert to autonomous Return before movement/service",
    "priority": "collision-v2 joint endpoint resolver, effective status after energy override",
    "unreachable": "wait/retry; necessary backtracking is allowed; no guaranteed progress under arbitrary blockers",
    "arrival": "Grounded pack replacement for configured whole slots, then full capacity; shared station hubs and unlimited prepared spare packs",
    "replanning": "BFS over current occupied non-hub cells plus previous cell; if no route, retry without previous-cell exclusion",
    "suffix": "only next hop must be currently free; future suffix occupancy is reevaluated on arrival",
    "memory": "per Returning UAV: remaining route and last departed cell, cleared on reset/exit Return; not actor input",
}


def assert_endpoint_occupancy(positions, status, stations):
    hubs = {tuple(x) for x in np.asarray(stations).tolist()}
    occupied = {}
    for agent, (position, state) in enumerate(zip(positions, status)):
        if int(state) == 3:  # curriculum absence
            continue
        cell = tuple(position)
        if cell in hubs:
            continue
        if cell in occupied:
            raise AssertionError(
                f"{COLLISION_VERSION}: non-station co-location {occupied.get(cell)}/{agent} at {cell}"
            )
        occupied[cell] = agent


def resolve_endpoints(proposed, old, status_before, stations):
    """Choose first feasible lexicographic move vector, <=2**5 assignments.

    Considering joint endpoints prevents reverting an agent into a cell another
    agent has already claimed. A stationary blocker is retained; a dependency
    chain may move iff its final endpoints all satisfy occupancy.
    """
    wanted = np.asarray(proposed).copy()
    old_copy = np.asarray(old).copy()
    status = np.asarray(status_before)
    assert wanted.shape == old_copy.shape == (len(status), 2)
    assert len(status) <= 5, "This bounded resolver is versioned for the five-UAV task"
    assert np.isin(status, [0, 1, 2, 3]).all()
    assert_endpoint_occupancy(old_copy, status, stations)
    assert np.array_equal(wanted[status == 2], old_copy[status == 2]), "Swapping UAV moved"
    assert np.array_equal(wanted[status == 3], old_copy[status == 3]), "Inactive UAV moved"
    assert np.all(np.abs(wanted - old_copy).sum(axis=1) <= 1), "Non-local requested movement"
    priority = sorted(
        (i for i in range(len(status)) if status[i] in (0, 1)),
        key=lambda i: (0 if status[i] == 1 else 1, i),
    )
    movers = [i for i in priority if not np.array_equal(wanted[i], old_copy[i])]
    attempts = 0
    chosen = None
    # True first means the earliest feasible tuple respects priority exactly.
    for flags in itertools.product((True, False), repeat=len(movers)):
        attempts += 1
        candidate = old_copy.copy()
        for agent, move in zip(movers, flags):
            if move:
                candidate[agent] = wanted[agent]
        try:
            assert_endpoint_occupancy(candidate, status, stations)
        except AssertionError:
            continue
        chosen = candidate
        break
    assert chosen is not None, "Original legal occupancy must be a feasible fallback"
    proposed[:] = chosen
    blocked = [i for i in movers if np.array_equal(chosen[i], old_copy[i])]
    blocking = {
        str(i): [
            j
            for j in range(len(status))
            if j != i and status[j] != 3 and np.array_equal(chosen[j], wanted[i])
        ]
        for i in blocked
    }
    return dict(
        version=COLLISION_VERSION,
        requested_positions=wanted.tolist(),
        resolved_positions=chosen.tolist(),
        priority=priority,
        blocked_ids=blocked,
        blocked_by=blocking,
        returning_blocked_ids=[i for i in blocked if status[i] == 1],
        active_blocked_ids=[i for i in blocked if status[i] == 0],
        assignments_considered=attempts,
    )


@lru_cache(maxsize=8192)
def shortest_return_path(grid, stations, blocked, start):
    """Immutable pure route, or None. First BFS parent gives action-path tie rule."""
    forbidden = set(blocked) - set(stations) - {start}
    goals = {p: i for i, p in enumerate(stations)}
    parents = {start: None}
    q = deque([start])
    found = []
    goal_depth = None
    depth = {start: 0}
    while q:
        p = q.popleft()
        d = depth[p]
        if goal_depth is not None and d > goal_depth:
            break
        if p in goals:
            found.append((goals[p], p))
            goal_depth = d
            continue
        for dr, dc in NEIGHBORS:
            nxt = (p[0] + dr, p[1] + dc)
            if (
                0 <= nxt[0] < grid
                and 0 <= nxt[1] < grid
                and nxt not in forbidden
                and nxt not in parents
            ):
                parents[nxt] = p
                depth[nxt] = d + 1
                q.append(nxt)
    if not found:
        return None
    station, end = min(found)
    path = []
    while end is not None:
        path.append(end)
        end = parents[end]
    return station, tuple(reversed(path))


class MovementMixin:
    def _init_mobile_ues(self):
        """Initialize correlated moving groups with per-UE local motion.

        Group centers are only a mobility generator; they are never exposed to
        the policy. The environment keeps exact continuous UE positions, while
        the policy observes only the aggregated 5x5 demand map.
        """
        counts = np.full(
            self.cfg.num_hotspots, self.cfg.num_ues // self.cfg.num_hotspots, dtype=int
        )
        counts[: self.cfg.num_ues % self.cfg.num_hotspots] += 1
        labels = []
        for h, c in enumerate(counts):
            labels.extend([h] * int(c))
        self.ue_hotspot = np.asarray(labels, dtype=np.int64)
        self.rng.shuffle(self.ue_hotspot)
        chosen: list[np.ndarray] = []
        candidates = self.affected_cells.copy()
        self.rng.shuffle(candidates)
        for cell in candidates:
            if not chosen or min((np.linalg.norm(cell - x) for x in chosen)) >= 3.0:
                chosen.append(cell.astype(np.float64))
                if len(chosen) == self.cfg.num_hotspots:
                    break
        if len(chosen) < self.cfg.num_hotspots:
            chosen = [x.astype(np.float64) for x in candidates[: self.cfg.num_hotspots]]
        self.hotspot_centers = np.stack(chosen, axis=0)
        # The former clustering pass discarded these values before final setup.
        # Retain its draws so W00 seeds keep exactly the same RNG sequence.
        self.rng.standard_normal(self.cfg.num_hotspots)
        self.rng.standard_normal((self.cfg.num_ues, 2))
        self.ue_offsets = self.rng.normal(
            0.0, self.cfg.group_spread_cells, size=(self.cfg.num_ues, 2)
        )
        self._clip_group_offsets()
        n_groups = self.cfg.num_hotspots
        center_angles = self.rng.uniform(0.0, 2.0 * math.pi, size=n_groups)
        center_speed_cells = (
            self.cfg.group_center_speed_mps * self.cfg.slot_s / self.cfg.cell_size_m
        )
        self.hotspot_vel = (
            np.stack([np.sin(center_angles), np.cos(center_angles)], axis=1) * center_speed_cells
        )
        local_speed_mps = self.rng.uniform(
            self.cfg.ue_local_speed_min_mps, self.cfg.ue_local_speed_max_mps, size=self.cfg.num_ues
        )
        local_angles = self.rng.uniform(0.0, 2.0 * math.pi, size=self.cfg.num_ues)
        local_speed_cells = local_speed_mps * self.cfg.slot_s / self.cfg.cell_size_m
        self.ue_local_velocity_cells_per_slot = (
            np.stack([np.sin(local_angles), np.cos(local_angles)], axis=1)
            * local_speed_cells[:, None]
        )
        self._project_group_positions_inside_region()

    def _advance_mobile_ues(self):
        """Advance reference centers and individual UE motion by one slot.

        Centers move slowly and persistently. Each UE keeps an independent
        local velocity relative to its center. A hard relative-distance bound
        prevents the group from diffusing away, and all absolute UE positions
        are kept inside the failed-BS Voronoi service region.
        """
        if not self.condition["mobility"]:
            return
        turns = self.rng.normal(0.0, self.cfg.group_turn_noise_std_rad, size=self.cfg.num_hotspots)
        self.hotspot_vel = self._rotate_vectors(self.hotspot_vel, turns)
        for h in range(self.cfg.num_hotspots):
            cur = self.hotspot_centers[h].copy()
            vel = self.hotspot_vel[h].copy()
            proposal = cur + vel
            if not self._center_is_valid_floor(proposal):
                candidates = (np.asarray([-vel[0], vel[1]]), np.asarray([vel[0], -vel[1]]), -vel)
                accepted = False
                for cand_vel in candidates:
                    cand = cur + cand_vel
                    if self._center_is_valid_floor(cand):
                        vel = cand_vel
                        proposal = cand
                        accepted = True
                        break
                if not accepted:
                    proposal = cur
                    vel = -vel
            self.hotspot_vel[h] = vel
            self.hotspot_centers[h] = proposal
        local_turn = self.rng.normal(
            0.0, self.cfg.ue_local_turn_noise_std_rad, size=self.cfg.num_ues
        )
        self.ue_local_velocity_cells_per_slot = self._rotate_vectors(
            self.ue_local_velocity_cells_per_slot, local_turn
        )
        for u in range(self.cfg.num_ues):
            h = int(self.ue_hotspot[u])
            offset = self.ue_offsets[u].copy()
            vel = self.ue_local_velocity_cells_per_slot[u].copy()
            proposal_offset = offset + vel
            if np.linalg.norm(proposal_offset) > self.cfg.max_ue_offset_cells:
                norm = max(float(np.linalg.norm(offset)), 1e-09)
                radial = offset / norm
                vel = vel - 2.0 * np.dot(vel, radial) * radial
                proposal_offset = offset + vel
                if np.linalg.norm(proposal_offset) > self.cfg.max_ue_offset_cells:
                    nrm = max(float(np.linalg.norm(proposal_offset)), 1e-09)
                    proposal_offset *= self.cfg.max_ue_offset_cells / nrm
            absolute = self.hotspot_centers[h] + proposal_offset
            if not self._ue_position_is_valid(absolute):
                candidates = (np.asarray([-vel[0], vel[1]]), np.asarray([vel[0], -vel[1]]), -vel)
                accepted = False
                for cand_vel in candidates:
                    cand_offset = offset + cand_vel
                    if np.linalg.norm(cand_offset) > self.cfg.max_ue_offset_cells:
                        continue
                    cand_abs = self.hotspot_centers[h] + cand_offset
                    if self._ue_position_is_valid(cand_abs):
                        vel = cand_vel
                        proposal_offset = cand_offset
                        absolute = cand_abs
                        accepted = True
                        break
                if not accepted:
                    proposal_offset = np.zeros(2, dtype=np.float64)
                    absolute = self.hotspot_centers[h].copy()
                    vel = -vel
            self.ue_offsets[u] = proposal_offset
            self.ue_local_velocity_cells_per_slot[u] = vel
            self.ue_positions[u] = absolute

    def _clip_group_offsets(self):
        norms = np.linalg.norm(self.ue_offsets, axis=1)
        bad = norms > self.cfg.max_ue_offset_cells
        if np.any(bad):
            self.ue_offsets[bad] *= (self.cfg.max_ue_offset_cells / norms[bad])[:, None]

    def _project_group_positions_inside_region(self):
        for u in range(self.cfg.num_ues):
            h = int(self.ue_hotspot[u])
            pos = self.hotspot_centers[h] + self.ue_offsets[u]
            if not self._ue_position_is_valid(pos):
                self.ue_offsets[u] = 0.0
                pos = self.hotspot_centers[h].copy()
            self.ue_positions[u] = pos

    def _ue_position_is_valid(self, pos: np.ndarray) -> bool:
        if np.any(pos < 0.0) or np.any(pos >= self.cfg.grid_size):
            return False
        cell = np.floor(pos).astype(int)
        cell = np.clip(cell, 0, self.cfg.grid_size - 1)
        return bool(self.affected_mask[tuple(cell)])

    def _center_is_valid_floor(self, center: np.ndarray) -> bool:
        return self._ue_position_is_valid(center)

    def ue_local_speed_mps(self) -> np.ndarray:
        return (
            np.linalg.norm(self.ue_local_velocity_cells_per_slot, axis=1)
            * self.cfg.cell_size_m
            / self.cfg.slot_s
        )

    def group_center_speed_mps(self) -> np.ndarray:
        return np.linalg.norm(self.hotspot_vel, axis=1) * self.cfg.cell_size_m / self.cfg.slot_s

    def _build_failed_bs_voronoi_mask(self) -> np.ndarray:
        g = self.cfg.grid_size
        mask = np.zeros((g, g), dtype=bool)
        for r in range(g):
            for c in range(g):
                xy = self._cell_to_xy(np.asarray([r, c]))
                nearest = int(np.argmin(np.linalg.norm(self.bs_xy - xy, axis=1)))
                mask[r, c] = nearest == self.cfg.failed_bs_idx
        return mask

    def _inside_grid(self, cell: np.ndarray) -> bool:
        return bool(np.all(cell >= 0) and np.all(cell < self.cfg.grid_size))

    def _validate_cells(self, cells: np.ndarray):
        for c in cells:
            if not self._inside_grid(c):
                raise ValueError(f"invalid grid cell {c.tolist()}")

    def _set_failure_topology(self, idx):
        self._check_id(idx, len(self.bs_xy))
        self.cfg.failed_bs_idx = int(idx)
        self.failed_bs_xy = self.bs_xy[idx].copy()
        self.failed_bs_cell = self.bs_cells[idx].copy()
        self.surviving_bs_xy = np.delete(self.bs_xy, idx, axis=0)
        self.surviving_bs_cells = np.delete(self.bs_cells, idx, axis=0)
        self.affected_mask = self._build_failed_bs_voronoi_mask()
        self.affected_cells = np.argwhere(self.affected_mask)
        if len(self.affected_cells) < self.cfg.num_hotspots:
            raise ValueError("selected failure region has too few hotspot cells")

    @staticmethod
    def _check_id(idx, n):
        if isinstance(idx, bool) or not isinstance(idx, Integral) or (not 0 <= idx < n):
            raise ValueError(f"failed BS ID must be an integer in [0,{n - 1}]")

    def _shortest_plan(self, cell, agent):
        stations = tuple(map(tuple, self.charging_stations.tolist()))
        blocked = tuple(
            sorted(
                (
                    tuple(p)
                    for i, (p, s) in enumerate(zip(self.uav_positions, self.status))
                    if i != agent and s != 3 and (tuple(p) not in stations)
                )
            )
        )
        result = shortest_return_path(self.cfg.grid_size, stations, blocked, tuple(map(int, cell)))
        if result is None:
            idx = int(np.argmin(np.abs(self.charging_stations - cell).sum(axis=1)))
            fallback = int(np.abs(self.charging_stations[idx] - cell).sum())
            return dict(station=idx, steps=None, path=None, trigger_steps=fallback)
        idx, path = result
        steps = len(path) - 1
        return dict(
            station=idx,
            steps=steps,
            trigger_steps=steps,
            path=[list(p) for p in path],
        )

    def _plan(self, cell, agent):
        if not self._return_planning:
            return self._shortest_plan(cell, agent)
        assert 0 <= agent < 5
        start = tuple(map(int, cell))
        stations = tuple(map(tuple, self.charging_stations.tolist()))
        blocked = {
            tuple(map(int, p))
            for i, (p, s) in enumerate(zip(self.uav_positions, self.status))
            if i != agent and s != 3 and (tuple(p) not in stations)
        }
        cached = self._route_cache[agent]
        valid_cache = cached is not None and tuple(cached["path"][0]) == start
        reusable = valid_cache and (
            len(cached["path"]) == 1 or tuple(cached["path"][1]) not in blocked
        )
        if reusable:
            plan = deepcopy(cached)
            reason = "retain_next_hop"
            avoided = False
        else:
            previous = self._previous_return_cell[agent]
            avoid = blocked | ({tuple(previous)} if previous is not None else set())
            result = shortest_return_path(self.cfg.grid_size, stations, tuple(sorted(avoid)), start)
            avoided = previous is not None and result is not None
            fallback = result is None and previous is not None
            if fallback:
                result = shortest_return_path(
                    self.cfg.grid_size, stations, tuple(sorted(blocked)), start
                )
            if result is None:
                plan = self._shortest_plan(cell, agent)
                reason = "unreachable_wait"
            else:
                station, path = result
                steps = len(path) - 1
                plan = dict(
                    station=station,
                    steps=steps,
                    trigger_steps=steps,
                    path=[list(p) for p in path],
                )
                reason = (
                    "necessary_backtrack"
                    if fallback
                    else "replan_blocked"
                    if valid_cache
                    else "initial_route"
                )
        self._route_cache[agent] = deepcopy(plan) if plan["path"] is not None else None
        self._route_decisions[str(agent)] = dict(
            reason=reason,
            previous_cell=deepcopy(self._previous_return_cell[agent]),
            previous_cell_excluded=avoided,
        )
        return plan

    def _move_slot(self, slot):
        slot.effective = self.status.copy()
        slot.proposed = slot.old.copy()
        slot.roles = self.logical_roles().copy()
        self._return_planning = True
        self._route_decisions = {}
        try:
            for aid in range(5):
                if slot.effective[aid] == 0:
                    move, role = self.decode_action(slot.raw[aid])
                    slot.proposed[aid] = slot.old[aid] + MOVE_DELTAS[move]
                    slot.roles[aid] = role
                elif slot.effective[aid] == 1:
                    plan = self._plan(slot.old[aid], aid)
                    self.target_station_idx[aid] = plan["station"]
                    return_move_j = (
                        slot.e.cruise_w * self.cfg.slot_s if slot.e.include_flight_energy else 0.0
                    )
                    if (
                        plan["path"] is not None
                        and len(plan["path"]) > 1
                        and (not slot.enabled or self.energy_j[aid] >= return_move_j)
                    ):
                        slot.proposed[aid] = plan["path"][1]
        finally:
            self._return_planning = False
        self.last_collision_resolution = resolve_endpoints(
            slot.proposed, slot.old, slot.effective, self.charging_stations
        )
        self.uav_positions = slot.proposed
        self._hybrid_ids = set(np.flatnonzero((slot.roles == 2) & (slot.effective == 0)).tolist())
        self.roles[:] = np.where(slot.roles == 2, 1, slot.roles)
        for aid in np.flatnonzero(slot.effective == 1):
            plan = self._route_cache[aid]
            if not np.array_equal(slot.old[aid], slot.proposed[aid]) and plan is not None:
                self._previous_return_cell[aid] = slot.old[aid].tolist()
                plan["path"] = plan["path"][1:]
                plan["steps"] = plan["trigger_steps"] = len(plan["path"]) - 1
            station = self.charging_stations[self.target_station_idx[aid]]
            if np.array_equal(slot.proposed[aid], station):
                self.status[aid] = 2
                self.swap_remaining[aid] = self.swap_slots
                self.station_arrival_time[aid] = self.t + 1
                self.energy_events.append(
                    dict(
                        event="swap_start",
                        slot=self.t + 1,
                        agent=int(aid),
                        station=int(self.target_station_idx[aid]),
                    )
                )

    @staticmethod
    def _rotate_vectors(vectors: np.ndarray, angles: np.ndarray) -> np.ndarray:
        c = np.cos(angles)
        s = np.sin(angles)
        x = vectors[:, 0].copy()
        y = vectors[:, 1].copy()
        out = vectors.copy()
        out[:, 0] = c * x - s * y
        out[:, 1] = s * x + c * y
        return out
