"""Fast coordinate-local-search baseline for dynamic Stage-B recovery.

Unlike the exact/static oracle, this planner never enumerates joint UAV
placements.  Holding the other UAVs fixed, it scans O(grid_cells * roles)
choices for one active UAV at a time and repeats only a few coordinate sweeps.
It replans when the active fleet changes (dropout/rejoin).

This provides a scalable online benchmark with privileged knowledge of the
Stage-B discovered UE map, while remaining polynomial in grid size.
"""
from __future__ import annotations

import numpy as np

from src.envs.v0_env import ACCESS, RELAY


class CoordinateGreedyPolicy:
    def __init__(self, env, sweeps: int = 3):
        self.env = env
        self.sweeps = int(sweeps)
        self.targets = None
        self.target_roles = None
        self.last_active = None

    def act(self, action_mask: np.ndarray) -> list[int]:
        active = getattr(self.env, "active", np.ones(self.env.cfg.num_uavs, dtype=bool)).copy()
        signature = tuple(bool(x) for x in active)
        if self.targets is None or signature != self.last_active:
            self._replan(active)
            self.last_active = signature

        actions = []
        for i in range(self.env.cfg.num_uavs):
            valid = np.flatnonzero(action_mask[i])
            if not active[i]:
                actions.append(int(valid[0]))
                continue
            move = self._move_toward(self.env.uav_positions[i], self.targets[i], action_mask[i], int(self.target_roles[i]))
            actions.append(self.env.encode_action(move, int(self.target_roles[i])))
        return actions

    def _replan(self, active: np.ndarray):
        positions = self.env.uav_positions.copy()
        roles = self.env.roles.copy()
        g = self.env.cfg.grid_size
        cells = np.asarray([(r, c) for r in range(g) for c in range(g)], dtype=np.int64)
        active_ids = np.flatnonzero(active)

        # Coordinate ascent over per-UAV (position, role) choices.  Start from
        # the actual current configuration, so replanning after a dropout or
        # rejoin is naturally local and travel-aware.
        for _ in range(self.sweeps):
            changed = False
            for i in active_ids:
                best_pos = positions[i].copy()
                best_role = int(roles[i])
                best_served = self._served_count(positions, roles, active)
                best_travel = 0

                for role in (ACCESS, RELAY):
                    for cell in cells:
                        old_pos = positions[i].copy(); old_role = int(roles[i])
                        positions[i] = cell; roles[i] = role
                        served = self._served_count(positions, roles, active)
                        travel = int(np.abs(cell - self.env.uav_positions[i]).sum())
                        score = (served, -travel, int(role == old_role))
                        best_score = (best_served, -best_travel, int(best_role == old_role))
                        if score > best_score:
                            best_served = served
                            best_travel = travel
                            best_pos = cell.copy()
                            best_role = role
                        positions[i] = old_pos; roles[i] = old_role

                if not np.array_equal(positions[i], best_pos) or int(roles[i]) != int(best_role):
                    changed = True
                positions[i] = best_pos
                roles[i] = best_role
            if not changed:
                break

        self.targets = positions.copy()
        self.target_roles = roles.copy()

    def _served_count(self, positions: np.ndarray, roles: np.ndarray, active: np.ndarray) -> int:
        access_ids = np.flatnonzero((roles == ACCESS) & active)
        relay_ids = np.flatnonzero((roles == RELAY) & active)
        valid_access = []
        for a in access_ids:
            apos = positions[a]
            direct = self.env._distance(apos, self.env.bs_position) <= self.env.cfg.backhaul_range
            via = False
            if not direct:
                for r in relay_ids:
                    rpos = positions[r]
                    if (
                        self.env._distance(apos, rpos) <= self.env.cfg.backhaul_range
                        and self.env._distance(rpos, self.env.bs_position) <= self.env.cfg.backhaul_range
                    ):
                        via = True; break
            if direct or via:
                valid_access.append(a)

        served = 0
        for ue in self.env.ue_positions:
            if any(self.env._distance(ue, positions[a]) <= self.env.cfg.access_range for a in valid_access):
                served += 1
        return served

    def _move_toward(self, current, target, mask, role: int) -> int:
        if np.array_equal(current, target):
            return 4
        dr = int(target[0] - current[0]); dc = int(target[1] - current[1])
        candidates=[]
        if dr < 0: candidates.append(0)
        elif dr > 0: candidates.append(1)
        if dc > 0: candidates.append(2)
        elif dc < 0: candidates.append(3)
        candidates.append(4)
        for move in candidates:
            if mask[self.env.encode_action(move, role)]:
                return move
        return 4
