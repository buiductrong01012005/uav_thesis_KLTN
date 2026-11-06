"""Vectorized multi-start coordinate-search policy for Stage-B recovery.

The planner searches static target (cell, role) assignments.  One coordinate
sweep optimizes one active UAV at a time while holding the other UAV targets
fixed.  Several inexpensive random restarts reduce local-optimum sensitivity.

The planner does not enumerate the full joint configuration space and does not
use learning.  It replans when fleet availability changes and then moves each
active UAV one grid step toward the selected target.
"""

from __future__ import annotations

import numpy as np

from src.envs.v0_env import ACCESS, RELAY


class MultiStartCoordinateGreedyPolicy:
    def __init__(self, env, sweeps: int = 3, restarts: int = 4, seed: int = 0):
        self.env = env
        self.sweeps = int(sweeps)
        self.restarts = int(restarts)
        if self.sweeps < 1 or self.restarts < 1:
            raise ValueError("sweeps and restarts must be >= 1")
        self.rng = np.random.default_rng(seed)
        g = env.cfg.grid_size
        self.cells = np.asarray([(r, c) for r in range(g) for c in range(g)], dtype=np.int64)
        # Candidate order: (cell_0, Access), (cell_0, Relay), ...
        self.candidate_positions = np.repeat(self.cells, 2, axis=0)
        self.candidate_roles = np.tile(np.asarray([ACCESS, RELAY], dtype=np.int64), len(self.cells))
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
            role = int(self.target_roles[i])
            move = self._move_toward(
                self.env.uav_positions[i], self.targets[i], action_mask[i], role
            )
            actions.append(self.env.encode_action(move, role))
        return actions

    def _batch_served_count(
        self, positions: np.ndarray, roles: np.ndarray, active: np.ndarray
    ) -> np.ndarray:
        """Return served-UE count for a batch of joint static configurations."""
        env = self.env
        batch, n_agents, _ = positions.shape
        bs = np.asarray(env.bs_position, dtype=np.float64)
        direct = (
            np.linalg.norm(positions - bs[None, None, :], axis=2) <= env.cfg.backhaul_range + 1e-9
        )
        valid_access = np.zeros((batch, n_agents), dtype=bool)

        for a in range(n_agents):
            if not active[a]:
                continue
            is_access = roles[:, a] == ACCESS
            via_relay = np.zeros(batch, dtype=bool)
            for r in range(n_agents):
                if r == a or not active[r]:
                    continue
                is_relay = roles[:, r] == RELAY
                a_to_r = (
                    np.linalg.norm(positions[:, a] - positions[:, r], axis=1)
                    <= env.cfg.backhaul_range + 1e-9
                )
                via_relay |= is_relay & direct[:, r] & a_to_r
            valid_access[:, a] = is_access & (direct[:, a] | via_relay)

        ue = np.asarray(env.ue_positions, dtype=np.float64)
        distance = np.linalg.norm(positions[:, None, :, :] - ue[None, :, None, :], axis=3)
        served = ((distance <= env.cfg.access_range + 1e-9) & valid_access[:, None, :]).any(axis=2)
        return served.sum(axis=1).astype(np.int64)

    def _coordinate_ascent(
        self,
        positions: np.ndarray,
        roles: np.ndarray,
        active: np.ndarray,
        actual_positions: np.ndarray,
    ):
        positions = positions.copy()
        roles = roles.copy()
        active_ids = np.flatnonzero(active)

        for _ in range(self.sweeps):
            changed = False
            for i in active_ids:
                n = len(self.candidate_roles)
                pos_batch = np.repeat(positions[None, :, :], n, axis=0)
                role_batch = np.repeat(roles[None, :], n, axis=0)
                pos_batch[:, i, :] = self.candidate_positions
                role_batch[:, i] = self.candidate_roles

                served = self._batch_served_count(pos_batch, role_batch, active)
                travel = np.abs(self.candidate_positions - actual_positions[i]).sum(axis=1)
                same_role = (self.candidate_roles == roles[i]).astype(np.int64)

                # Lexicographic maximum: served, -travel, keep-current-role.
                best = np.lexsort((same_role, -travel, served))[-1]
                new_pos = self.candidate_positions[best]
                new_role = int(self.candidate_roles[best])
                if not np.array_equal(new_pos, positions[i]) or new_role != int(roles[i]):
                    changed = True
                positions[i] = new_pos
                roles[i] = new_role
            if not changed:
                break

        served = int(self._batch_served_count(positions[None, :, :], roles[None, :], active)[0])
        travel = (
            int(np.abs(positions[active] - actual_positions[active]).sum()) if active.any() else 0
        )
        return positions, roles, served, travel

    def _replan(self, active: np.ndarray):
        actual = self.env.uav_positions.copy()
        base_roles = self.env.roles.copy()
        initializations = [(actual.copy(), base_roles.copy())]

        for _ in range(self.restarts - 1):
            positions = actual.copy()
            roles = base_roles.copy()
            for i in np.flatnonzero(active):
                positions[i] = self.cells[int(self.rng.integers(0, len(self.cells)))]
                roles[i] = int(self.rng.integers(0, 2))
            initializations.append((positions, roles))

        best = None
        for positions, roles in initializations:
            p, r, served, travel = self._coordinate_ascent(positions, roles, active, actual)
            score = (served, -travel)
            if best is None or score > best[0]:
                best = (score, p, r)

        self.targets = best[1].copy()
        self.target_roles = best[2].copy()

    def _move_toward(self, current, target, mask, role: int) -> int:
        if np.array_equal(current, target):
            return 4
        dr = int(target[0] - current[0])
        dc = int(target[1] - current[1])
        candidates = []
        if dr < 0:
            candidates.append(0)
        elif dr > 0:
            candidates.append(1)
        if dc > 0:
            candidates.append(2)
        elif dc < 0:
            candidates.append(3)
        candidates.append(4)
        for move in candidates:
            if mask[self.env.encode_action(move, role)]:
                return move
        return 4
